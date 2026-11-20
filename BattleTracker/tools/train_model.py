"""Train a clip classifier on reviewed clean-replay samples."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from battletracker.alignment import check_grouping
from battletracker.model import LABELS, FRAME_COUNT, WIDTH, HEIGHT
from battletracker.model import clip_features, export_and_compare, fit, metrics, softmax

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def split_clips(rows, assets, smoke):
    check_grouping(assets, rows)
    identities = {(row["battle_id"], row["sample_id"]) for row in rows}
    if len(identities) != len(rows):
        raise ValueError("duplicate samples in clip index")
    accepted = []
    for row in rows:
        status, kind = row["annotation_status"], row["event_type"]
        if status == "reviewed" and kind in LABELS[1:]:
            accepted.append((row, LABELS.index(kind)))
        elif status == "reviewed_hard_negative" and (
            kind.startswith("NON_MEDICINE_") or kind == "PET_REVIVAL"
        ):
            accepted.append((row, 0))
    if smoke:
        development = [item for item in accepted
                       if item[0]["dataset_split"] in {"development", "train"}]
        train = development
        validation = []
    else:
        train = [item for item in accepted if item[0]["dataset_split"] == "train"]
        validation = [item for item in accepted if item[0]["dataset_split"] == "validation"]
        if not train or not validation:
            raise ValueError("comparison requires reviewed train and validation battles in separate groups")
    if {target for _, target in train} != set(range(len(LABELS))):
        raise ValueError("training data must contain all configured classes")
    return train, validation


def dataset_features(items, assets):
    checked = {}
    features, labels, ids = [], [], []
    for row, target in items:
        item = assets[row["source_video_id"]]
        if item["path_base"] != "workspace" or item["sha256"] != row["source_sha256"]:
            raise ValueError("sample source must match the declared clean replay")
        path = ROOT / item["path"]
        if path not in checked:
            with path.open("rb") as stream:
                checked[path] = hashlib.file_digest(stream, "sha256").hexdigest()
        if checked[path] != item["sha256"]:
            raise ValueError("clean replay hash mismatch")
        features.append(clip_features(path, row["clip_start_ms"], row["clip_end_ms"]))
        labels.append(target)
        ids.append(dict(battle_id=row["battle_id"], sample_id=row["sample_id"],
                        source_video_id=row["source_video_id"],
                        interval_ms=[row["clip_start_ms"], row["clip_end_ms"]]))
    return np.stack(features), np.asarray(labels), ids


def score_split(model, data, minimum_confidence):
    features, expected, ids = data
    weight, bias = model
    start = perf_counter()
    probabilities = softmax(features @ weight + bias)
    inference_ms = (perf_counter() - start) * 1000 / len(features)
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    raw_metrics = metrics(expected, predicted)
    abstained = confidence < minimum_confidence
    errors = [dict(identity, expected=LABELS[truth], predicted="UNKNOWN" if unknown else LABELS[guess],
                   confidence=float(probability))
              for identity, truth, guess, unknown, probability in
              zip(ids, expected, predicted, abstained, confidence, strict=True)
              if unknown or guess != truth]
    return dict(raw_metrics, count=len(ids), labels=dict(Counter(LABELS[i] for i in expected)),
                abstentions=int(abstained.sum()), inference_ms_per_clip=inference_ms, errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "outputs/phase-03/clip_index.jsonl")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/phase-05.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/phase-05")
    parser.add_argument("--smoke", action="store_true", help="Fit development data; report fit only.")
    parser.add_argument("--require-runtime", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    if (config["steps"] <= 0 or config["rate"] <= 0 or config["regularization"] < 0
            or not 0 <= config["minimum_confidence"] <= 1 or config["onnx_atol"] <= 0):
        parser.error("invalid experiment training budget or thresholds")
    manifest_path = ROOT / "data/manifests/assets.json"
    assets = read_json(manifest_path)["assets"]
    raw_index = args.index.read_bytes()
    rows = [json.loads(line) for line in raw_index.splitlines() if line.strip()]
    train, validation = split_clips(rows, assets, args.smoke)
    by_id = {item["asset_id"]: item for item in assets}
    train_data = dataset_features(train, by_id)
    eval_data = dataset_features(validation, by_id) if validation else None

    start = perf_counter()
    model = fit(*train_data[:2], **{key: config[key] for key in
                                  ("seed", "steps", "rate", "regularization")})
    train_seconds = perf_counter() - start
    args.output.mkdir(parents=True, exist_ok=True)
    weight, bias = model
    np.savez(args.output / "model.npz", weight=weight, bias=bias)
    parity_data = np.concatenate([train_data[0], eval_data[0]]) if eval_data else train_data[0]
    parity = export_and_compare(args.output / "model.onnx", parity_data, weight, bias,
                                      config["onnx_atol"])
    code_files = [Path(__file__), ROOT / "src/battletracker/model.py",
                  ROOT / "src/battletracker/alignment.py"]
    report = dict(
        mode="development_smoke" if args.smoke else "held_out_validation",
        config=config, labels=LABELS, clip_index_sha256=hashlib.sha256(raw_index).hexdigest(),
        asset_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        code_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in code_files},
        preprocessing=dict(frame_count=FRAME_COUNT, resize=[WIDTH, HEIGHT],
                           crop_xyxy=[12, 7, 84, 39], grid=[12, 6], temporal_bins=3),
        onnx_input_shape=[None, weight.shape[0]], model_bytes=weight.nbytes + bias.nbytes,
        onnx_file_bytes=(args.output / "model.onnx").stat().st_size,
        output_probabilities="softmax; below minimum_confidence yields UNKNOWN",
        numpy_version=np.__version__,
        onnx_parity=parity, training_seconds=train_seconds,
        train_battles=sorted({row["battle_id"] for row, _ in train}),
        validation_battles=sorted({row["battle_id"] for row, _ in validation}),
        fit=score_split(model, train_data, config["minimum_confidence"]),
        validation=score_split(model, eval_data, config["minimum_confidence"]) if eval_data else None,
    )
    (args.output / "experiment.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"mode": report["mode"], "clips": report["fit"]["count"],
                      "onnx_parity": parity,
                      "validation_battles": report["validation_battles"]}))
    if args.require_runtime and parity["runtime_status"] != "PASS":
        raise SystemExit("Required ONNX Runtime parity did not pass")


if __name__ == "__main__":
    main()


