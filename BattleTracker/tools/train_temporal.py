"""Compare a tiny temporal CNN and nearest-centroid classifier on grouped clips."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from battletracker.runtime import bootstrap_native_runtime

bootstrap_native_runtime()

import torch

from battletracker.model import LABELS, clip_features, metrics
from battletracker.temporal_models import (
    clip_sequence, export_cnn, fit_centroids, fit_cnn, predict, predict_centroids,
)
from train_model import ROOT, split_clips


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_data(items, assets):
    checked = set()
    sequences, features, targets, identities = [], [], [], []
    for row, target in items:
        asset = assets[row["source_video_id"]]
        base = {"workspace": ROOT, "user_profile": Path.home()}[asset["path_base"]]
        path = base / asset["path"]
        if row["source_sha256"] != asset["sha256"]:
            raise ValueError("clip hash differs from asset manifest")
        if path not in checked:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != asset["sha256"]:
                raise ValueError("source file hash mismatch")
            checked.add(path)
        sequences.append(clip_sequence(path, row["clip_start_ms"], row["clip_end_ms"]))
        features.append(clip_features(path, row["clip_start_ms"], row["clip_end_ms"]))
        targets.append(target)
        identities.append({key: row[key] for key in ("battle_id", "sample_id", "source_video_id")})
    return np.stack(sequences), np.stack(features), np.asarray(targets), identities


def score(probabilities, targets, identities, elapsed, threshold):
    guessed = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    return dict(metrics(targets, guessed), inference_ms_per_clip=elapsed * 1000 / len(targets),
                abstentions=int((confidence < threshold).sum()),
                errors=[dict(identity, expected=LABELS[truth],
                             predicted="UNKNOWN" if certainty < threshold else LABELS[guess],
                             confidence=float(certainty))
                        for identity, truth, guess, certainty in
                        zip(identities, targets, guessed, confidence, strict=True)
                        if guess != truth or certainty < threshold])


def export_centroids(model, features, destination, tolerance):
    import onnx
    import onnxruntime as ort

    center, scale, centroids = model
    h = onnx.helper
    # Negative squared distance has the same softmax as these linear logits.
    weight = (2 * centroids / scale).T / features.shape[1]
    bias = -(centroids ** 2).mean(axis=1) - center @ weight
    graph = h.make_graph([
        h.make_node("Gemm", ["features", "weight", "bias"], ["logits"]),
        h.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
    ], "temporal_nearest_centroid",
        [h.make_tensor_value_info("features", onnx.TensorProto.FLOAT, [None, features.shape[1]])],
        [h.make_tensor_value_info("probabilities", onnx.TensorProto.FLOAT, [None, 3])],
        [onnx.numpy_helper.from_array(weight.astype(np.float32), name="weight"),
         onnx.numpy_helper.from_array(bias.astype(np.float32), name="bias")])
    exported = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)], ir_version=8)
    onnx.save(exported, str(destination))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(destination), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    actual = session.run(None, {"features": features})[0]
    expected = predict_centroids(model, features)
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0)
    return dict(runtime_status="PASS", max_probability_error=float(np.abs(actual - expected).max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "outputs/phase-05/grouped_index.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/phase-05/temporal")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/phase-05.json")
    args = parser.parse_args()
    config = read_json(args.config)
    rows = [json.loads(line) for line in args.index.read_text(encoding="utf-8").splitlines() if line]
    assets = read_json(ROOT / "data/manifests/assets.json")["assets"]
    train, validation = split_clips(rows, assets, smoke=False)
    by_id = {asset["asset_id"]: asset for asset in assets}
    started = perf_counter()
    train_data, validation_data = load_data(train, by_id), load_data(validation, by_id)
    extraction_seconds = perf_counter() - started
    args.output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name in ("cnn", "centroid"):
        started = perf_counter()
        model = (fit_cnn(train_data[0], train_data[2], seed=config["seed"], steps=config["steps"],
                         rate=config["rate"], regularization=config["regularization"])
                 if name == "cnn" else fit_centroids(train_data[1], train_data[2]))
        training_seconds = perf_counter() - started
        report = dict(training_seconds=training_seconds, feature_extraction_seconds=extraction_seconds,
                      seed=config["seed"], steps=config["steps"] if name == "cnn" else 0,
                      optimizer="SGD" if name == "cnn" else "closed_form_class_means",
                      model_bytes=sum(p.numel() * p.element_size() for p in model.parameters())
                      if name == "cnn" else sum(p.nbytes for p in model))
        for split, data in (("train", train_data), ("validation", validation_data)):
            started = perf_counter()
            probabilities = predict(model, data[0]) if name == "cnn" else predict_centroids(model, data[1])
            elapsed = perf_counter() - started
            report[split] = score(probabilities, data[2], data[3], elapsed, config["minimum_confidence"])
            predictions = [dict(identity, probabilities=values.tolist())
                           for identity, values in zip(data[3], probabilities, strict=True)]
            (args.output / f"{name}-{split}-predictions.json").write_text(
                json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
        if name == "cnn":
            torch.save(model.state_dict(), args.output / "temporal.pt")
            sequences = np.concatenate((train_data[0], validation_data[0]))
            report["onnx_parity"] = export_cnn(model, sequences, args.output / "temporal.onnx", config["onnx_atol"])
        else:
            np.savez(args.output / "centroid.npz", center=model[0], scale=model[1], centroids=model[2])
            features = np.concatenate((train_data[1], validation_data[1]))
            report["onnx_parity"] = export_centroids(model, features, args.output / "centroid.onnx", config["onnx_atol"])
        reports[name] = report
    report = dict(config=config, labels=LABELS, candidates=reports,
                  clip_index_sha256=hashlib.sha256(args.index.read_bytes()).hexdigest(),
                  manifest_sha256=hashlib.sha256((ROOT / "data/manifests/assets.json").read_bytes()).hexdigest(),
                  code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               (Path(__file__), ROOT / "src/battletracker/temporal_models.py")},
                  train_battles=sorted({row["battle_id"] for row, _ in train}),
                  validation_battles=sorted({row["battle_id"] for row, _ in validation}),
                  preprocessing=dict(shape=[12, 3, 32, 72], resize=[96, 48], crop=[12, 7, 84, 39]))
    (args.output / "experiment-temporal.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: item["validation"] for name, item in reports.items()}))


if __name__ == "__main__":
    main()
