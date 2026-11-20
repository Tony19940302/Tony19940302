"""Compare Windows OCR and the existing counter reader on the same labeled crops."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
from unicodedata import normalize

from PIL import Image

from battletracker.counters import feature, sampled_regions
from extract_counters import ROOT, key_name, load_json, reference_labels


def parse_read(text, field):
    text = "".join(normalize("NFKC", text).split())
    if field == "round":
        return int(text) if re.fullmatch(r"[1-9][0-9]?", text) else None
    if not re.fullmatch(r"[0-9]{1,2}/[1-9][0-9]?", text):
        return None
    used, limit = map(int, text.split("/"))
    return f"{used}/{limit}" if used <= limit else None


def grade(actual, expected):
    if expected == "?":
        return "obscured" if actual is None else "read_on_obscured"
    if actual is None:
        return "unknown"
    return "correct" if actual == expected else "incorrect"


def compare(config_path, annotations_path, extraction_report, output):
    config, annotations = load_json(config_path), load_json(annotations_path)
    baseline = load_json(extraction_report)
    manifest = load_json(ROOT / "data/manifests/assets.json")
    asset = next(row for row in manifest["assets"] if row["asset_id"] == config["source_asset"])
    with (ROOT / asset["path"]).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if (asset["role"] != "detailed_match_replay" or digest != asset["sha256"]
            or baseline["source_sha256"] != digest or baseline["calibration"] != config["version"]):
        raise ValueError("OCR comparison requires the same calibrated source as the template report")
    _, expected = reference_labels(config, annotations)
    raw_records = {}
    with extraction_report.with_name("raw_reads.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            for time in row["requests"]:
                raw_records[time] = row
    crops = output / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    requests, rows = [], []
    for frame, regions in sampled_regions(ROOT / asset["path"], config, expected):
        for time in frame["requests"]:
            if raw_records[time]["pts"] != frame["pts"]:
                raise ValueError(f"template and OCR crops have different PTS at {time}")
            for key, correct in expected[time].items():
                owner, field = key
                name = key_name(owner, field)
                identifier = f"{time:g}-{name}"
                pixels = regions[key]
                rows.append(dict(id=identifier, field=field, expected=correct,
                                 template=raw_records[time]["reads"][name]["value"]))
                images = {"ocr_raw": Image.fromarray(pixels),
                          "ocr_segmented": Image.fromarray(255 - feature(pixels, field))}
                for method, image in images.items():
                    tag = f"{identifier}-{method}"
                    path = crops / f"{tag}.png"
                    image.resize((image.width * 3, image.height * 3)).save(path)
                    requests.append(dict(id=tag, path=str(path.resolve())))
    if len(rows) != sum(map(len, expected.values())):
        raise ValueError("missing OCR reference crops")
    index = output / "ocr_index.json"
    index.write_text(json.dumps(requests, ensure_ascii=False), encoding="utf-8")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-File",
                             str(ROOT / "tools/windows_ocr.ps1"), str(index.resolve())],
                            check=True, stdout=subprocess.PIPE, encoding="utf-8")
    ocr = json.loads(result.stdout)
    texts = {item["id"]: item["text"] for item in ocr["results"]}
    metrics = defaultdict(lambda: defaultdict(Counter))
    for row in rows:
        field = row["field"]
        row["ocr_text"] = {
            method: texts[f"{row['id']}-{method}"] for method in ("ocr_raw", "ocr_segmented")
        }
        for method, text in row["ocr_text"].items():
            row[method] = parse_read(text, field)
        for method in ("template", "ocr_raw", "ocr_segmented"):
            metrics[method][field][grade(row[method], row["expected"])] += 1
    return dict(source_asset=asset["asset_id"], language=ocr["language"],
                reference_frames=len(expected), total_regions=len(rows), metrics=metrics, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/phase-02/ocr-comparison")
    parser.add_argument("--extraction-report", type=Path,
                        default=ROOT / "outputs/phase-02/counter_extraction_report.json")
    args = parser.parse_args()
    report = compare(ROOT / "configs/counters/thunder-tiger.json",
                     ROOT / "data/annotations/thunder-tiger-counters.json",
                     args.extraction_report, args.output)
    (args.output / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
