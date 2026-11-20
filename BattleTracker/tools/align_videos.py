"""Validate battle replay anchors and create an alignment report."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from statistics import median

from battletracker.alignment import Anchor, TimeAlignment

ROOT = Path(__file__).resolve().parents[1]
ANNOTATION = ROOT / "data/annotations/thunder-tiger-alignment.json"
DESTINATION = ROOT / "outputs/phase-03"


def read_json(path):
    return path.read_text(encoding="utf-8")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_pair(annotation):
    config = json.loads(read_json(annotation))
    assets = json.loads(read_json(ROOT / "data/manifests/assets.json"))["assets"]
    asset_map = {row["asset_id"]: row for row in assets}
    detailed, clean = (asset_map[config[f"{role}_asset_id"]] for role in ("detailed", "clean"))
    if not detailed["battle_id"] or detailed["battle_id"] != clean["battle_id"]:
        raise ValueError("paired replays must share a battle identity")
    for asset, role in ((detailed, "detailed"), (clean, "clean")):
        if asset["role"] != f"{role}_match_replay" or asset["path_base"] != "workspace":
            raise ValueError(f"invalid {role} replay in manifest")
        with (ROOT / asset["path"]).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != asset["sha256"]:
                raise ValueError(f"source hash mismatch for {asset['asset_id']}")
    return config, assets, detailed, clean


def prepare(annotation):
    config, assets, detailed, clean = load_pair(annotation)
    alignment = TimeAlignment(Anchor(**row) for row in config["anchors"])
    checks = tuple(Anchor(**row) for row in config["checkpoints"])
    for asset, name in ((detailed, "detailed_ms"), (clean, "clean_ms")):
        duration_ms = asset["inspection"]["stream_duration_seconds"] * 1000
        if any(not 0 <= getattr(item, name) < duration_ms
               for item in (*alignment.anchors, *checks)):
            raise ValueError("anchor or checkpoint exceeds source duration")
    if not checks:
        raise ValueError("alignment requires at least one checkpoint")
    report_rows = alignment.validate(checks)
    errors = [abs(row["error_ms"]) for row in report_rows]
    tolerance = config["review_tolerance_ms"]
    if max(errors) > tolerance:
        raise ValueError(f"checkpoint error {max(errors)} ms exceeds {tolerance} ms")
    report = dict(version=config["version"], timestamp_basis=config["timestamp_basis"],
                  battle_id=clean["battle_id"],
                  sources={row["asset_id"]: row["sha256"] for row in (detailed, clean)},
                  tolerance_ms=tolerance, max_abs_error_ms=max(errors),
                  median_abs_error_ms=median(errors), anchors=[asdict(a) for a in alignment.anchors],
                  checkpoints=report_rows)
    return alignment, report, assets, detailed, clean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=ANNOTATION)
    parser.add_argument("--output", type=Path, default=DESTINATION / "alignment_report.json")
    args = parser.parse_args()
    _, report, _, _, _ = prepare(args.annotations)
    write_json(args.output, report)
    print(f"{len(report['anchors'])} fitted anchors, {len(report['checkpoints'])} checkpoints,"
          f" maximum error {report['max_abs_error_ms']} ms")


if __name__ == "__main__":
    main()
