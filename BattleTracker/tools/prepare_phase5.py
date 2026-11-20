"""Freeze battle-disjoint training and sparse held-out review windows."""
import hashlib
import json
from pathlib import Path

from align_videos import prepare
from battletracker.alignment import check_grouping

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest_path = ROOT / "data/manifests/assets.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = manifest["assets"]
    review_path = ROOT / "data/annotations/dragon-tiger-review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    clean = next(item for item in assets if item["asset_id"] == review["clean_asset"])
    detailed = next(item for item in assets if item["asset_id"] == review["detailed_asset"])
    alignment, alignment_report, _, aligned_detailed, aligned_clean = prepare(
        ROOT / "data/annotations/dragon-tiger-alignment.json")
    if (aligned_clean["asset_id"], aligned_detailed["asset_id"]) != (
        clean["asset_id"], detailed["asset_id"]
    ):
        raise ValueError("review and alignment must reference the same validation recordings")
    for role, asset in (("clean", clean), ("detailed", detailed)):
        if (asset["battle_id"] != review["battle_id"] or asset["dataset_split"] != "validation"
                or asset["role"] != f"{role}_match_replay" or asset["path_base"] != "workspace"):
            raise ValueError("review must reference the registered validation recordings")
        with (ROOT / asset["path"]).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != asset["sha256"]:
                raise ValueError("validation recording hash mismatch")
    old = ROOT / "outputs/phase-03/clip_index.jsonl"
    rows = [json.loads(line) for line in old.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(row["dataset_split"] != "train" for row in rows):
        raise ValueError("rebuild the Phase 3 clip index with the current asset manifest")
    held_out = []
    annotation_digest = hashlib.sha256(review_path.read_bytes()).hexdigest()
    for kind, entries in (("reviewed", review["changes"]),
                          ("reviewed_hard_negative", review["hard_negatives"])):
        for entry in entries:
            row = dict(entry, annotation_status=kind, battle_id=review["battle_id"],
                       dataset_split="validation", source_video_id=clean["asset_id"],
                       source_sha256=clean["sha256"], source_path=clean["path"],
                       review_version=review["version"], owner_id=entry.get("owner_id"),
                       target_id=entry.get("target_id"),
                       review_note=entry["note"], reference_scope="sparse_reviewed_windows",
                       annotation_sha256=annotation_digest)
            if kind == "reviewed":
                mapped = [alignment.map(time) for time in entry["detailed_interval_ms"]]
                if None in mapped:
                    raise ValueError("reviewed change falls outside the verified alignment")
                row["official_reference"] = dict(asset_id=detailed["asset_id"],
                    source_sha256=detailed["sha256"], interval_ms=entry["detailed_interval_ms"],
                    before=entry["before"], after=entry["after"],
                    aligned_counter_interval_ms=mapped)
                row["alignment_version"] = alignment_report["version"]
            held_out.append(row)
    rows += held_out
    check_grouping(assets, rows)
    destination = ROOT / "outputs/phase-05"
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in (("grouped_index.jsonl", rows), ("validation_index.jsonl", held_out)):
        (destination / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                               for row in data), encoding="utf-8")
    print(json.dumps(dict(training_rows=len(rows)-len(held_out), validation_rows=len(held_out),
                         grouping="PASS")))


if __name__ == "__main__":
    main()
