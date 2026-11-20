"""Build a clean-replay clip index with reviewed and candidate labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from align_videos import ANNOTATION, DESTINATION, ROOT, prepare, read_json, write_json
from battletracker.alignment import check_grouping

OFFICIAL = ROOT / "outputs/phase-02/official_events.jsonl"
REVIEW = ROOT / "data/annotations/thunder-tiger-review.json"


def build(alignment, alignment_report, assets, detailed, clean, events_path, reviews):
    source_report = json.loads(read_json(events_path.with_name("counter_extraction_report.json")))
    source_asset = source_report["source_asset"]
    if (detailed["role"] != "detailed_match_replay" or source_asset != detailed["asset_id"]
            or source_report["source_sha256"] != detailed["sha256"]
            or source_report["source_sha256"] != alignment_report["sources"].get(source_asset)
            or source_report["battle_id"] != clean["battle_id"]):
        raise ValueError("official counter report must identify the paired detailed recording")
    event_bytes = events_path.read_bytes()
    digest = hashlib.sha256(event_bytes).hexdigest()
    if digest != source_report.get("official_events_sha256"):
        raise ValueError("official event log hash does not match its extraction report")
    text = event_bytes.decode("utf-8")
    events = [(number, json.loads(line)) for number, line in enumerate(text.splitlines(), 1)
              if line.strip()]
    if not events:
        raise ValueError("official events input is empty")
    common = dict(battle_id=clean["battle_id"], source_video_id=clean["asset_id"],
                  source_sha256=clean["sha256"], dataset_split=clean["dataset_split"],
                  alignment_version=alignment_report["version"], source_path=clean["path"],
                  review_version=reviews["version"])
    samples, unmatched = [], []
    seen_changes = set()
    for line, event in events:
        if event["battle_id"] != common["battle_id"]:
            raise ValueError("official event battle identity mismatch")
        owner, kind = event["owner_id"], event["event_type"]
        earliest, latest = event["earliest_timestamp_ms"], event["latest_timestamp_ms"]
        if not 0 <= earliest < latest:
            raise ValueError("invalid official event interval")
        change_id = f"{owner}:{latest}:{kind}"
        if change_id in seen_changes:
            raise ValueError("duplicate official change ID")
        seen_changes.add(change_id)
        start, end = alignment.map(earliest), alignment.map(latest)
        provenance = dict(asset_id=source_asset, source_sha256=source_report["source_sha256"],
                          line=line, change_id=change_id, events_file=events_path.name, events_sha256=digest,
                          detailed_interval_ms=[earliest, latest])
        if start is None or end is None:
            unmatched.append(dict(event=event, official_reference=provenance,
                                  reason="outside_verified_overlap"))
            continue
        review = reviews["changes"].get(change_id, {})
        actor = review.get("actor_id")
        if actor and (review.get("kind"), actor) not in {("player", owner), ("pet", f"{owner}_PET")}:
            raise ValueError("reviewed actor must agree with screen-slot owner")
        if actor and (kind not in {"USE_MEDICINE", "SUMMON_PET"} or
                      (kind == "SUMMON_PET" and review["kind"] != "player") or
                      event["delta"] != 1 or event["needs_review"]):
            raise ValueError("review cannot assign a single actor to an unresolved counter change")
        round_start, round_end = event["earliest_round"], event["latest_round"]
        samples.append(dict(
            common, sample_id=change_id, round=round_end if round_start == round_end else None,
            round_bounds=[round_start, round_end], event_type=kind, owner_id=owner, actor_id=actor,
            actor_kind=review.get("kind"), target_id=review.get("target_id"),
            clip_start_ms=max(alignment.anchors[0].clean_ms, start - 1100),
            clip_end_ms=min(alignment.anchors[-1].clean_ms, end + 500),
            mapped_counter_interval_ms=[start, end],
            official_counter_before=event["before"], official_counter_after=event["after"],
            official_reference=provenance, annotation_status="reviewed" if actor else "candidate",
            alignment_confidence="manual_anchor_interpolation",
            review_note=review.get("note"), reference_delta=event["delta"],
        ))
    mapped_ids = {row["sample_id"] for row in samples}
    if set(reviews["changes"]) - mapped_ids:
        raise ValueError("actor review references a change outside the mapped event set")

    for negative in reviews["hard_negatives"]:
        start, end = negative["start_ms"], negative["end_ms"]
        if not alignment.anchors[0].clean_ms <= start < end <= alignment.anchors[-1].clean_ms:
            raise ValueError("hard negative is outside the verified overlap")
        if any(start < last and first < end for first, last in
               (sample["mapped_counter_interval_ms"] for sample in samples
                if "mapped_counter_interval_ms" in sample)):
            raise ValueError("hard negative overlaps an observed official change")
        samples.append(dict(common, sample_id=negative["id"], round=negative["round"],
                            event_type=negative["event_type"], owner_id=None,
                            actor_id=negative["actor_id"], target_id=None, clip_start_ms=start,
                            clip_end_ms=end, annotation_status="reviewed_hard_negative",
                            alignment_confidence="direct_clean_video_annotation",
                            review_note=negative["cue"]))

    if len({row["sample_id"] for row in samples}) != len(samples):
        raise ValueError("duplicate clip index ID")
    groups = check_grouping(assets, samples)
    report = dict(battle_id=common["battle_id"], source_events=len(events),
                  matched_changes=len(events) - len(unmatched), unmatched_changes=len(unmatched),
                  label_status=dict(Counter(row["annotation_status"] for row in samples)),
                  dataset_groups=groups, development_battles=sum(split == "development"
                                                                for split in groups.values()),
                  data_leakage_check="PASS")
    return samples, unmatched, report


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", type=Path, default=ANNOTATION)
    parser.add_argument("--events", type=Path, default=OFFICIAL)
    parser.add_argument("--review", type=Path, default=REVIEW)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    parser.add_argument("--export-clips", action="store_true")
    args = parser.parse_args()
    alignment, report, assets, detailed, clean = prepare(args.alignment)
    samples, unmatched, summary = build(alignment, report, assets, detailed, clean, args.events,
                                        json.loads(read_json(args.review)))
    args.output.mkdir(parents=True, exist_ok=True)
    if args.export_clips:
        from battletracker.review_clips import export_review_clips

        export_review_clips(ROOT / clean["path"], samples, args.output)
        summary["exported_clips"] = len(samples)
    write_jsonl(args.output / "clip_index.jsonl", samples)
    write_jsonl(args.output / "unmatched_changes.jsonl", unmatched)
    write_json(args.output / "dataset_report.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()


