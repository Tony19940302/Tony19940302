"""Apply confirmed child-entry labels to a matching offline prediction run."""

import argparse
from collections import Counter
import json
from pathlib import Path

from evaluate import read_jsonl
from predict_clean import ROOT, load_json, sha256


def apply_reviews(rows, summary, review, *, tolerance_ms):
    source = (review["source_video_id"], review["source_sha256"], review["battle_id"])
    if (summary["source_asset"], summary["source_sha256"]) != source[:2]:
        raise ValueError("review must refer to this ordinary replay")
    if any((row["source_asset"], row["source_sha256"], row["battle_id"]) != source for row in rows):
        raise ValueError("prediction rows do not share the reviewed replay")

    cases = [row for row in review["cases"] if row["event_type"] == "SUMMON_CHILD"
             and row["status"] == "reviewed"]
    if not cases:
        raise ValueError("no confirmed child entries in the review")
    if len({case["owner_id"] for case in cases}) != len(cases):
        raise ValueError("more than one confirmed child entry for a single owner")

    result = list(rows)
    applied = []
    for case in cases:
        media = ROOT / case["review_media"]
        if sha256(media) != case["review_media_sha256"]:
            raise ValueError(f"review clip does not match {case['sample_id']}")
        owner, entry = case["owner_id"], case["first_visible_entry_ms"]
        first, last = case["source_window_ms"]
        if not first <= entry <= last:
            raise ValueError("reviewed entry time is outside its clip")
        matches = [(i, row) for i, row in enumerate(result)
                   if row["owner_id"] == row["actor_id"] == owner
                   and first <= row["time_ms"] <= last
                   and abs(row["start_ms"] - entry) <= tolerance_ms
                   and (row["event_type"] == "SUMMON_PET"
                        or (row["event_type"] == "SUMMON_CHILD"
                            and row["evidence"].get("reviewed_case_id") == case["sample_id"]))]
        if len(matches) != 1:
            raise ValueError(f"expected one reviewed entry candidate for {case['sample_id']}, found {len(matches)}")
        i, candidate = matches[0]
        already_reviewed = candidate["event_type"] == "SUMMON_CHILD"
        arrival = candidate["evidence"].get("arrival_slot_id") if already_reviewed else candidate.get("target_id")
        if not arrival or not arrival.endswith("_PET") or arrival != case["arrival_slot_id"]:
            raise ValueError("candidate slot differs from the reviewed arrival slot")
        if candidate["event_type"] == "SUMMON_PET":
            evidence = dict(candidate["evidence"], reviewed_case_id=case["sample_id"],
                            original_event_type=candidate["event_type"],
                            arrival_slot_id=arrival, reviewed_entry_ms=entry)
            result[i] = dict(candidate, event_type="SUMMON_CHILD", target_id=f"{owner}_CHILD",
                             status="confirmed", unit_kind="child", source="manual_review",
                             evidence=evidence, payload=dict(candidate.get("payload", {}),
                                                             child_name=case["display_name"]))
            formation_slots = candidate.get("formation_slots", {})
            if "target" in formation_slots:
                evidence["arrival_formation_slot"] = formation_slots["target"]
                result[i]["formation_slots"] = {role: number for role, number in formation_slots.items()
                                                if role != "target"}
        elif (candidate.get("status") != "confirmed" or candidate["target_id"] != f"{owner}_CHILD"
              or candidate.get("payload", {}).get("child_name") != case["display_name"]):
            raise ValueError("existing child correction differs from the reviewed entry")
        applied.append(dict(sample_id=case["sample_id"], prediction_id=candidate["prediction_id"],
                            owner_id=owner, entry_ms=entry, display_name=case["display_name"]))
    return result, applied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--entry-tolerance-ms", type=int, default=200)
    args = parser.parse_args()
    if args.output.resolve() == args.predictions.parent.resolve():
        parser.error("use a separate output folder to retain the original predictions")
    if args.entry_tolerance_ms < 0:
        parser.error("entry tolerance must be nonnegative")
    summary = load_json(args.predictions.with_name("visual_summary.json"))
    rows, applied = apply_reviews(read_jsonl(args.predictions), summary, load_json(args.review),
                                  tolerance_ms=args.entry_tolerance_ms)
    summary.update(event_types=Counter(row["event_type"] for row in rows),
                   manual_child_replay={"reviews": applied, "review_sha256": sha256(args.review),
                                        "source_prediction_sha256": sha256(args.predictions)})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (args.output / "visual_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(applied, ensure_ascii=False))


if __name__ == "__main__":
    main()
