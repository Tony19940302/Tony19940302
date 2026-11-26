"""Evaluate clean predictions against the Phase 3 reviewed event intervals."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median

from battletracker.battle import MedicineConsumption
from predict_clean import OUTPUT, ROOT

RESOURCE_TYPES = {"USE_MEDICINE", "SUMMON_PET"}


def prediction_consumption(row):
    if row["event_type"] == "USE_MEDICINE":
        return MedicineConsumption(row.get("payload", {}).get("medicine_consumption", MedicineConsumption.CHARGED))
    return MedicineConsumption.CHARGED


def distance(time, bounds):
    start, end = bounds
    return max(start - time, 0, time - end)


def match_events(predictions, references, tolerance, *, require_same_owner=False):
    """Maximum one-to-one temporal assignment by type, optionally constrained by owner."""
    edges = []
    for ref in references:
        bounds = ref["mapped_counter_interval_ms"]
        options = [i for i, pred in enumerate(predictions)
                   if pred["event_type"] == ref["event_type"]
                   and (not require_same_owner or (ref["owner_id"] is not None
                                                    and pred.get("owner_id") == ref["owner_id"]))
                   and distance(pred["time_ms"], bounds) <= tolerance]
        edges.append(sorted(options, key=lambda i: (
            distance(predictions[i]["time_ms"], bounds),
            abs(predictions[i]["time_ms"] - sum(bounds) / 2),
            str(predictions[i]["prediction_id"]),
        )))
    assigned = {}

    def place(ref_index, seen):
        for index in edges[ref_index]:
            if index in seen:
                continue
            seen.add(index)
            if index not in assigned or place(assigned[index], seen):
                assigned[index] = ref_index
                return True
        return False

    for ref_index in sorted(range(len(references)), key=lambda i: len(edges[i])):
        place(ref_index, set())
    return assigned


def evaluate(predictions, clips, tolerance):
    clean_sources = {(row["battle_id"], row["source_video_id"], row["source_sha256"])
                     for row in clips}
    if len(clean_sources) != 1:
        raise ValueError("expected predictions and references for a single clean replay")
    if len({row["prediction_id"] for row in predictions}) != len(predictions):
        raise ValueError("duplicate prediction IDs")
    for row in predictions:
        if (row["battle_id"], row["source_asset"], row["source_sha256"]) not in clean_sources:
            raise ValueError("predictions came from a different replay")
    reviewed = [row for row in clips if row["annotation_status"] == "reviewed"
                and row["event_type"] in RESOURCE_TYPES]
    if not reviewed:
        raise ValueError("no reviewed resource references")
    negatives = [row for row in clips if row["annotation_status"] == "reviewed_hard_negative"]
    unknown_refs = [row for row in clips if row["annotation_status"] == "candidate"]
    resources = [row for row in predictions if row["event_type"] in RESOURCE_TYPES]
    candidates = [row for row in resources if prediction_consumption(row) == MedicineConsumption.CHARGED]
    matches = match_events(candidates, reviewed, tolerance)
    unverified = [dict(prediction=row, reference=None) for row in resources
                  if prediction_consumption(row) == MedicineConsumption.UNKNOWN]
    false_positives, attributions, paired = [], [], []
    misses = [ref for i, ref in enumerate(reviewed) if i not in matches.values()]
    for i, pred in enumerate(candidates):
        time = pred["time_ms"]
        if i in matches:
            ref = reviewed[matches[i]]
            bounds = ref["mapped_counter_interval_ms"]
            item = dict(prediction=pred, reference=ref, distance_to_interval_ms=distance(time, bounds),
                        offset_from_interval_midpoint_ms=time - sum(bounds) / 2,
                        owner_correct=pred["owner_id"] == ref["owner_id"],
                        actor_correct=(pred["actor_id"] == ref["actor_id"]
                                       if ref["actor_id"] is not None else None),
                        target_correct=(pred["target_id"] == ref["target_id"]
                                        if ref["target_id"] is not None else None))
            paired.append(item)
            if (not item["owner_correct"] or item["actor_correct"] is False
                    or item["target_correct"] is False):
                attributions.append(item)
            continue
        negative = next((row for row in negatives if row["clip_start_ms"] <= time <= row["clip_end_ms"]), None)
        pending = next((row for row in unknown_refs if distance(time, row["mapped_counter_interval_ms"])
                        <= tolerance), None)
        nearby = next((row for row in reviewed if distance(time, row["mapped_counter_interval_ms"])
                       <= tolerance), None)
        if pending is None and (negative or nearby):
            false_positives.append(dict(prediction=pred, reference=negative or nearby,
                                        reason="hard_negative" if negative else "extra_or_wrong_type"))
        else:
            unverified.append(dict(prediction=pred, reference=pending))
    metrics = {}
    for kind in sorted(RESOURCE_TYPES):
        hits = [row for row in paired if row["prediction"]["event_type"] == kind]
        fp = sum(row["prediction"]["event_type"] == kind for row in false_positives)
        fn = sum(row["event_type"] == kind for row in misses)
        metrics[kind] = dict(reviewed_hits=len(hits), scoped_false_positives=fp, misses=fn,
                             reviewed_recall=len(hits) / (len(hits) + fn) if hits or fn else None,
                             scoped_precision=len(hits) / (len(hits) + fp) if hits or fp else None,
                             correct_owner=sum(row["owner_correct"] for row in hits),
                             actors_with_reference=sum(row["actor_correct"] is not None for row in hits),
                             correct_actor=sum(row["actor_correct"] is True for row in hits),
                             targets_with_reference=sum(row["target_correct"] is not None for row in hits),
                             correct_target=sum(row["target_correct"] is True for row in hits),
                             median_interval_distance_ms=median(row["distance_to_interval_ms"] for row in hits)
                             if hits else None)
    return dict(tolerance_ms=tolerance, source=sorted(clean_sources)[0], metrics=metrics,
                reviewed_reference_count=len(reviewed), hard_negative_count=len(negatives),
                unverified_prediction_count=len(unverified),
                prediction_types=dict(Counter(row["event_type"] for row in predictions)),
                matches=paired, attribution_errors=attributions, misses=misses,
                scoped_false_positives=false_positives, unverified_predictions=unverified)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=OUTPUT / "predictions.jsonl")
    parser.add_argument("--references", type=Path, default=ROOT / "outputs/phase-03/clip_index.jsonl")
    parser.add_argument("--output", type=Path, default=OUTPUT / "evaluation.json")
    parser.add_argument("--tolerance-ms", type=int, default=500)
    args = parser.parse_args()
    if args.tolerance_ms < 0:
        parser.error("tolerance must be nonnegative")
    report = evaluate(read_jsonl(args.predictions), read_jsonl(args.references), args.tolerance_ms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failures = []
    for category in ("scoped_false_positives", "attribution_errors", "misses"):
        for row in report[category]:
            ref = row.get("reference", row)
            pred = row.get("prediction")
            failures.append(dict(category=category, sample_id=ref["sample_id"],
                                 source_path=ref["source_path"], review_media=ref.get("review_media"),
                                 review_start_ms=ref["clip_start_ms"], review_end_ms=ref["clip_end_ms"],
                                 predicted_time_ms=pred["time_ms"] if pred else None,
                                 prediction_id=pred["prediction_id"] if pred else None))
    args.output.with_name("review_cases.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures), encoding="utf-8"
    )
    print(json.dumps(report["metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()



