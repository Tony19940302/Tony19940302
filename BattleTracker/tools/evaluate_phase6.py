"""Evaluate reviewed events and counter checkpoints within a paired replay."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median

from PIL import Image

from align_videos import ROOT, prepare, write_json
from battletracker.alignment import check_grouping
from battletracker.battle import MedicineConsumption
from battletracker.vision import frames
from evaluate import RESOURCE_TYPES, match_events, prediction_consumption, read_jsonl


OUTPUT = ROOT / "outputs/phase-06"
COUNTER_EVENTS = {"medicine": "USE_MEDICINE", "pet": "SUMMON_PET"}
EVENT_CLASSES = ("PLAYER_USE_MEDICINE", "PET_USE_MEDICINE", "SUMMON_PET")


def event_class(row):
    if row["event_type"] == "SUMMON_PET":
        return "SUMMON_PET"
    actor = row.get("actor_id")
    if row["event_type"] == "USE_MEDICINE" and actor is not None:
        return "PET_USE_MEDICINE" if actor.endswith("_PET") else "PLAYER_USE_MEDICINE"
    return None


def counter_used(value):
    if value == "?":
        return None
    used, limit = map(int, value.split("/"))
    if not 0 <= used <= limit:
        raise ValueError("invalid manually reviewed counter values")
    return used


def checkpoint_metrics(predictions, rows, owners, alignment, alignment_error_ms, round_ends):
    """Measure interval count error from the first manually readable checkpoint."""
    checkpoints = []
    seen_round_end = set()
    for seconds, round_number, *values in rows:
        detailed_ms = round(seconds * 1000)
        clean_ms = alignment.map(detailed_ms)
        if clean_ms is None:
            continue
        if len(values) != len(owners):
            raise ValueError("counter checkpoint must list all configured owners")
        round_end = seconds in round_ends
        if round_end and round_number in seen_round_end:
            raise ValueError("duplicate round-end checkpoint")
        if round_end:
            seen_round_end.add(round_number)
        counts = {}
        for owner, pair in zip(owners, values, strict=True):
            if owner in counts:
                raise ValueError("duplicate owner in calibration")
            counts[owner] = dict(zip(COUNTER_EVENTS, map(counter_used, pair.split()), strict=True))
        checkpoints.append(dict(round=round_number, detailed_ms=detailed_ms, clean_ms=clean_ms,
                                round_end=round_end, counts=counts))
    checkpoints.sort(key=lambda c: c["clean_ms"])
    if len({c["clean_ms"] for c in checkpoints}) != len(checkpoints):
        raise ValueError("duplicate counter checkpoint time")

    charged = [row for row in predictions if prediction_consumption(row) == MedicineConsumption.CHARGED]
    unresolved = [row for row in predictions if prediction_consumption(row) == MedicineConsumption.UNKNOWN]

    def automatic_count(owner, event_type, start, end):
        return sum(row["event_type"] == event_type and row["owner_id"] == owner
                   and start < row["time_ms"] <= end for row in charged)

    def summarize(fields):
        pending = sum(f["error"] is None for f in fields)
        fields = [f for f in fields if f["error"] is not None]
        count = len(fields)
        result = dict(compared=count, matching=sum(f["error"] == 0 for f in fields),
                      exact_rate=sum(f["error"] == 0 for f in fields) / count if count else None,
                      absolute_error_sum=sum(abs(f["error"]) for f in fields) if count else None,
                      max_absolute_error=max((abs(f["error"]) for f in fields), default=None),
                      timing_sensitive=sum(f["timing_sensitive"] for f in fields))
        if pending:
            result["unresolved_consumption"] = pending
        return result

    if len(checkpoints) < 2:
        return dict(status="unverified", reason="fewer than two aligned manual checkpoints")
    initial, *later = checkpoints
    baseline_ms = initial["clean_ms"]
    latest_known = {owner: values.copy() for owner, values in initial["counts"].items()}
    details = []
    for checkpoint in later:
        end = checkpoint["clean_ms"]
        fields = []
        for owner, before in initial["counts"].items():
            for field, event_type in COUNTER_EVENTS.items():
                start_used = before[field]
                official_used = checkpoint["counts"][owner][field]
                if official_used is None:
                    continue
                prior = latest_known[owner][field]
                if prior is not None and official_used < prior:
                    raise ValueError("manual checkpoint counter regresses")
                latest_known[owner][field] = official_used
                if start_used is None:
                    continue
                guess = start_used + automatic_count(owner, event_type, baseline_ms, end)
                lower = start_used + automatic_count(owner, event_type,
                    baseline_ms + alignment_error_ms, end - alignment_error_ms)
                upper = start_used + automatic_count(owner, event_type,
                    baseline_ms - alignment_error_ms, end + alignment_error_ms)
                pending = [] if event_type != "USE_MEDICINE" else [
                    row["prediction_id"] for row in unresolved if row["owner_id"] == owner
                    and baseline_ms - alignment_error_ms < row["time_ms"] <= end + alignment_error_ms
                ]
                result = dict(owner_id=owner, field=field, initial_used=start_used,
                              official_used=official_used, candidate_used=guess,
                              error=None if pending else guess - official_used, timing_sensitive=lower != upper,
                              candidate_used_at_alignment_bounds=[lower, upper])
                if pending:
                    result["unresolved_prediction_ids"] = pending
                fields.append(result)
        details.append(dict(checkpoint, fields=fields, summary=summarize(fields)))
    return dict(status="partial_replay", baseline=initial, checkpoints=details,
                empirical_alignment_margin_ms=alignment_error_ms,
                all_checkpoints=summarize([f for row in details for f in row["fields"]]),
                round_end_only=summarize([f for row in details if row["round_end"]
                                          for f in row["fields"]]),
                last_checkpoint=details[-1]["summary"], final_battle_counts=None)


def completion_metrics(completion, counts, references, tolerance, *, counter_only_changes=()):
    """Check terminal counts separately from complete event-level coverage."""
    initial = counts.get("baseline")
    checkpoints = counts.get("checkpoints", [])
    if completion is None or not initial or not checkpoints:
        return None
    terminal = checkpoints[-1]
    if (completion["opening_checkpoint_ms"] != initial["detailed_ms"]
            or completion["final_checkpoint_ms"] != terminal["detailed_ms"]
            or initial["round"] != 1 or not terminal["round_end"]):
        raise ValueError("complete replay requires opening and final-round-end checkpoints")
    window = [initial["clean_ms"], terminal["clean_ms"]]
    all_terminal_fields = len(terminal["fields"]) == len(initial["counts"]) * len(COUNTER_EVENTS)
    expected = {(field["owner_id"], COUNTER_EVENTS[field["field"]]):
                list(range(field["initial_used"] + 1, field["official_used"] + 1))
                for field in terminal["fields"]}
    observed = {key: [] for key in expected}
    by_counter = {key: [] for key in expected}
    localized = True
    changes = sorted([*references, *counter_only_changes], key=lambda row: row["mapped_counter_interval_ms"])
    for row in changes:
        start, end = row["mapped_counter_interval_ms"]
        if not window[0] < start <= end <= window[1]:
            raise ValueError("reviewed counter change falls outside the opening and final checkpoints")
        localized &= end - start <= 2 * tolerance
        key = (row["owner_id"], row["event_type"])
        observed.setdefault(key, []).append(counter_used(row["after"]))
        by_counter.setdefault(key, []).append(row["detailed_interval_ms"])
    round_ends = [checkpoint for checkpoint in checkpoints if checkpoint["round_end"]]
    conflicts = []
    for checkpoint in round_ends:
        time = checkpoint["detailed_ms"]
        for field in checkpoint["fields"]:
            key = (field["owner_id"], COUNTER_EVENTS[field["field"]])
            intervals = by_counter.get(key, [])
            lower = sum(end <= time for start, end in intervals)
            upper = sum(start < time for start, end in intervals)
            delta = field["official_used"] - field["initial_used"]
            if not lower <= delta <= upper:
                conflicts.append(dict(round=checkpoint["round"], owner_id=key[0],
                                      event_type=key[1], official_delta=delta,
                                      reviewed_delta_bounds=[lower, upper]))
    return dict(terminal_counts_complete=all_terminal_fields, complete_event_reference=(
                    all_terminal_fields and localized and observed == expected and not conflicts
                    and not counter_only_changes),
                checkpoint_conflicts=conflicts,
                complete_round_end_checkpoints=(
                    {c["round"] for c in round_ends} == set(range(1, terminal["round"] + 1))
                    and all(len(c["fields"]) == len(initial["counts"]) * len(COUNTER_EVENTS)
                            for c in round_ends)),
                reviewed_combat_window_ms=window,
                final_round=terminal["round"],
                settlement_evidence=completion["evidence"])


def misclassified_actor(action):
    """Scope a reviewed false positive to the actor reported by the detector."""
    return action.get("misclassified_actor_id", action.get("actor_id"))


def event_metrics(predictions, references, empty_windows, overlap, tolerance, *, complete_window=None,
                  non_resource_actions=()):
    left, right = overlap
    if complete_window is None:
        scoped = [row for row in predictions if left <= row["time_ms"] <= right]
    else:
        opened, finished = complete_window
        if not left <= opened < finished <= right:
            raise ValueError("complete reviewed combat window lies outside the aligned overlap")
        scoped = [row for row in predictions if opened < row["time_ms"] <= finished]
    resources = [row for row in scoped if row["event_type"] in RESOURCE_TYPES]
    candidates = [row for row in resources if prediction_consumption(row) == MedicineConsumption.CHARGED]
    if len({row["sample_id"] for row in references}) != len(references):
        raise ValueError("duplicate reviewed event ID")
    for ref in references:
        start, end = ref["mapped_counter_interval_ms"]
        if not left <= start <= end <= right:
            raise ValueError("reviewed event lies outside the aligned overlap")
    for start, end in empty_windows:
        if not left <= start < end <= right:
            raise ValueError("verified empty window lies outside the aligned overlap")
        if any(start <= b and a <= end for a, b in
               (r["mapped_counter_interval_ms"] for r in references)):
            raise ValueError("verified empty window overlaps a reviewed resource event")
    for action in non_resource_actions:
        if (not left <= action["clip_start_ms"] <= action["clip_end_ms"] <= right
                or action["misclassified_as"] not in RESOURCE_TYPES or not misclassified_actor(action)):
            raise ValueError("invalid reviewed non-resource action")

    false_positives, eligible = [], []
    for pred in candidates:
        known_non_resource = any(
            pred.get("actor_id") == misclassified_actor(action)
            and pred["event_type"] == action["misclassified_as"]
            and action["clip_start_ms"] <= pred["time_ms"] <= action["clip_end_ms"]
            for action in non_resource_actions
        )
        if known_non_resource or any(start <= pred["time_ms"] <= end for start, end in empty_windows):
            false_positives.append(pred)
        else:
            eligible.append(pred)
    assigned = match_events(eligible, references, tolerance)
    owner_assigned = match_events(eligible, references, tolerance, require_same_owner=True)
    matches = []
    unverified = [row for row in resources if prediction_consumption(row) == MedicineConsumption.UNKNOWN]
    for index, pred in enumerate(eligible):
        timestamp = pred["time_ms"]
        if index in assigned:
            ref = references[assigned[index]]
            start, end = ref["mapped_counter_interval_ms"]
            ready = pred.get("decision_ready_ms")
            ambiguous = ref.get("timing_group")
            matches.append(dict(prediction=pred, reference=ref,
                owner_correct=pred["owner_id"] == ref["owner_id"] if not ambiguous else None,
                actor_correct=pred.get("actor_id") == ref["actor_id"] if ref["actor_id"] and not ambiguous else None,
                target_correct=pred.get("target_id") == ref["target_id"] if ref["target_id"] and not ambiguous else None,
                offset_bounds_ms=[timestamp - end, timestamp - start],
                decision_offset_bounds_ms=[ready - end, ready - start] if ready is not None else None))
        elif complete_window is not None:
            false_positives.append(pred)
        else:
            unverified.append(pred)

    missed = [ref for index, ref in enumerate(references) if index not in assigned.values()]
    by_event_type = {}
    for kind in sorted(RESOURCE_TYPES):
        support = sum(row["event_type"] == kind for row in references)
        hits = sum(match["reference"]["event_type"] == kind for match in matches)
        owner_hits = sum(references[index]["event_type"] == kind for index in owner_assigned.values())
        verified_wrong = sum(row["event_type"] == kind for row in false_positives)
        by_event_type[kind] = dict(references=support, hits=hits, misses=support - hits,
                                   owner_and_time_hits=owner_hits,
                                   owner_and_time_recall=owner_hits / support if support else None,
                                   verified_false_positives=verified_wrong,
                                   unverified_predictions=sum(row["event_type"] == kind for row in unverified),
                                   reviewed_recall=hits / support if support else None,
                                   scoped_precision=(hits / (hits + verified_wrong)
                                                     if hits + verified_wrong else None))
    classes = {}
    for kind in EVENT_CLASSES:
        support = sum(event_class(ref) == kind for ref in references)
        correct = sum(event_class(m["reference"]) == event_class(m["prediction"]) == kind
                      for m in matches)
        verified_wrong = sum(event_class(pred) == kind for pred in false_positives)
        verified_wrong += sum(event_class(m["prediction"]) == kind
                              and event_class(m["reference"]) not in {None, kind} for m in matches)
        classes[kind] = dict(references=support, hits=correct, misses=support - correct,
                             verified_false_positives=verified_wrong,
                             reviewed_recall=correct / support if support else None,
                             scoped_precision=(correct / (correct + verified_wrong)
                                               if correct + verified_wrong else None))
    attribution = {}
    for key in ("owner_correct", "actor_correct", "target_correct"):
        values = [m[key] for m in matches if m[key] is not None]
        attribution[key] = dict(correct=sum(values), compared=len(values),
                                accuracy=sum(values) / len(values) if values else None)
    timings = [match["decision_offset_bounds_ms"] for match in matches
               if match["decision_offset_bounds_ms"] is not None and not match["reference"].get("timing_group")]
    return dict(classes=classes, by_event_type=by_event_type, attribution=attribution,
                reviewed_reference_count=len(references),
                verified_empty_windows_ms=empty_windows,
                precision_scope=("complete reviewed combat window" if complete_window is not None
                                 else "reviewed matches and verified negative evidence only"),
                timing_scope="prediction offset from reviewed event interval, not processing latency",
                recorded_decision_offsets=dict(compared=len(timings),
                    median_midpoint_ms=median(sum(bounds) / 2 for bounds in timings) if timings else None,
                    max_upper_ms=max((bounds[1] for bounds in timings), default=None),
                    scope="recorded video time, excluding live capture and processing"),
                prediction_types=dict(Counter(row["event_type"] for row in scoped)),
                unknown_fraction=(sum(row["event_type"] == "UNKNOWN" for row in scoped)
                                  / len(scoped) if scoped else None),
                matches=matches, misses=missed, verified_false_positives=false_positives,
                unverified_predictions=unverified,
                out_of_overlap_prediction_count=len(predictions) - len(scoped))


def event_metrics_by_side(predictions, references, empty_windows, overlap, tolerance, *,
                          complete_window, non_resource_actions=()):
    """Measure hits against the same side so wrong-side events cannot count as hits."""
    by_side = {}
    for side in ("L", "R"):
        stats = event_metrics(
            [row for row in predictions if (row.get("owner_id") or "").startswith(side)],
            [row for row in references if row["owner_id"].startswith(side)],
            empty_windows, overlap, tolerance, complete_window=complete_window,
            non_resource_actions=non_resource_actions,
        )
        by_side[side] = dict(
            by_event_type=stats["by_event_type"], classes=stats["classes"],
            precision_scope=stats["precision_scope"],
            attribution=stats["attribution"], recorded_decision_offsets=stats["recorded_decision_offsets"],
            matches=[dict(sample_id=row["reference"]["sample_id"],
                          prediction_id=row["prediction"]["prediction_id"], owner_correct=row["owner_correct"])
                     for row in stats["matches"]],
            missed_sample_ids=[row["sample_id"] for row in stats["misses"]],
            verified_false_positive_ids=[row["prediction_id"] for row in stats["verified_false_positives"]],
        )
    return by_side


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def engineering_targets(events, counts, coverage):
    complete_events = bool(coverage and coverage["complete_event_reference"])
    complete_rounds = bool(coverage and coverage["complete_round_end_checkpoints"])

    def comparison(observed, target, verified):
        return dict(observed=observed, target=target,
                    met=observed >= target if verified and observed is not None else None)

    result = {}
    for event_type, threshold in (("USE_MEDICINE", .98), ("SUMMON_PET", .99)):
        row = events["by_event_type"][event_type]
        result[event_type] = dict(precision=comparison(row["scoped_precision"], threshold, complete_events),
                                  recall=comparison(row["reviewed_recall"], threshold, complete_events))
    result["owner_accuracy"] = comparison(events["attribution"]["owner_correct"]["accuracy"], .99, complete_events)
    result["round_end_resource_accuracy"] = comparison(counts.get("round_end_only", {}).get("exact_rate"),
                                                       .95, complete_rounds)
    return result


def export_failure_frames(source, events, directory):
    cases = []
    for row in events["misses"]:
        start, end = row["mapped_counter_interval_ms"]
        cases.append(dict(reason="miss", sample_id=row["sample_id"], requested_ms=(start + end) // 2))
    for row in events["verified_false_positives"]:
        cases.append(dict(reason="false_positive", prediction_id=row["prediction_id"],
                          requested_ms=row["time_ms"]))
    for match in events["matches"]:
        if any(match[field] is False for field in ("actor_correct", "owner_correct", "target_correct")):
            row = match["prediction"]
            cases.append(dict(reason="attribution_error", prediction_id=row["prediction_id"],
                              sample_id=match["reference"]["sample_id"], requested_ms=row["time_ms"]))
    if not cases:
        return cases
    directory.mkdir(parents=True, exist_ok=True)
    times = {case["requested_ms"] for case in cases}
    observed = {}
    for actual, requested, frame in frames(source, {"width": 1120, "height": 512}, requested=times):
        destination = directory / f"{actual:06d}.jpg"
        Image.fromarray(frame).save(destination, quality=90)
        observed.update({time: (actual, destination.as_posix()) for time in requested})
    for case in cases:
        case["frame_ms"], case["frame_path"] = observed[case["requested_ms"]]
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", type=Path,
                        default=ROOT / "data/annotations/bird-mirror-alignment.json")
    parser.add_argument("--counter-layout", type=Path,
                        default=ROOT / "configs/counters/bird-mirror.json")
    parser.add_argument("--counter-reference", type=Path,
                        default=ROOT / "data/annotations/bird-mirror-counters.json")
    parser.add_argument("--review", type=Path,
                        default=ROOT / "data/annotations/bird-mirror-reviewed-events.json")
    parser.add_argument("--predictions", type=Path,
                        default=OUTPUT / "bird-baseline/predictions.jsonl")
    parser.add_argument("--output", type=Path, default=OUTPUT / "bird-evaluation.json")
    parser.add_argument("--export-evidence", action="store_true")
    args = parser.parse_args()
    alignment, aligned, assets, detailed, clean = prepare(args.alignment)
    check_grouping(assets, [])
    review = read_json(args.review)
    if (clean["dataset_split"] != "test" or detailed["dataset_split"] != "test"
            or review["battle_id"] != clean["battle_id"]
            or review["clean_asset_id"] != clean["asset_id"]
            or review["detailed_asset_id"] != detailed["asset_id"]
            or review["alignment_version"] != aligned["version"]):
        raise ValueError("review must identify the registered test replay pair")
    layout = read_json(args.counter_layout)
    if layout["source_asset"] != detailed["asset_id"]:
        raise ValueError("counter layout belongs to another replay")
    predictions = read_jsonl(args.predictions)
    visual_path = args.predictions.with_name("visual_summary.json")
    visual = read_json(visual_path)
    expected = (clean["battle_id"], clean["asset_id"], clean["sha256"])
    if any((r["battle_id"], r["source_asset"], r["source_sha256"]) != expected
           for r in predictions):
        raise ValueError("prediction belongs to another replay")
    if ((visual["source_asset"], visual["source_sha256"]) != expected[1:]
            or dict(Counter(r["event_type"] for r in predictions)) != visual["event_types"]):
        raise ValueError("prediction summary does not match the evaluated replay")
    if any(r.get("input_view", "replay") != visual.get("input_view", "replay") for r in predictions):
        raise ValueError("prediction view does not match the replay summary")
    if len({r["prediction_id"] for r in predictions}) != len(predictions):
        raise ValueError("duplicate prediction IDs")
    if any(r.get("decision_ready_ms") is not None and
           not r["time_ms"] <= r["decision_ready_ms"] <= visual["duration_ms"] for r in predictions):
        raise ValueError("decision time is outside observed frames")
    overlap = [alignment.anchors[0].clean_ms, alignment.anchors[-1].clean_ms]
    checkpoint_reference = read_json(args.counter_reference)
    owners = checkpoint_reference["owner_order"]
    if (checkpoint_reference["source_asset"] != detailed["asset_id"]
            or checkpoint_reference["source_sha256"] != detailed["sha256"]
            or len(owners) != len(layout["players"])
            or set(owners) != set(layout["players"])):
        raise ValueError("manual counter reference has a different source or owner layout")
    targets = {*owners, *(f"{owner}_PET" for owner in owners), None}
    changes = review["changes"]
    source_rows = [*changes, *review.get("counter_only_changes", [])]
    if len({row["sample_id"] for row in source_rows}) != len(source_rows):
        raise ValueError("duplicate event or counter-change ID")
    aligned_rows = []
    for row in source_rows:
        start, end = row["detailed_interval_ms"]
        mapped = [alignment.map(t) for t in (start, end)]
        owner = row["owner_id"]
        if (None in mapped or mapped != row["mapped_counter_interval_ms"] or start >= end
                or owner not in layout["players"] or row["event_type"] not in RESOURCE_TYPES
                or counter_used(row["after"]) != counter_used(row["before"]) + 1
                or row["actor_id"] not in {owner, f"{owner}_PET", None}
                or row["target_id"] not in targets
                or (row["event_type"] == "SUMMON_PET" and
                    (row["actor_id"] != owner or row["target_id"] != f"{owner}_PET"))):
            raise ValueError("invalid reviewed resource event for the aligned replay")
        aligned_rows.append(dict(row, aligned_official_counter_interval_ms=mapped))
    references = aligned_rows[:len(changes)]
    counter_only_changes = aligned_rows[len(changes):]
    hard_negatives = review.get("hard_negatives", [])
    non_resource_actions = [row for row in hard_negatives if "misclassified_as" in row]
    if any(misclassified_actor(row) not in targets - {None} for row in non_resource_actions):
        raise ValueError("reviewed non-resource action must identify the predicted battle unit")
    empty_windows = review.get("verified_empty_windows_ms", [])
    verified = set()
    for evidence in review.get("verified_empty_window_evidence", []):
        first, last = [alignment.map(t) for t in evidence["detailed_endpoints_ms"]]
        start, end = evidence["clean_window_ms"]
        margin = aligned["tolerance_ms"]
        if (None in {first, last} or [first, last] != evidence["clean_aligned_endpoints_ms"]
                or not first + margin <= start < end <= last - margin):
            raise ValueError("empty window falls outside the verified counter endpoints")
        verified.add((start, end))
    if any(tuple(window) not in verified for window in empty_windows):
        raise ValueError("missing manually reviewed evidence for an empty window")
    counts = checkpoint_metrics(predictions, checkpoint_reference["reference"],
                                owners, alignment, aligned["tolerance_ms"],
                                checkpoint_reference.get("round_end_reference_seconds", []))
    completion = review.get("completion")
    if completion is not None:
        boundary = read_json(args.alignment).get("replay_coverage", {})
        if (not completion["evidence"]
                or not all((ROOT / evidence).is_file() for evidence in completion["evidence"])
                or completion["final_checkpoint_ms"] != boundary.get("last_combat_detailed_ms")
                or alignment.map(completion["final_checkpoint_ms"]) != boundary.get("last_combat_clean_ms")
                or counts["checkpoints"][-1]["round"] != boundary.get("terminal_round")):
            raise ValueError("complete replay requires local start and settlement evidence matching the pair")
    tolerance = 500
    coverage = completion_metrics(completion, counts, references, tolerance,
                                  counter_only_changes=counter_only_changes)
    complete_window = (coverage["reviewed_combat_window_ms"]
                       if coverage and coverage["complete_event_reference"] else None)
    events = event_metrics(predictions, references, empty_windows, overlap, tolerance,
                           complete_window=complete_window, non_resource_actions=non_resource_actions)
    events_by_side = event_metrics_by_side(predictions, references, empty_windows, overlap, tolerance,
                                           complete_window=complete_window,
                                           non_resource_actions=non_resource_actions)
    final_verified = bool(coverage and coverage["terminal_counts_complete"])
    if final_verified:
        counts.update(status="complete_replay", final_battle_counts=counts["last_checkpoint"])
    if not events["reviewed_reference_count"] or not counts.get("all_checkpoints", {}).get("compared"):
        raise ValueError("the overlap requires both reviewed events and readable counter checkpoints")
    sources = (args.alignment, args.counter_layout, args.counter_reference, args.review,
               args.predictions, visual_path)
    unverified = ["live_capture_latency", "other_formations_and_viewpoints"]
    scope = "complete replay" if final_verified else "aligned partial replay"
    if visual.get("input_view") == "participant_simulation":
        scope += " with simulated participant visibility"
        unverified.append("actual_participant_view")
    if not complete_window:
        unverified.extend(["whole_battle_reference", "whole_battle_precision"])
    if not final_verified:
        unverified.append("final_resource_consistency")
    if any(event_class(row) is None for row in references):
        unverified.append("full_medicine_actor_type_metrics")
    result = dict(scope=scope,
                  battle_id=clean["battle_id"], source_files=aligned["sources"],
                  clean_overlap_ms=overlap, tolerance_ms=tolerance, final_battle_verified=final_verified,
                  completion_coverage=coverage, unverified_dimensions=unverified,
                  input_sha256={str(path): digest(path) for path in sources}, events=events,
                  events_by_side=events_by_side, counts=counts,
                  engineering_targets=engineering_targets(events, counts, coverage),
                  counter_only_changes=counter_only_changes,
                  hard_negative_examples=hard_negatives, visual_run=visual)
    if args.export_evidence:
        result["failure_frames"] = export_failure_frames(ROOT / clean["path"], events,
            args.output.parent / ("bird-failure-frames" if args.output.name == "bird-evaluation.json"
                                  else f"{args.output.stem}-failure-frames"))
    write_json(args.output, result)
    print(json.dumps(dict(references=events["reviewed_reference_count"],
                          verified_false_positives=len(events["verified_false_positives"]),
                          unverified_predictions=len(events["unverified_predictions"]),
                          counts=counts.get("all_checkpoints"))))


if __name__ == "__main__":
    main()
