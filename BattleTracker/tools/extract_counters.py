"""Read official resource counters from the annotated detailed replay."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from PIL import Image

from battletracker.counters import FIELDS, StableValue, TemplateReader, changes_for, sampled_regions

ROOT = Path(__file__).resolve().parents[1]


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_crop(output, timestamp, owner, field, image):
    name = f"evidence/{timestamp:06d}-{owner or 'match'}-{field}.png"
    path = output / name
    path.parent.mkdir(exist_ok=True)
    Image.fromarray(image).save(path)
    return name


def key_name(owner, field):
    return f"{owner}.{field}" if owner else field


def counter_values(value):
    if value is None:
        return None
    used, limit = map(int, value.split("/"))
    return dict(used=used, limit=limit)


def reference_labels(config, annotations):
    players = tuple(config["players"])
    keys = [(None, "round")] + [(p, field) for p in players for field in FIELDS]
    reference = {}
    for time, round_number, *values in annotations["reference"]:
        if len(values) != len(players):
            raise ValueError("reference does not cover all players")
        flattened = [value for pair in values for value in pair.split()]
        reference[time] = dict(zip(keys, (round_number, *flattened), strict=True))
    return keys, reference


def extract(source, config, annotations, output, step, battle_id):
    labels = annotations["templates"]
    players = tuple(config["players"])
    keys, reference = reference_labels(config, annotations)
    train = {row[0] for row in labels}
    if train & reference.keys():
        raise ValueError("a template frame also occurs in the reference")
    templates = {
        request: regions
        for frame, regions in sampled_regions(source, config, train)
        for request in frame["requests"]
    }
    if templates.keys() != train:
        raise ValueError("missing template times in source")
    reader = TemplateReader(labels, templates, config.get("template_bounds"))
    tracks = {key: StableValue() for key in keys}
    previous_round_at_value = {}
    rounds_by_time = {}
    coverage = Counter()
    observed_rounds = set()
    outcomes = {"round": Counter(), "medicine": Counter(), "pet": Counter()}
    formation_evidence = {}
    errors = []
    anomalies = []
    seen_anomalies = set()
    seen_ref = set()
    raw_unknowns = Counter()
    stable_unknowns = Counter()
    event_types = Counter()
    samples = 0
    output.mkdir(parents=True, exist_ok=True)
    with ((output / "raw_reads.jsonl").open("w", encoding="utf-8") as raw,
          (output / "official_snapshots.jsonl").open("w", encoding="utf-8") as snapshots,
          (output / "official_events.jsonl").open("w", encoding="utf-8") as events):
        for frame, regions in sampled_regions(source, config, reference, step):
            time = frame["timestamp_ms"]
            samples += 1
            if samples == 1:
                for side, item in config.get("formations", {}).items():
                    formation_evidence[side] = dict(
                        expected=item["expected"], timestamp_ms=time, pts=frame["pts"],
                        evidence=export_crop(output, time, side, "formation", regions[side, "formation"]),
                    )
            readings = {key: reader.read(regions[key], key[1]) for key in keys}
            jsonl(raw, dict(frame, reads={key_name(*key): reading for key, reading in readings.items()}))
            for request in frame["requests"]:
                seen_ref.add(request)
                for key, expected in reference[request].items():
                    owner, field = key
                    actual = readings[key]["value"]
                    if expected == "?":
                        outcome = "obscured" if actual is None else "read_on_obscured"
                    elif actual is None:
                        outcome = "unknown"
                    else:
                        outcome = "correct" if actual == expected else "incorrect"
                    outcomes[field][outcome] += 1
                    if outcome in {"unknown", "incorrect", "read_on_obscured"}:
                        errors.append(dict(request_seconds=request, timestamp_ms=time, owner_id=owner,
                                           field=field, expected=expected, actual=actual, outcome=outcome,
                                           evidence=export_crop(output, time, owner, field, regions[key])))

            stable = {}
            round_number = None
            for key in keys:
                owner, field = key
                value = readings[key]["value"]
                track = tracks[key]
                raw_unknowns[field] += int(value is None)
                if value is not None and track.value is not None:
                    used = value if field == "round" else int(value.split("/")[0])
                    previous = track.value if field == "round" else int(track.value.split("/")[0])
                    if used < previous:
                        signature = (key, track.value, value)
                        if signature not in seen_anomalies:
                            seen_anomalies.add(signature)
                            anomalies.append(dict(timestamp_ms=time, owner_id=owner, field=field,
                                                  prior=track.value, observed=value,
                                                  reason="counter_regression",
                                                  evidence=export_crop(output, time, owner, field, regions[key])))
                        value = None
                stable[key], change = track.observe(value, time)
                stable_unknowns[field] += int(stable[key] is None)
                if field == "round":
                    round_number = stable[key]
                    rounds_by_time[time] = round_number
                    if round_number is not None:
                        observed_rounds.add(round_number)
                elif change is not None:
                    for event in changes_for(owner, field, change, frame, rounds_by_time[change[3]],
                                             previous_round_at_value.get(key)):
                        if event["needs_review"]:
                            event["evidence"] = export_crop(output, time, owner, field, regions[key])
                        jsonl(events, dict(event, battle_id=battle_id))
                        event_types[event["event_type"]] += 1
                if stable[key] is not None:
                    previous_round_at_value[key] = round_number
                    coverage[key_name(*key)] += 1
            jsonl(snapshots, dict(frame, round=stable[None, "round"], battle_id=battle_id,
                                 players={p: {f: counter_values(stable[p, f]) for f in FIELDS}
                                          for p in players}))

    if seen_ref != reference.keys():
        raise ValueError("one or more reference frames are beyond the video")
    metrics = {}
    for field, counts in outcomes.items():
        readable = counts["correct"] + counts["incorrect"] + counts["unknown"]
        calls = counts["correct"] + counts["incorrect"]
        units = 1 if field == "round" else len(players)
        metrics[field] = dict(counts, readable_reference=readable,
                              precision_on_readable=counts["correct"] / calls if calls else None,
                              reference_unknown_rate=counts["unknown"] / readable if readable else None,
                              raw_unknown_rate=raw_unknowns[field] / (units * samples),
                              stable_unknown_rate=stable_unknowns[field] / (units * samples))
    return dict(calibration=config["version"], source_asset=config["source_asset"],
                samples=samples, step_seconds=float(step), reference_frames=len(seen_ref),
                metrics=metrics, reference_review=errors, anomalies=anomalies, event_types=event_types,
                formation_evidence=formation_evidence,
                battle_id=battle_id, observed_rounds=sorted(observed_rounds), stable_coverage={
                    key_name(*key): coverage[key_name(*key)] for key in keys
                })


def validate_baseline(report, baseline):
    if (report["source_asset"] != baseline["source_asset"]
            or report["reference_frames"] != baseline["reference_frames"]
            or report["step_seconds"] != baseline["step_seconds"]):
        raise ValueError("counter regression baseline applies to a specific source and sampling setup")
    if set(report["formation_evidence"]) != {"left", "right"}:
        raise ValueError("both formation icon crops are required")
    for field, expected in baseline["metrics"].items():
        actual = report["metrics"][field]
        if (actual["readable_reference"] != expected["readable_reference"]
                or actual.get("correct", 0) < expected["minimum_correct"]
                or actual.get("incorrect", 0) > expected["maximum_incorrect"]
                or actual.get("read_on_obscured", 0) > expected["maximum_read_on_obscured"]):
            raise ValueError(f"reference read regression: {field}")
    for name, minimum in baseline["minimum_stable_coverage"].items():
        if report["stable_coverage"].get(name, 0) < minimum:
            raise ValueError(f"stable read coverage regression: {name}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, default=ROOT / "configs/counters/thunder-tiger.json")
    parser.add_argument("--annotations", type=Path,
                        default=ROOT / "data/annotations/thunder-tiger-counters.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/phase-02")
    parser.add_argument("--step", type=Fraction, default=Fraction(1, 4))
    parser.add_argument("--verify-reference", action="store_true")
    parser.add_argument("--baseline", type=Path,
                        default=ROOT / "configs/counters/thunder-tiger-baseline.json")
    args = parser.parse_args()
    config = load_json(args.layout)
    manifest = load_json(ROOT / "data/manifests/assets.json")
    asset = next(row for row in manifest["assets"] if row["asset_id"] == config["source_asset"])
    if asset["role"] != "detailed_match_replay":
        raise ValueError("counter calibration requires a detailed replay")
    source = ROOT / asset["path"]
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != asset["sha256"]:
        raise ValueError("source hash does not match the calibrated replay")
    report = extract(source, config, load_json(args.annotations), args.output, args.step, asset["battle_id"])
    report["source_sha256"] = digest
    with (args.output / "official_events.jsonl").open("rb") as events:
        report["official_events_sha256"] = hashlib.file_digest(events, "sha256").hexdigest()
    (args.output / "counter_extraction_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in [
        "samples", "metrics", "event_types", "observed_rounds", "stable_coverage"
    ]}, indent=2))
    if args.verify_reference:
        validate_baseline(report, load_json(args.baseline))
        expected_rounds = {row[1] for row in load_json(args.annotations)["reference"]}
        if not expected_rounds <= set(report["observed_rounds"]):
            raise SystemExit("At least one manually labeled round is missing in stable reads")
        if not all(report["stable_coverage"].values()) or not all(
            report["metrics"][field].get("correct", 0) for field in (*FIELDS, "round")
        ):
            raise SystemExit("Stable coverage or reference comparison is empty for a calibrated field")
        if not report["event_types"]:
            raise SystemExit("The calibrated replay produced no resource changes")


if __name__ == "__main__":
    main()



