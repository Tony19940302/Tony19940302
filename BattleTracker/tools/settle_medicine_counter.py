"""Compare raw detailed-replay readings for a reviewed medicine waiver."""

import argparse
import hashlib
import json
from pathlib import Path

from battletracker.counters import MedicineReading, settle_medicine_waiver


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def settle(raw_path, review_path, case_id, max_gap_ms):
    report = read_json(raw_path.with_name("counter_extraction_report.json"))
    with raw_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != report.get("raw_reads_sha256"):
        raise ValueError("raw counter readings do not match their extraction report")
    cases = [case for case in read_json(review_path)["cases"] if case["case_id"] == case_id]
    if len(cases) != 1 or cases[0].get("status") != "reviewed":
        raise ValueError("settlement requires a single reviewed case")
    case, = cases
    display = case["detailed_counter_display"]
    expected = (display["source_asset"], display["source_sha256"], case["battle_id"])
    if expected != (report["source_asset"], report["source_sha256"], report["battle_id"]):
        raise ValueError("review and counter readings belong to different sources")
    actions = [action for action in case["actions"]
               if action["event_type"] == "USE_MEDICINE"
               and action.get("payload", {}).get("medicine_consumption") == "waived"]
    if len(actions) != 1 or actions[0]["geometry_owner_id"] != display["geometry_owner_id"]:
        raise ValueError("review must identify one waived action and its counter owner")
    action, = actions
    owner = action["geometry_owner_id"]
    field = f"{owner}.medicine"
    action_ms = round(display["first_green_at_ms"])
    with raw_path.open(encoding="utf-8") as stream:
        rows = (json.loads(line) for line in stream if line.strip())
        readings = (MedicineReading(row["timestamp_ms"], row["reads"]["round"]["value"],
                                    row["reads"][field]["value"]) for row in rows)
        result = settle_medicine_waiver(readings, action_ms=action_ms,
                                        round_number=case["round"], max_gap_ms=max_gap_ms)
    with review_path.open("rb") as stream:
        review_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(source_asset=report["source_asset"], source_sha256=report["source_sha256"],
                raw_reads_sha256=digest, review_sha256=review_digest, case_id=case_id,
                action_id=action["action_id"], action_ms=action_ms, round=case["round"],
                owner_id=owner, max_gap_ms=max_gap_ms, **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-reads", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--max-gap-ms", type=int, default=525)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    inputs = (args.raw_reads, args.review, *(args.raw_reads.with_name(name) for name in (
        "counter_extraction_report.json", "official_events.jsonl", "official_snapshots.jsonl",
    )))
    if any(output == source.resolve() or (output.is_file() and source.is_file() and output.samefile(source))
           for source in inputs):
        parser.error("output must not replace one of the source evidence files")
    report = settle(args.raw_reads, args.review, args.case_id, args.max_gap_ms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
