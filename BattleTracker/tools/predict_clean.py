"""Generate offline events from the clean replay, without loading reference labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from time import perf_counter

from battletracker.vision import classify, observe

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vision/thunder-tiger.json"
OUTPUT = ROOT / "outputs/phase-04"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_predictions(output, events, config, asset, summary):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for i, row in enumerate(events, 1):
            record = dict(row, prediction_id=i, source_asset=asset["asset_id"],
                          source_sha256=asset["sha256"], battle_id=asset["battle_id"],
                          calibration=config["version"])
            if config["participant_view"]:
                record["input_view"] = summary["input_view"]
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = dict(summary, event_types=Counter(row["event_type"] for row in events))
    (output / "visual_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def check_pet_ownership(events, config):
    for row in events:
        owner = row.get("owner_id")
        if owner in config.get("unverified_pet_owners", []) and (
            row["event_type"] == "SUMMON_PET" or str(row.get("actor_id", "")).endswith("_PET")
        ):
            row.update(event_type="UNKNOWN", actor_id=None, owner_id=None)
            row["evidence"]["ownership_status"] = "unverified_pet_mapping"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--detector", choices=("baseline", "temporal"), default="baseline")
    parser.add_argument("--view", choices=("replay", "participant"), default="replay",
                        help="simulate the participant view by masking enemy gauges on the replay")
    parser.add_argument("--baseline-output", type=Path,
                        help="save baseline predictions from the same temporal observation run")
    args = parser.parse_args()
    if args.baseline_output and (args.detector != "temporal" or
                                 args.baseline_output.resolve() == args.output.resolve()):
        parser.error("--baseline-output requires the temporal detector and a distinct output directory")
    config = dict(load_json(args.layout), participant_view=args.view == "participant")
    assets = load_json(ROOT / "data/manifests/assets.json")["assets"]
    asset = next(row for row in assets if row["asset_id"] == config["source_asset"])
    if asset["role"] != "clean_match_replay" or asset["path_base"] != "workspace":
        raise ValueError("visual baseline requires an explicitly registered clean replay")
    path = ROOT / asset["path"]
    if sha256(path) != asset["sha256"]:
        raise ValueError("clean replay hash mismatch")
    template_path = None
    template_config = None
    template_source = None
    if config.get("template_layout"):
        template_config = load_json(ROOT / config["template_layout"])
        template_asset = next(row for row in assets
                              if row["asset_id"] == template_config["source_asset"])
        if (template_asset["role"] != "clean_match_replay" or
                template_asset["path_base"] != "workspace"):
            raise ValueError("templates require a clean replay")
        if asset["dataset_split"] == "test" and template_asset["dataset_split"] != "train":
            raise ValueError("held-out replay templates must come from training data")
        template_path = ROOT / template_asset["path"]
        if sha256(template_path) != template_asset["sha256"]:
            raise ValueError("template replay hash mismatch")
        template_source = {"asset_id": template_asset["asset_id"], "sha256": template_asset["sha256"],
                           "layout_sha256": sha256(ROOT / config["template_layout"])}
        # Template coordinates are tied to their training video. A new layout must
        # supply its own text search locations to enable text matching.
        if not config.get("skill_templates"):
            template_config = dict(template_config, skill_templates=[])
    elif asset["dataset_split"] == "test" and (config["portal_templates"] or config["skill_templates"]):
        raise ValueError("held-out replay cannot supply its own templates")
    temporal = args.detector == "temporal"
    observers = ()
    if temporal:
        from battletracker.medicine_vision import MedicineObserver
        from battletracker.summon_vision import SummonObserver, merge_summons

        summons, medicine = SummonObserver(config), MedicineObserver(config)
        observers = (summons, medicine)
    implementation = ["tools/predict_clean.py", "src/battletracker/vision.py"]
    if temporal:
        implementation.extend(["src/battletracker/summon_vision.py", "src/battletracker/medicine_vision.py"])
    provenance = {name: sha256(ROOT / name) for name in implementation}
    start = perf_counter()
    timeline, cues, healing = observe(path, config, template_path, template_config, observers=observers)
    observation_seconds = perf_counter() - start
    summary = dict(source_asset=asset["asset_id"], source_sha256=asset["sha256"],
                   duration_ms=timeline[-1][0], sampled_frames=len(timeline),
                   formations=config["formations"], player_slots=len(config["players"]),
                   pet_slots=len(config["pets"]), calibration=config["version"],
                   observation_seconds=observation_seconds, observer_mode=args.detector,
                   input_view="participant_simulation" if config["participant_view"] else "replay",
                   detector_file_sha256=provenance, layout_sha256=sha256(args.layout),
                   template_source=template_source)
    if not temporal or args.baseline_output:
        start = perf_counter()
        baseline = classify(timeline, cues, healing, config)
        baseline_summary = dict(summary, detector="baseline", classification_seconds=perf_counter() - start)
        check_pet_ownership(baseline, config)
        save_predictions(args.baseline_output or args.output, baseline, config, asset, baseline_summary)
    if temporal:
        start = perf_counter()
        events = classify(timeline, cues, healing, config, include_medicine=False)
        events = merge_summons(events, summons.events())
        skills = [(actor, name, cue) for (kind, actor, name), cue in cues if kind == "skill"]
        events.extend(medicine.classify(skills, summons=events))
        events.sort(key=lambda row: (row["time_ms"], row["event_type"]))
        temporal_summary = dict(summary, detector="temporal", classification_seconds=perf_counter() - start)
        check_pet_ownership(events, config)
        save_predictions(args.output, events, config, asset, temporal_summary)


if __name__ == "__main__":
    main()
