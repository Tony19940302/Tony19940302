import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]


def command_check(name, argv, artifacts=()):
    started = perf_counter()
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, encoding="utf-8",
                            errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    print(output)
    return {
        "check_name": name, "command": argv, "status": "PASS" if result.returncode == 0 else "FAIL",
        "exit_code": result.returncode, "duration": round(perf_counter() - started, 3),
        "result_summary": output, "artifact_paths": list(artifacts),
        "failure_reason": output if result.returncode else None,
    }


def asset_checks():
    manifest = ROOT / "data/manifests/assets.json"
    rows = json.loads(manifest.read_text(encoding="utf-8"))["assets"]
    if not rows:
        raise ValueError("empty asset manifest")
    for asset in rows:
        started = perf_counter()
        relative = Path(asset["path"])
        bases = {"workspace": ROOT, "user_profile": Path.home()}
        path = bases[asset["path_base"]] / relative
        status, reason, summary = "SKIPPED", None, "Source file unavailable on this host"
        if path.is_dir():
            summary = "Directory exists; individual assets require enumeration and hashes"
        if path.is_file():
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            expected = asset.get("sha256")
            if expected is None:
                summary = "Source now exists and needs inventory metadata and a recorded hash"
            elif digest != expected:
                status, reason, summary = "FAIL", "SHA-256 mismatch", digest
            else:
                status, summary = "PASS", f"SHA-256 {digest}"
        yield {
            "check_name": f"asset:{asset['asset_id']}", "command": ["sha256", str(path)],
            "status": status, "exit_code": {"SKIPPED": None, "PASS": 0, "FAIL": 1}[status],
            "duration": round(perf_counter() - started, 3),
            "result_summary": summary, "artifact_paths": [str(manifest)], "failure_reason": reason,
        }


def main():
    parser = argparse.ArgumentParser(description="Run BattleTracker acceptance checks.")
    parser.add_argument("--profile", required=True,
                        choices=["phase-01", "phase-02", "phase-03", "phase-04", "phase-05",
                                 "phase-06-overlap", "phase-06-final", "phase-06-participant"])
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--output", type=Path, help="override the participant profile's replay output directory")
    args = parser.parse_args()
    if args.output and args.profile != "phase-06-participant":
        parser.error("--output requires --profile phase-06-participant")
    checks = [
        command_check("unit_and_integration", [sys.executable, "-m", "pytest", "-q"]),
        command_check("environment_dependencies", [sys.executable, "-m", "pip", "check"]),
    ]
    try:
        checks.extend(asset_checks())
    except (OSError, ValueError, KeyError, TypeError) as error:
        checks.append({
            "check_name": "asset_manifest", "command": ["read", "data/manifests/assets.json"],
            "status": "FAIL", "exit_code": 1, "duration": 0,
            "result_summary": "Asset manifest could not be validated",
            "artifact_paths": [], "failure_reason": str(error),
        })
    if args.profile in {"phase-02", "phase-03"}:
        destination = ROOT / "outputs/phase-02"
        outputs = [
            destination / filename for filename in (
                "official_snapshots.jsonl", "official_events.jsonl", "raw_reads.jsonl",
                "counter_extraction_report.json",
            )
        ]
        checks.append(command_check("detailed_replay_counter_extraction", [
            sys.executable, "tools/extract_counters.py", "--output", str(destination),
            "--verify-reference",
        ], map(str, outputs)))
    if args.profile == "phase-03":
        phase_dir = ROOT / "outputs/phase-03"
        checks.append(command_check("paired_replay_alignment", [
            sys.executable, "tools/align_videos.py",
        ], [str(phase_dir / "alignment_report.json")]))
        checks.append(command_check("clean_replay_dataset", [
            sys.executable, "tools/build_dataset.py", "--export-clips",
        ], [str(phase_dir / name) for name in (
            "clip_index.jsonl", "unmatched_changes.jsonl", "dataset_report.json", "clips",
        )]))
    if args.profile == "phase-04":
        phase_dir = ROOT / "outputs/phase-04"
        prediction = command_check("clean_visual_baseline", [
            sys.executable, "tools/predict_clean.py",
        ], [str(phase_dir / "predictions.jsonl"), str(phase_dir / "visual_summary.json")])
        checks.append(prediction)
        if prediction["status"] == "PASS":
            checks.append(command_check("reviewed_event_evaluation", [
                sys.executable, "tools/evaluate.py",
            ], [str(phase_dir / "evaluation.json")]))
    if args.profile == "phase-05":
        phase_dir = ROOT / "outputs/phase-05"
        grouped_index = str(phase_dir / "grouped_index.jsonl")
        tasks = [
            ("held_out_alignment", ["tools/align_videos.py", "--annotations",
                "data/annotations/dragon-tiger-alignment.json", "--output",
                str(phase_dir / "alignment.json")], ("alignment.json",)),
            ("battle_disjoint_review_index", ["tools/prepare_phase5.py"],
                ("grouped_index.jsonl", "validation_index.jsonl")),
            ("linear_training_and_runtime", ["tools/train_model.py", "--index", grouped_index,
                "--output", str(phase_dir / "linear"), "--require-runtime"], ("linear/experiment.json",)),
            ("cnn_centroid_training_and_runtime", ["tools/train_temporal.py", "--index", grouped_index,
                "--output", str(phase_dir / "temporal")], ("temporal/experiment-temporal.json",)),
            ("held_out_visual_baseline", ["tools/predict_clean.py", "--layout",
                "configs/vision/dragon-tiger.json", "--output", str(phase_dir / "baseline")],
                ("baseline/visual_summary.json", "baseline/predictions.jsonl")),
        ]
        for model in ("linear", "cnn", "centroid"):
            tasks.append((f"held_out_replay_comparison_{model}",
                          ["tools/compare_phase5.py", "--model", model],
                          (f"comparison-{model}.json", f"predictions-{model}.jsonl")))
        for name, command, artifacts in tasks:
            result = command_check(name, [sys.executable, *command],
                                   [str(phase_dir / path) for path in artifacts])
            checks.append(result)
            if result["status"] == "FAIL":
                break
    if args.profile == "phase-06-overlap":
        output = ROOT / "outputs/phase-06"
        tasks = [
            ("paired_replay_alignment", ["tools/align_videos.py", "--annotations",
                "data/annotations/bird-mirror-alignment.json", "--output",
                str(output / "nov19/alignment.json")], "nov19/alignment.json"),
            ("detailed_counter_quality", ["tools/extract_counters.py", "--layout",
                "configs/counters/bird-mirror.json", "--annotations",
                "data/annotations/bird-mirror-counters.json", "--output",
                str(output / "bird-counters")], "bird-counters/counter_extraction_report.json"),
            ("clean_visual_baseline", ["tools/predict_clean.py", "--layout",
                "configs/vision/bird-mirror.json", "--output",
                str(output / "bird-baseline")], "bird-baseline/predictions.jsonl"),
            ("reviewed_overlap_metrics", ["tools/evaluate_phase6.py", "--export-evidence"],
                "bird-evaluation.json"),
        ]
        for name, command, artifact in tasks:
            result = command_check(name, [sys.executable, *command], [str(output / artifact)])
            checks.append(result)
            if result["status"] == "FAIL":
                break
    if args.profile in {"phase-06-final", "phase-06-participant"} and all(
            check["status"] == "PASS" for check in checks):
        output = ROOT / "outputs/phase-06/wind-cloud"
        alignment = "data/annotations/wind-cloud-alignment.json"
        evaluation = ["tools/evaluate_phase6.py", "--alignment", alignment, "--counter-layout",
                      "configs/counters/wind-cloud.json", "--counter-reference",
                      "data/annotations/wind-cloud-counters.json", "--review",
                      "data/annotations/wind-cloud-reviewed-events.json"]
        tasks = [
            ("final_pair_alignment", ["tools/align_videos.py", "--annotations", alignment,
                "--output", str(output / "alignment.json")], "alignment.json"),
            ("final_blind_prediction", ["tools/predict_clean.py", "--layout",
                "configs/vision/wind-cloud.json", "--detector", "temporal", "--output",
                str(output / "temporal"), "--baseline-output", str(output / "baseline")],
                "temporal/predictions.jsonl"),
            ("final_baseline_metrics", [*evaluation, "--predictions",
                str(output / "baseline/predictions.jsonl"), "--output", str(output / "baseline-metrics.json")],
                "baseline-metrics.json"),
            ("final_temporal_metrics", [*evaluation, "--predictions",
                str(output / "temporal/predictions.jsonl"), "--output", str(output / "temporal-metrics.json"),
                "--export-evidence"], "temporal-metrics.json"),
        ]
        if args.profile == "phase-06-participant":
            output = args.output.resolve() if args.output else output / "participant-simulation"
            tasks = [
                ("participant_visibility_prediction", ["tools/predict_clean.py", "--layout",
                    "configs/vision/wind-cloud.json", "--detector", "temporal", "--view",
                    "participant", "--output", str(output)], "predictions.jsonl"),
                ("participant_visibility_metrics", [*evaluation, "--predictions",
                    str(output / "predictions.jsonl"), "--output", str(output / "metrics.json")],
                    "metrics.json"),
            ]
        for name, command, artifact in tasks:
            result = command_check(name, [sys.executable, *command], [str(output / artifact)])
            checks.append(result)
            if result["status"] != "PASS":
                break
    for check in checks:
        check["profile"] = args.profile
    status = "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL"
    if args.report is None:
        args.report = ROOT / "harness/reports" / f"{args.profile}.json"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"profile": args.profile, "status": status, "checks": checks},
                                      indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{status}: {args.report}")
    return int(status == "FAIL")


if __name__ == "__main__":
    raise SystemExit(main())







