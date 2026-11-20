import pytest

pytest.importorskip("cv2")

from battletracker.counters import StableValue, TemplateReader, changes_for


def test_counter_needs_consecutive_observations_and_reports_interval():
    track = StableValue()
    assert track.observe("1/6", 0) == (None, None)
    assert track.observe("1/6", 250) == ("1/6", (None, "1/6", None, 0))
    assert track.observe("1/7", 500) == (None, None)
    assert track.observe(None, 750) == (None, None)
    assert track.observe("1/7", 1000) == (None, None)
    assert track.observe("1/7", 1250) == ("1/7", ("1/6", "1/7", 250, 1000))


def test_counter_jump_keeps_one_interval_and_unknown_actor():
    events = changes_for("L1", "pet", ("1/6", "3/7", 250, 1000),
                         {"timestamp_ms": 1250}, 3, 2)
    assert [(x["event_type"], x["delta"]) for x in events] == [
        ("SUMMON_PET", 2), ("PET_LIMIT_CHANGE", 1)]
    assert all(x["actor_id"] is None and x["needs_review"] for x in events)
    assert all(x["earliest_timestamp_ms"] == 250 and x["latest_timestamp_ms"] == 1000
               for x in events)


def test_baseline_does_not_create_action():
    assert changes_for("R2", "medicine", (None, "3/10", None, 0),
                       {"timestamp_ms": 250}, 2, None) == []


def test_template_bounds_preserve_second_round_digit():
    import numpy as np

    key = (None, "round")
    eleven = np.zeros((92, 110, 3), dtype=np.uint8)
    eleven[20:70, 12:23] = 255
    eleven[20:70, 82:94] = 255
    twelve = eleven.copy()
    twelve[45:55, 75:104] = 255
    twelve[55:63, 82:94] = 0
    labels = [(0, None, "round", 11), (1, None, "round", 12)]
    frames = {0: {key: eleven}, 1: {key: twelve}}

    assert TemplateReader(labels, frames).read(twelve, "round")["value"] is None
    reader = TemplateReader(labels, frames, {"round": [4, 6, 108, 87]})
    assert reader.read(eleven, "round")["value"] == 11
    assert reader.read(twelve, "round")["value"] == 12


def test_reference_baseline_rejects_lost_reads_and_false_positives(monkeypatch):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "tools"))
    from extract_counters import load_json, validate_baseline

    baseline = load_json(root / "configs/counters/thunder-tiger-baseline.json")
    report = dict(source_asset=baseline["source_asset"], reference_frames=baseline["reference_frames"],
                  step_seconds=baseline["step_seconds"], formation_evidence={"left": {}, "right": {}},
                  stable_coverage=dict(baseline["minimum_stable_coverage"]),
                  metrics={field: dict(readable_reference=bounds["readable_reference"],
                                       correct=bounds["minimum_correct"])
                           for field, bounds in baseline["metrics"].items()})
    validate_baseline(report, baseline)
    report["metrics"]["medicine"]["correct"] -= 1
    with pytest.raises(ValueError, match="medicine"):
        validate_baseline(report, baseline)
    report["metrics"]["medicine"]["correct"] += 1
    report["metrics"]["pet"]["read_on_obscured"] = 3
    with pytest.raises(ValueError, match="pet"):
        validate_baseline(report, baseline)
    report["metrics"]["pet"]["read_on_obscured"] = 0
    report["stable_coverage"]["L4.medicine"] -= 1
    with pytest.raises(ValueError, match="L4.medicine"):
        validate_baseline(report, baseline)
