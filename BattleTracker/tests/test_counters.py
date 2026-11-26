import pytest

pytest.importorskip("cv2")

from battletracker.counters import (
    MedicineReading, StableValue, TemplateReader, changes_for, settle_medicine_waiver,
    settle_medicine_waiver_adaptively,
)


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


def settle_window(rows):
    return settle_medicine_waiver((MedicineReading(*row) for row in rows),
                                 action_ms=200, round_number=9, max_gap_ms=250)


def test_reviewed_waiver_requires_stable_rollback_in_next_round():
    rows = [(0, 9, "2/10"), (100, 9, "2/10"),
            (200, 9, "3/10"), (300, 9, "3/10"), (400, 9, "3/10")]
    result = settle_window(rows + [(500, 10, "2/10"), (600, 10, "2/10")])
    assert (result["status"], result["baseline_used"], result["settled_used"]) == ("settled", 2, 2)
    assert result["net_waiver_delta"] == 0
    assert settle_window(rows + [(500, 10, "3/10"), (600, 10, "3/10")])["status"] == "unresolved"
    assert settle_window(rows + [(500, 10, "2/10")])["status"] == "unresolved"


def test_waiver_keeps_a_later_ordinary_increase_and_ignores_single_frame_noise():
    rows = [(0, 9, "2/10"), (100, 9, "2/10"), (200, 9, "3/10"),
            (250, 9, "3/10"), (300, 9, "2/10"), (350, 9, "3/10"), (400, 9, "3/10"),
            (500, 9, "4/10"), (600, 9, "4/10"), (700, 10, "3/10"), (800, 10, "3/10")]
    result = settle_window(rows)
    assert result["settled_used"] == 3 and result["net_waiver_delta"] == 0
    assert result["other_charges"] == [{"before_ms": 400, "after_ms": 500, "used_before": 2, "used_after": 3}]


@pytest.mark.parametrize("tail", [
    [(400, 9, "2/10"), (500, 9, "2/10"), (600, 10, "2/10"), (700, 10, "2/10")],
    [(400, 9, "3/11"), (500, 9, "3/11"), (600, 10, "2/11"), (700, 10, "2/11")],
    [(400, 9, "5/10"), (500, 9, "5/10"), (600, 10, "4/10"), (700, 10, "4/10")],
    [(450, 9, None), (600, 10, "2/10"), (700, 10, "2/10")],
    [(400, 11, "2/10"), (500, 11, "2/10")],
])
def test_ambiguous_display_never_sets_a_waiver(tail):
    rows = [(0, 9, "2/10"), (100, 9, "2/10"), (200, 9, "3/10"), (300, 9, "3/10")]
    result = settle_window(rows + tail)
    assert result["status"] == "unresolved"
    assert "net_waiver_delta" not in result


def test_an_already_raised_value_cannot_be_claimed_by_a_later_waiver():
    rows = [(0, 9, "2/10"), (50, 9, "2/10"), (100, 9, "3/10"), (150, 9, "3/10"),
            (200, 9, "3/10"), (250, 9, "3/10"), (300, 10, "2/10"), (350, 10, "2/10")]
    assert settle_window(rows)["status"] == "unresolved"


def test_continuous_known_round_can_skip_unread_plateau_but_not_changed_values():
    prefix = [(0, 9, "2/10"), (100, 9, "2/10"), (200, 9, "3/10"), (300, 9, "3/10")]
    hidden = [(400, 9, None), (500, 9, None)]
    suffix = [(600, 9, "3/10"), (700, 9, "3/10"), (800, 10, "2/10"), (900, 10, "2/10")]
    result = settle_window(prefix + hidden + suffix)
    assert result["status"] == "settled"
    assert result["unread_same_value_intervals_ms"] == [[300, 600]]
    assert settle_window(prefix + suffix)["status"] == "unresolved"
    assert settle_window(prefix + [(400, None, None), hidden[1]] + suffix)["status"] == "settled"
    assert settle_window(prefix + [(400, None, None), (500, None, None)] + suffix)["status"] == "unresolved"
    assert settle_window(prefix + [(400, 10, None), hidden[1]] + suffix)["status"] == "unresolved"
    assert settle_window(prefix + hidden + [(600, 10, "2/10"), (700, 10, "2/10")])["status"] == "unresolved"


def test_adaptive_waiver_uses_lower_bottle_when_coarse_round_is_unknown():
    coarse = [MedicineReading(*row) for row in [
        (100, 9, "2/10"), (600, 9, "3/10"), (800, 9, "3/10"), (1000, 9, "3/10"),
        (1200, None, "2/10"), (1400, None, "2/10"),
    ]]
    native = [MedicineReading(*row) for row in [
        (90, 9, "2/10"), (100, 9, "2/10"), (210, 9, "3/10"), (220, 9, "3/10"),
        (540, 9, "3/10"), (1140, 10, "2/10"), (1155, 10, "2/10"),
    ]]
    report = settle_medicine_waiver_adaptively(coarse,
        lambda start, end: [row for row in native if start <= row.timestamp_ms <= end],
        action_ms=200, round_number=9, max_gap_ms=525)
    assert report["coarse"]["status"] == "unresolved"
    assert report["native_windows_ms"] == [(0, 550), (800, 1450)]
    assert report["samples"] == 12
    assert report["result"]["rollback_first_ms"] == 1140
    assert report["result"]["net_waiver_delta"] == 0


def test_adaptive_waiver_can_check_round_change_without_assuming_a_rollback():
    coarse = [MedicineReading(*row) for row in [
        (100, 9, "2/10"), (400, 9, "3/10"), (600, 9, "3/10"), (800, 9, "3/10"),
        (1000, 10, "3/10"), (1200, 10, "3/10"),
        (1400, 11, "3/10"), (1600, 11, "3/10"), (1800, None, "3/10"), (2000, None, "2/10"),
    ]]
    report = settle_medicine_waiver_adaptively(coarse, lambda start, end: [],
        action_ms=200, round_number=9, max_gap_ms=525)
    assert report["native_windows_ms"] == [(0, 550), (600, 1250)]
    assert report["result"]["status"] == "unresolved"
    assert "net_waiver_delta" not in report["result"]


def test_adaptive_waiver_rechecks_even_when_the_coarse_result_looks_settled():
    coarse = [MedicineReading(*row) for row in [
        (0, 9, "2/10"), (100, 9, "2/10"), (200, 9, "3/10"), (300, 9, "3/10"),
        (500, 10, "2/10"), (600, 10, "2/10"),
    ]]
    arguments = dict(action_ms=200, round_number=9, max_gap_ms=525)
    absent = settle_medicine_waiver_adaptively(coarse, lambda *_: [], **arguments)
    assert absent["coarse"]["status"] == "settled"
    assert absent["result"] == {"status": "unresolved", "reason": "native_edge_not_confirmed"}

    rereads = [*coarse, MedicineReading(460, 9, "2/10"), MedicineReading(475, 9, "2/10")]
    contradicted = settle_medicine_waiver_adaptively(coarse,
        lambda start, end: [row for row in rereads if start <= row.timestamp_ms <= end], **arguments)
    assert contradicted["result"] == {"status": "unresolved", "reason": "ambiguous_intervening_change"}


@pytest.mark.parametrize("replacement", [MedicineReading(100, 9, "3/10"), MedicineReading(900, 9, "2/10")])
def test_adaptive_waiver_rejects_conflicting_or_out_of_window_native_read(replacement):
    with pytest.raises(ValueError, match="native reading"):
        settle_medicine_waiver_adaptively([MedicineReading(100, 9, "2/10")], lambda start, end: [replacement],
            action_ms=200, round_number=9, max_gap_ms=525)


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
