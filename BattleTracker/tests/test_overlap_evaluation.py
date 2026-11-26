"""Regression checks for the scope of partial replay metrics."""
from pathlib import Path

import pytest

from battletracker.alignment import Anchor, TimeAlignment


def prediction(number, time, owner="L1", actor="L1_PET", kind="USE_MEDICINE"):
    return dict(prediction_id=number, time_ms=time, owner_id=owner, actor_id=actor,
                target_id="L2", event_type=kind)


def test_only_reviewed_empty_windows_create_false_positive_counts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from evaluate_phase6 import event_metrics

    reference = dict(sample_id="observed", event_type="USE_MEDICINE",
                     owner_id="L1", actor_id="L1_PET", target_id="L2",
                     mapped_counter_interval_ms=[990, 1020])
    rows = [prediction(1, 1000, "L2", "L2_PET"), prediction(2, 2000), prediction(3, 3000)]
    report = event_metrics(rows, [reference], [[1800, 2100]], [0, 4000], 500)
    assert report["classes"]["PET_USE_MEDICINE"]["hits"] == 1
    assert report["classes"]["PET_USE_MEDICINE"]["scoped_precision"] == 0.5
    assert report["attribution"]["owner_correct"]["accuracy"] == 0
    assert report["verified_false_positives"] == [rows[1]]
    assert report["unverified_predictions"] == [rows[2]]
    boundary = event_metrics([prediction(4, 800)], [reference], [[500, 900]], [0, 4000], 500)
    assert len(boundary["verified_false_positives"]) == len(boundary["misses"]) == 1


def test_checkpoint_uses_known_baseline_and_preserves_uncertain_last_read(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from evaluate_phase6 import checkpoint_metrics

    alignment = TimeAlignment([Anchor("a", 1000, 1100, "a", 1),
                               Anchor("b", 8000, 8100, "b", 4)])
    rows = [[2, 1, "2/10 ?"], [4, 2, "3/10 1/6"], [6, 3, "4/10 1/6"]]
    candidates = [prediction(1, 1900), prediction(2, 4075)]
    report = checkpoint_metrics(candidates, rows, ["L1"], alignment, 100, {4})
    first, last = report["checkpoints"]
    assert first["fields"][0]["candidate_used_at_alignment_bounds"] == [2, 3]
    assert first["summary"]["timing_sensitive"] == 1
    assert last["summary"]["absolute_error_sum"] == 1
    assert report["round_end_only"]["compared"] == 1
    assert report["all_checkpoints"]["compared"] == 2
    assert report["final_battle_counts"] is None
    with pytest.raises(ValueError, match="regresses"):
        checkpoint_metrics(candidates, [[2, 1, "0/10 ?"], [4, 2, "3/10 ?"],
                                        [6, 3, "2/10 ?"]], ["L1"], alignment, 100, set())


def test_counter_metrics_keep_waivers_and_unresolved_cost_out_of_charged_events(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from evaluate_phase6 import checkpoint_metrics, event_metrics

    waived = dict(prediction(1, 1500), payload={"medicine_consumption": "waived"})
    unknown = dict(prediction(2, 2500), payload={"medicine_consumption": "unknown"})
    charged = prediction(3, 2700)
    reference = dict(sample_id="normal", event_type="USE_MEDICINE", owner_id="L1", actor_id="L1_PET",
                     target_id="L2", mapped_counter_interval_ms=[2600, 2750])
    report = event_metrics([waived, unknown, charged], [reference], [], (0, 4000), 150,
                           complete_window=(0, 4000))
    assert not report["verified_false_positives"]
    assert report["unverified_predictions"] == [unknown]
    assert report["matches"][0]["prediction"] == charged

    alignment = TimeAlignment([Anchor("a", 0, 0, "a", 1), Anchor("b", 4000, 4000, "b", 3)])
    counts = checkpoint_metrics([waived, unknown, charged], [[0, 1, "2/10 1/6"], [2, 1, "2/10 1/6"],
                                                           [4, 3, "3/10 1/6"]],
                                ["L1"], alignment, 0, {2, 4})
    first, last = counts["checkpoints"]
    assert first["fields"][0]["candidate_used"] == 2
    assert first["summary"]["matching"] == 2
    assert last["fields"][0]["unresolved_prediction_ids"] == [2]
    assert last["fields"][0]["error"] is None
    assert last["summary"]["compared"] == last["summary"]["unresolved_consumption"] == 1


def test_reviewed_attack_cannot_claim_nearby_medicine_reference(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from evaluate_phase6 import event_metrics, event_metrics_by_side

    reference = dict(sample_id="medicine", event_type="USE_MEDICINE", owner_id="L1",
                     actor_id="L1_PET", target_id="L2", mapped_counter_interval_ms=[995, 1010])
    attack = dict(actor_id="L2_PET", misclassified_as="USE_MEDICINE",
                  clip_start_ms=900, clip_end_ms=1100)
    rows = [prediction(1, 1000, "L2", "L2_PET"), prediction(2, 1050),
            prediction(3, 1700, "L2", "L2_PET")]

    report = event_metrics(rows, [reference], [], [0, 2500], 500, non_resource_actions=[attack])
    assert report["verified_false_positives"] == [rows[0]]
    assert report["matches"][0]["prediction"] == rows[1]
    assert report["unverified_predictions"] == [rows[2]]
    by_side = event_metrics_by_side(rows, [reference], [], [0, 2500], 500,
                                    complete_window=None, non_resource_actions=[attack])
    assert by_side["L"]["verified_false_positive_ids"] == [1]
    assert by_side["L"]["matches"][0]["prediction_id"] == 2


def test_end_of_round_heal_can_mark_false_use_without_a_known_caster(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from evaluate_phase6 import event_metrics

    periodic_heal = dict(actor_id=None, misclassified_actor_id="R2_PET", misclassified_as="USE_MEDICINE",
                         clip_start_ms=900, clip_end_ms=1100)
    rows = [prediction(1, 1000, "R2", "R2_PET"), prediction(2, 1000, "R1", "R1_PET"),
            prediction(3, 1200, "R2", "R2_PET"), prediction(4, 1050, "R2", "R2_PET", "SUMMON_PET")]

    report = event_metrics(rows, [], [], [0, 2500], 500, non_resource_actions=[periodic_heal])
    assert report["verified_false_positives"] == [rows[0]]
    assert report["unverified_predictions"] == rows[1:]


def test_complete_replay_requires_events_to_agree_at_every_round_end(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from evaluate_phase6 import checkpoint_metrics, completion_metrics

    alignment = TimeAlignment([Anchor("a", 0, 0, "a", 1), Anchor("b", 3000, 3000, "b", 2)])
    rows = [[0, 1, "0/10 1/6"], [1, 1, "1/10 1/6"], [3, 2, "1/10 1/6"]]
    counts = checkpoint_metrics([], rows, ["L1"], alignment, 0, {1, 3})
    end = dict(opening_checkpoint_ms=0, final_checkpoint_ms=3000, evidence=["settlement"])
    change = dict(owner_id="L1", event_type="USE_MEDICINE", after="1/10",
                  detailed_interval_ms=[850, 1000], mapped_counter_interval_ms=[850, 1000])
    assert completion_metrics(end, counts, [change], 500)["complete_event_reference"]
    counter_only = completion_metrics(end, counts, [], 500, counter_only_changes=[change])
    assert counter_only["terminal_counts_complete"] and counter_only["complete_round_end_checkpoints"]
    assert not counter_only["checkpoint_conflicts"]
    assert not counter_only["complete_event_reference"]
    misplaced = dict(change, detailed_interval_ms=[1400, 1500], mapped_counter_interval_ms=[1400, 1500])
    report = completion_metrics(end, counts, [misplaced], 500)
    assert report["terminal_counts_complete"] and not report["complete_event_reference"]
    assert report["checkpoint_conflicts"][0]["round"] == 1
    counts = checkpoint_metrics([], [rows[0], [1, 1, "1/10 ?"], rows[2]],
                                ["L1"], alignment, 0, {1, 3})
    assert not completion_metrics(end, counts, [change], 500)["complete_round_end_checkpoints"]


def test_simultaneous_count_changes_preserve_unresolved_attribution(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from evaluate_phase6 import event_metrics

    reference = dict(sample_id="s1", owner_id="L1", event_type="USE_MEDICINE", actor_id=None,
                     target_id=None, timing_group="simultaneous", mapped_counter_interval_ms=[950, 1000])
    rows = [dict(prediction(1, 1000, "L2"), decision_ready_ms=1400),
            prediction(2, 1600), prediction(3, 3000), prediction(4, 100)]
    report = event_metrics(rows, [reference], [], [0, 3500], 200, complete_window=[500, 2500])
    assert report["attribution"]["owner_correct"]["compared"] == 0
    assert report["recorded_decision_offsets"]["compared"] == 0
    assert report["verified_false_positives"] == [rows[1]]
    assert report["out_of_overlap_prediction_count"] == 2


def test_overlapping_references_cannot_consume_duplicate_predictions(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from evaluate_phase6 import event_metrics

    first = dict(sample_id="r1", owner_id="R2", actor_id=None, target_id=None,
                 event_type="USE_MEDICINE", mapped_counter_interval_ms=[950, 1000])
    second = dict(first, sample_id="r2", owner_id="R3")
    rows = [prediction(i, 1000) for i in (1, 2, 3)]
    report = event_metrics(rows, [first, second], [], [0, 3000], 200, complete_window=[0, 3000])
    assert {match["reference"]["sample_id"] for match in report["matches"]} == {"r1", "r2"}
    assert len(report["matches"]) == 2
    assert len(report["verified_false_positives"]) == 1
    assert report["misses"] == []


def test_wrong_side_time_match_is_a_side_false_positive_and_miss(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from evaluate_phase6 import event_metrics, event_metrics_by_side

    rows = [prediction(1, 1000, "L4", "L4_PET")]
    references = [dict(sample_id="r1", owner_id="R2", actor_id=None, target_id=None,
                       event_type="USE_MEDICINE", timing_group="simultaneous",
                       mapped_counter_interval_ms=[950, 1000])]
    overall = event_metrics(rows, references, [], [0, 3000], 500, complete_window=[0, 3000])
    sides = event_metrics_by_side(rows, references, [], [0, 3000], 500, complete_window=[0, 3000])
    assert overall["by_event_type"]["USE_MEDICINE"]["hits"] == 1
    assert overall["attribution"]["owner_correct"]["compared"] == 0
    assert sides["L"]["verified_false_positive_ids"] == [1]
    assert sides["L"]["by_event_type"]["USE_MEDICINE"]["hits"] == 0
    assert sides["R"]["missed_sample_ids"] == ["r1"]
    assert sides["R"]["by_event_type"]["USE_MEDICINE"]["hits"] == 0


def test_owner_and_time_score_can_match_a_farther_correct_owner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from evaluate_phase6 import event_metrics

    reference = dict(sample_id="use", owner_id="L4", actor_id=None, target_id=None,
                     event_type="USE_MEDICINE", mapped_counter_interval_ms=[950, 1000])
    wrong_owner = prediction(1, 1000, "L2", "L2_PET")
    right_owner = prediction(2, 1140, "L4", "L4_PET")
    report = event_metrics([wrong_owner, right_owner], [reference], [], [0, 3000], 200)
    assert report["matches"][0]["prediction"] == wrong_owner
    assert report["by_event_type"]["USE_MEDICINE"]["owner_and_time_hits"] == 1

    wrong_only = event_metrics([wrong_owner], [reference], [], [0, 3000], 200)
    assert wrong_only["by_event_type"]["USE_MEDICINE"]["hits"] == 1
    assert wrong_only["by_event_type"]["USE_MEDICINE"]["owner_and_time_hits"] == 0
