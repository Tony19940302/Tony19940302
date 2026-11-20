import numpy as np
import pytest

pytest.importorskip("cv2")

from battletracker.medicine_vision import MedicineObserver
from battletracker.vision import Cue


def observe_medicine(*, ambiguous=False, rising=True, unverified=False, amber=True,
                     allow_weak_amber=False, empty_before=False, visible_impact=False,
                     sustained_recovery=True, participant_view=False, actor="L2_PET"):
    observer = MedicineObserver(dict(
        players={"L1": [100, 100], "L2": [180, 100], "R1": [500, 100]},
        pets={"L2": [220, 120]}, unverified_pet_owners=["L2"] if unverified else [],
        medicine_allow_weak_amber=allow_weak_amber,
        participant_view=participant_view,
    ))
    for time in range(0, 2080, 80):
        image = np.zeros((240, 540, 3), dtype=np.uint8)
        has_health = (not empty_before or time >= 960) and (sustained_recovery or time != 1040)
        if has_health:
            image[99:102, 91:103 if time < 960 else 112] = (0, 255, 220)
        motion = {}
        if 400 <= time <= 720:
            motion[actor] = 0.50
            if ambiguous:
                motion["L2"] = 0.49
        if amber and 800 <= time <= 1200:
            image[113:136, 93:107] = (17, 255, 240)
            motion["L1"] = 0.22
        if visible_impact and 880 <= time <= 1120:
            motion["L1"] = 0.35
        if 960 <= time <= 1360:
            y = 88 - (time - 960) // 80 if rising else 88
            image[y:y + 6, 90:110] = (64, 235, 240)
        observer.observe(time, image, image, motion)
    return observer


def test_medicine_uses_fixed_pet_owner_and_separate_receiver():
    events = observe_medicine().classify([])
    assert [(row["event_type"], row["actor_id"], row["owner_id"], row["target_id"]) for row in events] == [
        ("USE_MEDICINE", "L2_PET", "L2", "L1")
    ]
    event = events[0]
    assert event["decision_ready_ms"] == 1680
    assert event["status"] == "candidate"


def test_ambiguous_actor_and_unverified_pet_mapping_do_not_attribute_use():
    for observer in [observe_medicine(ambiguous=True), observe_medicine(unverified=True)]:
        event, = observer.classify([])
        assert event["event_type"] == "UNKNOWN"
        assert event["actor_id"] is None
        assert event["owner_id"] is None


def test_healing_requires_number_motion_and_a_matching_receiver_effect():
    assert observe_medicine(rising=False).classify([]) == []
    event, = observe_medicine(amber=False).classify([])
    assert event["event_type"] == "UNKNOWN"
    assert event["owner_id"] is None


def test_participant_medicine_uses_visible_effects_and_ignores_enemy_health_pixels():
    observer = observe_medicine(participant_view=True)
    assert all(set(frame.health) == {"R1"} for frame in observer.frames)
    event, = observer.classify([])
    assert (event["event_type"], event["actor_id"], event["owner_id"], event["target_id"]) == (
        "USE_MEDICINE", "L2_PET", "L2", "L1")
    assert event["evidence"]["receiver_health_gain_px"] is None


def test_participant_cannot_use_weak_effect_rule_on_an_enemy_health_bar():
    event, = observe_medicine(participant_view=True, allow_weak_amber=True, amber=False,
                              empty_before=True, visible_impact=True).classify([])
    assert event["event_type"] == "UNKNOWN"
    assert event["evidence"]["receiver_health_gain_px"] is None


def test_weak_amber_pet_route_requires_an_explicit_opt_in():
    data = dict(amber=False, empty_before=True, visible_impact=True)
    disabled, = observe_medicine(**data).classify([])
    assert disabled["event_type"] == "UNKNOWN"
    assert "persistent_hp_return_during_weak_amber" not in disabled["evidence"]

    observer = observe_medicine(allow_weak_amber=True, **data)
    enabled, = observer.classify([])
    assert (enabled["event_type"], enabled["actor_id"], enabled["owner_id"], enabled["target_id"]) == (
        "USE_MEDICINE", "L2_PET", "L2", "L1"
    )
    assert enabled["evidence"]["persistent_hp_return_during_weak_amber"]
    assert enabled["decision_ready_ms"] == 1680
    relevant = [("L2_PET", "healing skill", Cue(720, 1000, 1.0))]
    blocked, = observer.classify(relevant)
    assert blocked["event_type"] == "NON_MEDICINE"
    assert blocked["owner_id"] is None


@pytest.mark.parametrize("change", [
    {"ambiguous": True}, {"unverified": True}, {"empty_before": False},
    {"visible_impact": False}, {"sustained_recovery": False},
])
def test_weak_amber_pet_route_preserves_unknown_when_evidence_is_incomplete(change):
    data = dict(amber=False, empty_before=True, visible_impact=True, allow_weak_amber=True)
    event, = observe_medicine(**{**data, **change}).classify([])
    assert event["event_type"] == "UNKNOWN"
    assert event["actor_id"] is None


def test_only_the_earlier_action_from_the_same_actor_blocks_medicine():
    observer = observe_medicine()
    unrelated = [("R1", "nearby status", Cue(760, 950, 1.0)),
                 ("L2_PET", "next action", Cue(1100, 1300, 1.0))]
    assert observer.classify(unrelated)[0]["event_type"] == "USE_MEDICINE"
    relevant = [("L2_PET", "healing skill", Cue(720, 1000, 1.0))]
    assert observer.classify(relevant)[0]["event_type"] == "NON_MEDICINE"


def test_summon_exclusion_uses_the_ring_end_and_confirmation_time():
    observer = observe_medicine(participant_view=True, actor="L2")
    summon = dict(event_type="SUMMON_PET", actor_id="L2", time_ms=320,
                  end_ms=800, decision_ready_ms=1440)
    assert observer.classify([])[0]["event_type"] == "USE_MEDICINE"
    blocked, = observer.classify([], summons=[summon])
    assert blocked["event_type"] == "UNKNOWN"
    assert blocked["evidence"]["overlapping_summon"]
    unconfirmed, = observer.classify([], summons=[dict(summon, decision_ready_ms=1700)])
    assert unconfirmed["event_type"] == "USE_MEDICINE"


def test_nearby_multi_target_heal_stays_unknown_and_later_uses_stay_separate():
    observer = MedicineObserver(dict(players={}, pets={}))
    observer.times = list(range(0, 2400, 40))

    def candidate(time, target):
        return dict(time_ms=time, event_type="USE_MEDICINE", actor_id="L4_PET", owner_id="L4",
                    target_id=target, decision_ready_ms=time + 400, evidence={})

    nearby = observer._resolve_nearby_tracks([candidate(800, "L1"), candidate(880, "L2"),
                                             candidate(1040, "L3"), candidate(1400, "L1")])
    assert [event["event_type"] for event in nearby] == ["UNKNOWN", "USE_MEDICINE"]
    assert nearby[0]["owner_id"] is None
    assert nearby[0]["evidence"]["overlapping_receivers"]
    assert nearby[0]["evidence"]["receiver_observations"] == [
        {"target_id": "L1", "time_ms": 800},
        {"target_id": "L2", "time_ms": 880},
        {"target_id": "L3", "time_ms": 1040},
    ]
    assert nearby[1]["decision_ready_ms"] == 2080
