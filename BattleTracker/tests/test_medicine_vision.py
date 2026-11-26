from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

from battletracker.medicine_vision import FloatingNumber, MedicineObserver, NumberTrack
from battletracker.vision import Cue, classify


def observe_medicine(*, ambiguous=False, rising=True, unverified=False, amber=True,
                     allow_weak_amber=False, empty_before=False, visible_impact=False,
                     sustained_recovery=True, participant_view=False, actor="L2_PET",
                     action_first=False, duration_ms=2080):
    observer = MedicineObserver(dict(
        players={"L1": [100, 100], "L2": [180, 100], "R1": [500, 100]},
        pets={"L2": [220, 120]}, unverified_pet_owners=["L2"] if unverified else [],
        medicine_allow_weak_amber=allow_weak_amber,
        medicine_action_first=action_first,
        participant_view=participant_view,
    ))
    for time in range(0, duration_ms, 80):
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


def test_opt_in_action_first_can_connect_a_stationary_number_to_one_earlier_caster():
    observer = observe_medicine(rising=False, action_first=True)
    event, = observer.classify([])
    assert (event["event_type"], event["actor_id"], event["owner_id"], event["target_id"]) == (
        "USE_MEDICINE", "L2_PET", "L2", "L1")
    assert event["evidence"]["action_first"]
    assert event["start_ms"] < event["time_ms"] <= event["decision_ready_ms"]
    assert event["decision_ready_ms"] >= event["start_ms"] + 860 + 660


def test_action_first_keeps_ambiguity_and_known_skill_overlaps_out_of_resources():
    assert observe_medicine(rising=False, ambiguous=True, action_first=True).classify([]) == []
    observer = observe_medicine(rising=False, action_first=True)
    skills = [("L2_PET", "healing skill", Cue(720, 1000, 1.0))]
    assert observer.classify(skills) == []


def test_action_first_leaves_existing_medicine_credited_once():
    expected = observe_medicine().classify([])
    actual = observe_medicine(action_first=True).classify([])
    assert actual == expected


def test_action_first_waits_for_and_respects_nearby_same_team_credit():
    assert observe_medicine(rising=False, action_first=True, duration_ms=1920).classify([]) == []
    observer = observe_medicine(rising=False, action_first=True)
    overlapping = dict(event_type="USE_MEDICINE", actor_id="L1", owner_id="L1",
                       target_id="L2", time_ms=1120, decision_ready_ms=1840)
    assert observer._classify_caster_motion([overlapping], [], []) == []
    other_side = dict(overlapping, actor_id="R1", owner_id="R1", target_id="R1")
    assert len(observer._classify_caster_motion([other_side], [], [])) == 1


def test_action_first_excludes_overlapping_legacy_summon_without_inventing_readiness():
    observer = observe_medicine(rising=False, actor="L2", action_first=True)
    event, = observer.classify([])
    summon = dict(event_type="SUMMON_PET", actor_id="L2", time_ms=480, end_ms=480)
    assert observer.classify([], summons=[summon]) == []
    later = dict(summon, decision_ready_ms=event["decision_ready_ms"] + 80)
    assert observer.classify([], summons=[later]) == [event]


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


@pytest.mark.parametrize("action_first", [False, True])
def test_medicine_trigger_keeps_the_action_and_leaves_consumption_unconfirmed(action_first):
    observer = observe_medicine(rising=not action_first, action_first=action_first)
    hint = [("L2_PET", "寸草心", Cue(1010, 1150, 1.0))]
    action, = observer.classify(hint)
    assert (action["event_type"], action["actor_id"], action["target_id"]) == ("USE_MEDICINE", "L2_PET", "L1")
    assert action["payload"] == {"medicine_consumption": "unknown", "medicine_trigger_hints": ["寸草心"]}
    for actor in ("L1", None):
        unrelated, = observer.classify([(actor, "寸草心", hint[0][2])])
        assert unrelated["event_type"] == "USE_MEDICINE"
        assert "payload" not in unrelated


def test_legacy_healing_observation_does_not_treat_medicine_trigger_as_a_skill_veto():
    timeline = [(800, {"L2_PET": 0.4, "L1": 0.0}, {})]
    cues = [(("skill", "L2_PET", "寸草心"), Cue(920, 1200, 1.0))]
    actions = classify(timeline, cues, [("L1", Cue(1000, 1200, 200))], {})
    assert [row["event_type"] for row in actions] == ["SKILL", "USE_MEDICINE"]
    assert actions[1]["payload"]["medicine_consumption"] == "unknown"


@pytest.mark.parametrize("caption_ms", [1050, 1081])
def test_one_passive_onset_does_not_move_to_the_next_recovery(monkeypatch, caption_ms):
    observer = MedicineObserver(dict(players={"L1": (100, 100), "L2": (180, 100)},
                                     pets={"L2": (220, 120)}, participant_view=True))
    observer.times = list(range(0, 2200, 20))
    monkeypatch.setattr(observer, "_rank_targets",
                        lambda first: [(0.4, "L1", 0.2, 0.2, None, first.time_ms - 100)])
    monkeypatch.setattr(observer, "_rank_actors", lambda *_: [(0.3, "L2_PET"), (0.1, "L2")])
    observer.completed = [NumberTrack([FloatingNumber(t + i * 80, 100, 90 - i * 2, 200) for i in range(3)])
                          for t in (1000, 1300)]
    first, second = observer.classify([("L2_PET", "寸草心", Cue(caption_ms, 1350, 1.0))])
    assert (first["time_ms"], second["time_ms"]) == (1000, 1300)
    assert first["payload"]["medicine_trigger_hints"] == ["寸草心"]
    assert "payload" not in second


def test_ambiguous_caption_keeps_candidates_pending_without_binding_to_either(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from predict_clean import check_pet_ownership

    timeline = [(800, {"L2_PET": 0.4, "L1": 0.0}, {})]
    cues = [(("skill", "L2_PET", "寸草心"), Cue(1140, 1480, 1.0))]
    actions = classify(timeline, cues, [("L1", Cue(1000, 1150, 200)), ("L3", Cue(1200, 1350, 200))], {})
    assert sum(row["event_type"] == "SKILL" for row in actions) == 1
    medicines = [row for row in actions if row["event_type"] == "USE_MEDICINE"]
    assert len(medicines) == 2
    for row in medicines:
        assert row["payload"] == {"medicine_consumption": "unknown"}
        assert row["evidence"]["ambiguous_medicine_trigger"] == ["寸草心"]
    check_pet_ownership(actions, {"unverified_pet_owners": ["L2"]})
    assert all(row["event_type"] == "UNKNOWN" and "payload" not in row for row in medicines)


def test_cyan_overlap_can_continue_a_rising_number_only_when_enabled():
    observers = [MedicineObserver(dict(players={"L1": (100, 90)}, pets={},
                                       medicine_recover_cyan_overlap=enabled)) for enabled in (False, True)]
    for time, y in ((0, 70), (80, 67), (160, 64)):
        hsv = np.zeros((160, 160, 3), dtype=np.uint8)
        hsv[y:y + 8, 89:111] = (60, 220, 230)
        if time == 160:
            hsv[y - 11:y, 95:99] = (80, 220, 230)
        for observer in observers:
            observer.observe(time, None, hsv, {})
    baseline, recovered = (observer.active[0] for observer in observers)
    assert baseline.samples == recovered.samples[:2]
    assert [sample.time_ms for sample in recovered.samples] == [0, 80, 160]
    assert recovered.samples[-1].y < recovered.samples[1].y


def test_cyan_overlap_does_not_create_or_choose_between_tracks():
    observer = MedicineObserver(dict(players={"L1": (100, 90)}, pets={}, medicine_recover_cyan_overlap=True))
    hsv = np.zeros((160, 160, 3), dtype=np.uint8)
    hsv[64:72, 89:111] = (60, 220, 230)
    hsv[53:64, 95:99] = (80, 220, 230)
    observer.observe(0, None, hsv, {})
    assert not observer.active
    observer.active = [NumberTrack([FloatingNumber(0, x, 76, 150), FloatingNumber(80, x, 73, 150)])
                       for x in (98, 102)]
    observer.observe(160, None, hsv, {})
    assert [len(track.samples) for track in observer.active] == [2, 2]


def test_cyan_overlap_cannot_continue_beyond_the_original_player_region():
    observer = MedicineObserver(dict(players={"L1": (100, 90)}, pets={}, medicine_recover_cyan_overlap=True))
    observer.active = [NumberTrack([FloatingNumber(0, 124, 76, 150), FloatingNumber(80, 124, 73, 150)])]
    hsv = np.zeros((160, 160, 3), dtype=np.uint8)
    hsv[64:72, 117:139] = (60, 220, 230)
    hsv[53:64, 122:126] = (80, 220, 230)
    observer.observe(160, None, hsv, {})
    assert len(observer.active[0].samples) == 2


@pytest.mark.parametrize("second_recovery_y, expected_times", [(65, [0, 80, 160]), (69, [0])])
def test_cyan_overlap_with_one_clear_seed_needs_two_recovered_frames_and_ascent(second_recovery_y, expected_times):
    observer = MedicineObserver(dict(players={"L1": (100, 90)}, pets={}, medicine_recover_cyan_overlap=True))
    for time, y in ((0, 70), (80, 69), (160, second_recovery_y)):
        hsv = np.zeros((160, 160, 3), dtype=np.uint8)
        hsv[y:y + 8, 89:111] = (60, 220, 230)
        if time:
            hsv[y - 11:y, 95:99] = (80, 220, 230)
        observer.observe(time, None, hsv, {})
        if time == 80:
            assert len(observer.active[0].samples) == 1
    assert [sample.time_ms for sample in observer.active[0].samples] == expected_times


def test_fresh_clear_number_replaces_unconfirmed_overlap_samples():
    observer = MedicineObserver(dict(players={"L1": (100, 90)}, pets={}, medicine_recover_cyan_overlap=True))
    for time, y in ((0, 70), (80, 68), (160, 66)):
        hsv = np.zeros((160, 160, 3), dtype=np.uint8)
        hsv[y:y + 8, 89:111] = (60, 220, 230)
        if time == 80:
            hsv[y - 11:y, 95:99] = (80, 220, 230)
        observer.observe(time, None, hsv, {})
    track, = observer.active
    assert [sample.time_ms for sample in track.samples] == [0, 160]
    assert not track.pending_overlap


@pytest.mark.parametrize("starting_frames", [((0, 70), (80, None)), ((0, 70), (80, 68), (160, None))])
def test_cyan_overlap_with_weak_seed_cannot_bridge_a_missing_frame(starting_frames):
    observer = MedicineObserver(dict(players={"L1": (100, 90)}, pets={}, medicine_recover_cyan_overlap=True))
    final = starting_frames[-1][0]
    for time, y in (*starting_frames, (final + 80, 67), (final + 160, 65)):
        hsv = np.zeros((160, 160, 3), dtype=np.uint8)
        if y is not None:
            hsv[y:y + 8, 89:111] = (60, 220, 230)
            if time:
                hsv[y - 11:y, 95:99] = (80, 220, 230)
        observer.observe(time, None, hsv, {})
    track, = (*observer.active, *observer.completed)
    assert [sample.time_ms for sample in track.samples] == [0]
    assert not track.pending_overlap


def test_rescued_cyan_overlap_track_cannot_swallow_a_later_clear_wave():
    observer = MedicineObserver(dict(players={"L1": (100, 90)}, pets={}, medicine_recover_cyan_overlap=True))
    for time, y in ((0, 70), (80, 68), (160, 67), (240, None), (320, 65), (400, 63), (480, 61)):
        hsv = np.zeros((160, 160, 3), dtype=np.uint8)
        if y is not None:
            hsv[y:y + 8, 89:111] = (60, 220, 230)
            if time in (80, 160):
                hsv[y - 11:y, 95:99] = (80, 220, 230)
        observer.observe(time, None, hsv, {})
    first, = observer.completed
    second, = observer.active
    assert [sample.time_ms for sample in first.samples] == [0, 80, 160]
    assert [sample.time_ms for sample in second.samples] == [320, 400, 480]


def test_pet_body_check_reranks_recovered_tracks_without_overriding_skill_or_recent_entry(monkeypatch):
    observer = observe_medicine()
    track, = [row for row in (*observer.completed, *observer.active) if row.samples[0].time_ms == 960]
    observer.pet_body_points = {"L2_PET": (220, 120)}
    observer.times.insert(0, -80)
    observer.frames.insert(0, replace(observer.frames[0], time_ms=-80))
    observer.frames = [replace(frame, pet_body_detail={"L2_PET": (0.02, 12.0)}) for frame in observer.frames]

    def ranks(*_, excluded=()):
        return [(0.4, "L2")] if "L2_PET" in excluded else [(0.4, "L2_PET"), (0.1, "L2")]

    monkeypatch.setattr(observer, "_rank_actors", ranks)
    assert observer.classify([])[0]["actor_id"] == "L2_PET"

    track.recovered_from_weak_seed = True
    changed, = observer.classify([])
    assert (changed["actor_id"], changed["owner_id"]) == ("L2", "L2")
    assert changed["evidence"]["low_detail_pet_slots"] == ["L2_PET"]

    entry = dict(event_type="SUMMON_PET", owner_id="L2", actor_id="L2", target_id="L2_PET",
                 time_ms=350, start_ms=350, end_ms=450, decision_ready_ms=1520)
    protected, = observer.classify([], summons=[entry])
    assert protected["actor_id"] == "L2_PET"
    assert protected["decision_ready_ms"] >= entry["decision_ready_ms"]
    assert observer.classify([], summons=[dict(entry, decision_ready_ms=1800)])[0]["actor_id"] == "L2"

    explained, = observer.classify([("L2_PET", "healing skill", Cue(720, 1000, 1.0))])
    assert explained["event_type"] == "NON_MEDICINE"
    passive, = observer.classify([("L2_PET", "寸草心", Cue(980, 1140, 1.0))])
    assert passive["actor_id"] == "L2_PET"
    assert passive["payload"]["medicine_consumption"] == "unknown"


def test_pet_body_check_requires_four_readable_frames():
    observer = observe_medicine()
    track, = [row for row in (*observer.completed, *observer.active) if row.samples[0].time_ms == 960]
    track.recovered_from_weak_seed = True
    observer.pet_body_points = {"L2_PET": (220, 120)}
    observer.frames = [replace(frame, pet_body_detail={"L2_PET": (0.02, 12.0)}) if i < 3 else frame
                       for i, frame in enumerate(observer.frames)]
    assert observer._pet_body_exclusions(track, "L1", [], 1400) == set()


def test_unresolved_multi_target_action_keeps_trigger_as_evidence_only():
    observer = MedicineObserver(dict(players={}, pets={}))
    observer.times = list(range(0, 2400, 40))
    first = dict(event_type="USE_MEDICINE", actor_id="L2_PET", owner_id="L2", target_id="L1",
                 time_ms=960, decision_ready_ms=1400, evidence={},
                 payload={"medicine_consumption": "unknown", "medicine_trigger_hints": ["寸草心"]})
    second = dict(first, time_ms=1000, target_id="L2", payload={}, evidence={})
    unresolved, = observer._resolve_nearby_tracks([first, second])
    assert unresolved["event_type"] == "UNKNOWN"
    assert unresolved["evidence"]["medicine_trigger_hints"] == ["寸草心"]
    assert "payload" not in unresolved


def test_unverified_legacy_owner_drops_the_medicine_consumption_payload(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from predict_clean import check_pet_ownership

    timeline = [(800, {"L2_PET": 0.4, "L1": 0.0}, {})]
    cues = [(("skill", "L2_PET", "寸草心"), Cue(920, 1200, 1.0))]
    actions = classify(timeline, cues, [("L1", Cue(1000, 1200, 200))], {})
    check_pet_ownership(actions, {"unverified_pet_owners": ["L2"]})
    assert actions[1]["event_type"] == "UNKNOWN"
    assert actions[1]["evidence"]["medicine_trigger_hints"] == ["寸草心"]
    assert "payload" not in actions[1]


def test_mapped_left_pet_uses_independent_owner_and_receiver_geometry(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "tools"))
    from predict_clean import check_pet_ownership, save_predictions

    config = json.loads((root / "configs/vision/dragon-tiger.json").read_text(encoding="utf-8"))
    probe = MedicineObserver(config)
    assert {actor: owner for actor, owner in probe.actor_owners.items()
            if actor.startswith("L") and actor.endswith("_PET")} == {
        "L1_PET": "L3", "L2_PET": "L4", "L3_PET": "L2", "L4_PET": "L5", "L5_PET": "L1",
    }
    assert not probe.unverified_actors

    # Exercise the guard with an intentionally incomplete calibration.
    config["formation_slots"]["L"]["verified_pets"] = ["L2", "L4"]

    timeline = [(800, {"L4_PET": 0.4, "L1": 0.0}, {})]
    entries = [(("skill", "L5_PET", "test"), Cue(100, 200, 1.0))]
    rows = classify(timeline, entries, [("L4", Cue(1000, 1200, 200))], config)
    check_pet_ownership(rows, config)
    unknown, medicine = rows
    assert unknown["event_type"] == "UNKNOWN" and unknown["owner_id"] is None
    assert (medicine["actor_id"], medicine["owner_id"], medicine["target_id"]) == ("L4_PET", "L5", "L4")
    unresolved, = classify([(800, {"L3_PET": 0.4, "L1": 0.0}, {})], [],
                           [("L4", Cue(1000, 1200, 200))], config)
    assert (unresolved["event_type"], unresolved["actor_id"], unresolved["owner_id"]) == (
        "UNKNOWN", None, None,
    )
    assert unresolved["evidence"]["ownership_status"] == "unverified_pet_mapping"

    save_predictions(tmp_path, [medicine], dict(config, participant_view=False),
                     {"asset_id": "sample", "sha256": "sample-sha", "battle_id": "sample-battle"}, {})
    exported = json.loads((tmp_path / "predictions.jsonl").read_text(encoding="utf-8"))
    assert exported["formation_slots"] == {"actor": 3, "owner": 3, "target": 2}


def test_verified_pet_number_cannot_be_attached_to_a_different_team():
    config = dict(players={"R1": [2, 2]}, pets={"L2": [3, 3]},
                  formation_slots={"L": {"players": {}, "pets": {"L2": 1}, "verified_pets": ["L2"]},
                                   "R": {"players": {"R1": 1}, "pets": {}, "verified_pets": []}})
    with pytest.raises(ValueError, match="no player with the same formation number"):
        MedicineObserver(config)


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
