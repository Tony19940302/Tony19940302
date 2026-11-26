import json

import pytest


def observe_sequence(*, already_present=False, rays=True, actor_alive=True, last=1600,
                     owner="L2", pet_slot=None, verified=True, participant_view=False):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from battletracker.summon_vision import SummonObserver

    config = {"width": 700, "height": 320, "pets": {pet_slot or owner: (220, 135)},
              "players": {owner: (175, 104)}, "participant_view": participant_view}
    if pet_slot is not None:
        config["formation_slots"] = {owner[0]: {
            "players": {owner: 2}, "pets": {pet_slot: 2},
            "verified_pets": [pet_slot] if verified else [],
        }}
    detector = SummonObserver(config)
    for time in range(0, last + 1, 80):
        image = np.zeros((320, 700, 3), np.uint8)
        if actor_alive:
            cv2.rectangle(image, (161, 102), (188, 106), (230, 43, 43), -1)
        if 720 <= time <= 960:
            cv2.ellipse(image, (210, 129), (30, 10), 0, 0, 360, (255, 181, 43), 2)
            if rays:
                for x in (193, 202, 213, 225):
                    cv2.line(image, (x, 140), (x, 162), (255, 181, 43), 2)
        if time >= 800 or already_present:
            cv2.rectangle(image, (206, 133), (233, 137), (230, 43, 43), -1)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        detector.observe(time, image, hsv, {owner: 0.30 if time == 720 else 0.0})
    return detector.events()


def test_entry_ring_requires_a_persistent_bar_and_reports_the_observed_decision_time():
    event, = observe_sequence()
    assert json.loads(json.dumps(event)) == event
    assert event["event_type"] == "SUMMON_PET"
    assert event["owner_id"] == event["actor_id"] == "L2"
    assert event["target_id"] == "L2_PET"
    assert event["evidence"]["health_bar_first_ms"] == 800
    assert event["decision_ready_ms"] == 1440
    assert 800 <= event["time_ms"] <= 960


def test_entry_observes_the_calibrated_pet_slot_but_attributes_its_actual_owner():
    event, = observe_sequence(owner="L5", pet_slot="L4")
    assert (event["actor_id"], event["owner_id"], event["target_id"]) == ("L5", "L5", "L4_PET")
    assert observe_sequence(owner="L5", pet_slot="L4", verified=False) == []


def test_ring_around_an_existing_pet_or_unfinished_tail_is_not_an_entry():
    assert observe_sequence(already_present=True) == []
    assert observe_sequence(last=1320) == []


@pytest.mark.parametrize("gap_starts,returns,replacement,last,guard,expected", [
    (2240, 2640, False, 3280, True, "UNKNOWN"),
    (1840, 2640, False, 3280, True, "SUMMON_PET"),
    (2240, 2800, False, 3280, True, "SUMMON_PET"),
    (2240, 2640, True, 3280, True, "SUMMON_PET"),
    (2240, 2640, False, 2880, True, "SUMMON_PET"),
    (2240, 2640, False, 3280, False, "SUMMON_PET"),
])
def test_retained_pet_review_needs_same_visible_name_before_and_after(gap_starts, returns, replacement, last,
                                                                guard, expected):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from battletracker.summon_vision import SummonObserver, merge_summons

    config = {"width": 700, "height": 320, "pets": {"L2": (220, 135)},
              "players": {"L5": (175, 104)}, "participant_view": True, "summon_guard_retained": guard,
              "formation_slots": {"L": {"players": {"L5": 2}, "pets": {"L2": 2}, "verified_pets": ["L2"]}}}
    detector = SummonObserver(config)
    for time in range(0, last + 1, 80):
        image = np.zeros((320, 700, 3), np.uint8)
        if time < gap_starts or time >= returns:
            cv2.circle(image, (218, 170), 12, (215, 110, 180), -1)
            label = "XX" if replacement and time >= returns else "TYPE"
            cv2.putText(image, label, (198, 204), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (15, 220, 235), 1)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        detector.observe(time, image, hsv, {})
    original = dict(event_type="SUMMON_PET", actor_id="L5", owner_id="L5", target_id="L2_PET",
                    status="candidate", start_ms=2320, end_ms=2400, time_ms=2400, decision_ready_ms=2480,
                    evidence={"entry_ray_samples": 1})
    strong = dict(original, evidence={"vacant_slot_gold_beam": True})
    wrong_owner = dict(original, owner_id="L2")
    only_template = dict(original, target_id=None)
    revised, preserved, unassigned, inferred = detector.review_retained_entries(
        [original, strong, wrong_owner, only_template])
    assert original["event_type"] == preserved["event_type"] == unassigned["event_type"] == "SUMMON_PET"
    assert revised["event_type"] == inferred["event_type"] == expected
    assert revised["time_ms"] == original["time_ms"]
    if expected == "UNKNOWN":
        continuity = revised["evidence"]["retained_pet"]
        assert continuity["slot_id"] == "L2_PET"
        assert continuity["near_cue_ms"][-1] == 2160
        assert revised["decision_ready_ms"] >= max(continuity["after_ms"][-1], continuity["reference_ms"])
        assert "retained_pet" not in original["evidence"]
        detector._late_events.append(strong)
        merged = merge_summons([original], detector.events())
        assert merged == [original]
        assert detector.review_retained_entries(merged) == merged
        skill = dict(event_type="SKILL", actor_id="L2_PET", skill="死亡召唤",
                     start_ms=2660, end_ms=2660, time_ms=2660)
        merged = merge_summons([original, skill], detector.events())
        assert merged == [original, skill]
        assert detector.review_retained_entries(merged) == [revised, skill]


def test_unaccompanied_arc_and_entry_with_downed_owner_remain_unconfirmed():
    assert observe_sequence(rays=False) == []
    assert observe_sequence(actor_alive=False) == []


def participant_sequence(*, red_bars=False, model_start=800, model_end=1760, move=True, rays=True,
                         last=1760, track_identity=False, late_settle=False,
                         ring_windows=((720, 960),), motion_times=(720,), mapped=False):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from battletracker.summon_vision import SummonObserver

    owner = "L5" if mapped else "L2"
    config = {"width": 700, "height": 320, "participant_view": True,
              "pets": {"L2": (220, 135)}, "players": {owner: (175, 104)},
              "summon_track_identity": track_identity, "summon_late_settle": late_settle}
    if mapped:
        config["formation_slots"] = {"L": {"players": {owner: 2}, "pets": {"L2": 2}, "verified_pets": ["L2"]}}
    observer = SummonObserver(config)
    assert observer.points == {}
    for time in range(0, last + 1, 80):
        image = np.zeros((320, 700, 3), np.uint8)
        if red_bars:
            cv2.rectangle(image, (161, 102), (188, 106), (230, 43, 43), -1)
            if time >= 800:
                cv2.rectangle(image, (206, 133), (233, 137), (230, 43, 43), -1)
        if any(start <= time <= end for start, end in ring_windows):
            cv2.ellipse(image, (210, 129), (30, 10), 0, 0, 360, (255, 181, 43), 2)
            if rays:
                for x in (193, 202, 213, 225):
                    cv2.line(image, (x, 140), (x, 162), (255, 181, 43), 2)
        if model_start <= time <= model_end:
            cv2.ellipse(image, (219, 167), (12, 10), 0, 0, 360, (30, 230, 205), -1)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        observer.observe(time, image, hsv, {owner: 0.30 if move and time in motion_times else 0.0})
    return observer.events()


def test_participant_enemy_summon_uses_visible_model_with_no_gauges():
    event, = participant_sequence()
    assert event["event_type"] == "SUMMON_PET"
    assert event["owner_id"] == event["actor_id"] == "L2"
    assert event["target_id"] == "L2_PET"
    assert event["status"] == "candidate"
    assert event["evidence"]["enemy_gauges_available"] is False
    assert "new_health_bar_samples" not in event["evidence"]
    assert event["time_ms"] == event["start_ms"] == 720
    assert event["end_ms"] == 960
    assert event["decision_ready_ms"] == 1680
    assert participant_sequence(last=1600) == []
    assert participant_sequence(red_bars=True) == [event]


def test_participant_enemy_summon_uses_mapped_player_motion_and_preserves_the_pet_slot():
    event, = participant_sequence(mapped=True)
    assert (event["actor_id"], event["owner_id"], event["target_id"]) == ("L5", "L5", "L2_PET")


def test_participant_friendly_summon_uses_its_visible_bars():
    event, = observe_sequence(owner="R2", participant_view=True)
    assert event["owner_id"] == "R2"
    assert event["evidence"]["new_health_bar_samples"] > 0
    assert event["decision_ready_ms"] == 1440


@pytest.mark.parametrize("missing", [{"model_start": 0}, {"model_end": 1120},
                                     {"model_end": 0, "red_bars": True},
                                     {"move": False}, {"rays": False}])
def test_participant_enemy_summon_needs_cast_rays_and_persistent_model_change(missing):
    assert participant_sequence(**missing) == []


@pytest.mark.parametrize("skill_name", ["高级神佑复生", "死亡召唤"])
def test_merge_keeps_consecutive_summons_and_excludes_pet_skills(skill_name):
    pytest.importorskip("cv2")
    from battletracker.summon_vision import merge_summons

    def entry(time, owner):
        return dict(event_type="SUMMON_PET", owner_id=owner, target_id=f"{owner}_PET",
                    start_ms=time - 100, end_ms=time, time_ms=time)

    old = entry(1000, "L2")
    same = entry(1220, "L2")
    other = entry(1240, "L3")
    same_slot_during_skill = entry(2500, "L2")
    skill = dict(event_type="SKILL", actor_id="L2_PET", skill=skill_name,
                 start_ms=2450, end_ms=2600, time_ms=2450)
    assert merge_summons([old, skill], [same, other, same_slot_during_skill]) == [same, other, skill]


def test_template_dedup_uses_cue_end_after_enemy_summon_timestamp_moves():
    pytest.importorskip("cv2")
    from battletracker.summon_vision import merge_summons

    event, = participant_sequence()
    template = dict(event_type="SUMMON_PET", owner_id="L2", target_id="L2_PET",
                    start_ms=1440, end_ms=1560, time_ms=1560)
    assert merge_summons([template], [event]) == [event]


def nameplate_sequence(*, replacement=True, motion=True, visibility_gap=True, enabled=True,
                       transient_model=False):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from battletracker.summon_vision import SummonObserver

    observer = SummonObserver({"width": 700, "height": 320, "participant_view": True,
                               "pets": {"L2": (220, 135)}, "players": {"L2": (175, 104)},
                               "summon_track_identity": enabled})
    for time in range(0, 1601, 80):
        image = np.zeros((320, 700, 3), np.uint8)
        if not visibility_gap or time <= 400 or time >= 800:
            changed = replacement and time >= 800 and (not transient_model or time == 1040)
            color = (220, 50, 30) if changed else (42, 220, 60)
            cv2.rectangle(image, (201, 156), (234, 179), color, -1)
            if replacement and time >= 800:
                cv2.line(image, (195, 195), (243, 195), (25, 190, 240), 2)
            else:
                for x in (195, 211, 228, 243):
                    cv2.line(image, (x, 190), (x, 198), (25, 190, 240), 1)
        activity = {"L2": 0.30 if motion and time == 720 else 0.0}
        observer.observe(time, image, cv2.cvtColor(image, cv2.COLOR_RGB2HSV), activity)
    return observer


def test_summon_identity_requires_distinct_stable_appearance_and_player_motion():
    event, = nameplate_sequence().events()
    assert event["time_ms"] == 800
    assert event["decision_ready_ms"] == 1040
    assert event["evidence"]["stable_name_shape_change"] is True
    assert nameplate_sequence(replacement=False).events() == []
    assert nameplate_sequence(motion=False).events() == []
    assert nameplate_sequence(enabled=False).events() == []


def test_name_shape_change_with_single_frame_model_effect_is_not_an_entry():
    assert nameplate_sequence(transient_model=True).events() == []


def test_continuously_visible_pet_can_veto_template_but_gap_stays_unknown():
    template = dict(event_type="SUMMON_PET", owner_id="L2", start_ms=560, end_ms=720)
    unrelated = dict(event_type="USE_MEDICINE", owner_id="L2", start_ms=560, end_ms=720)
    visible = nameplate_sequence(replacement=False, visibility_gap=False)
    occluded = nameplate_sequence(replacement=False)
    assert visible.filter_events([template, unrelated]) == [unrelated]
    assert occluded.filter_events([template, unrelated]) == [template, unrelated]


def test_continuously_visible_pet_uses_the_event_target_slot_after_binding():
    visible = nameplate_sequence(replacement=False, visibility_gap=False)
    visible.pet_owners = {"L2": "L5"}
    template = dict(event_type="SUMMON_PET", owner_id="L5", target_id="L2_PET", start_ms=560, end_ms=720)
    assert not visible.filter_events([template])
    del template["target_id"]
    assert not visible.filter_events([template])


def test_summon_identity_can_wait_for_a_model_to_settle_after_the_short_window():
    assert participant_sequence(model_start=1920, model_end=2240, last=2240) == []
    event, = participant_sequence(model_start=1920, model_end=2240, last=2240, track_identity=True)
    assert event["time_ms"] == 720
    assert event["decision_ready_ms"] == 2240


def test_late_settle_preserves_the_short_decision_and_requires_a_verified_ring():
    assert participant_sequence(last=2400, model_end=2400, late_settle=True) == participant_sequence(
        last=2400, model_end=2400)
    frames = dict(model_start=1920, model_end=2240, last=2240)
    assert participant_sequence(**frames) == []
    event, = participant_sequence(**frames, late_settle=True)
    assert event["time_ms"] == 720
    assert event["decision_ready_ms"] == 2240
    assert event["evidence"]["late_model_settle"] is True
    assert participant_sequence(**frames, late_settle=True, rays=False) == []
    assert participant_sequence(**{**frames, "last": 2160}, late_settle=True) == []


def test_late_candidate_does_not_displace_a_subsequent_default_decision():
    frames = dict(model_start=1920, model_end=2560, last=2560,
                  ring_windows=((720, 960), (1440, 1680)), motion_times=(720, 1440))
    event, = participant_sequence(**frames)
    assert event["time_ms"] == 1440
    assert event["decision_ready_ms"] == 2400
    assert participant_sequence(**frames, late_settle=True) == [event]


def test_late_candidate_does_not_displace_template_only_output():
    pytest.importorskip("cv2")
    from battletracker.summon_vision import merge_summons

    event, = participant_sequence(model_start=1920, model_end=2240, last=2240, late_settle=True)
    template = dict(event_type="SUMMON_PET", owner_id="L2", target_id="L2_PET",
                    start_ms=960, end_ms=1040, time_ms=1040)
    other = dict(template, owner_id="L3", target_id="L3_PET")
    assert merge_summons([template], [event]) == [template]
    assert merge_summons([other], [event]) == [event, other]


def test_vacant_beam_requires_prior_vacancy_and_waits_for_a_settled_pet(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from battletracker import summon_vision

    config = {"width": 700, "height": 320, "participant_view": True, "summon_vacant_beam": True,
              "pets": {"L3": (220, 135)}, "players": {"L2": (175, 104)},
              "formation_slots": {"L": {"players": {"L2": 1}, "pets": {"L3": 1},
                                        "verified_pets": ["L3"]}}}
    image = np.zeros((320, 700, 3), np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    monkeypatch.setattr(summon_vision, "ring_candidates", lambda *_: {})

    def replay(*, occupied=False, stays=True, last=2240, activity=True, enabled=True):
        observer = summon_vision.SummonObserver(dict(config, summon_vacant_beam=enabled))

        def beam(*_):
            textured = occupied if time < 720 else stays
            area = 850 if 720 <= time <= 880 else 0
            return {"L3": (area, 4 if area else 0, 32 if textured else 9, 0.4 if textured else 0.04)}

        monkeypatch.setattr(summon_vision, "entry_beam_detail", beam)
        monkeypatch.setattr(summon_vision, "model_signature", lambda *_: np.array([1, 0] if time < 720 else [0, 1]))
        for time in range(0, last + 1, 80):
            observer.observe(time, image, hsv, {"L2": 0.3 if activity and time == 800 else 0.0})
        return observer.events()

    event, = replay()
    assert (event["actor_id"], event["owner_id"], event["target_id"]) == ("L2", "L2", "L3_PET")
    assert event["time_ms"] == 720 and event["decision_ready_ms"] == 2240
    assert event["evidence"]["enemy_gauges_available"] is False
    assert replay(last=2160) == []
    assert replay(occupied=True) == []
    assert replay(stays=False) == []
    assert replay(activity=False) == []
    assert replay(enabled=False) == []


def test_vacant_beam_does_not_duplicate_a_long_overlapping_template():
    pytest.importorskip("cv2")
    from battletracker.summon_vision import merge_summons

    beam = dict(event_type="SUMMON_PET", owner_id="L1", target_id="L5_PET",
                start_ms=1000, end_ms=1240, time_ms=1000, evidence={"vacant_slot_gold_beam": True})
    template = dict(event_type="SUMMON_PET", owner_id="L1", target_id="L5_PET",
                    start_ms=1080, end_ms=1950, time_ms=1950)
    separate = dict(template, start_ms=2110, end_ms=2300, time_ms=2300)
    assert merge_summons([template, separate], [beam]) == [template, separate]
    assert merge_summons([separate], [beam]) == [beam, separate]


def test_vacant_beam_return_needs_same_new_name_and_consecutive_empty_frames(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from battletracker import summon_vision

    config = {"width": 700, "height": 320, "participant_view": True, "summon_vacant_beam": True,
              "pets": {"L2": (220, 135)}, "players": {"L2": (175, 104)}}
    image = np.zeros((320, 700, 3), np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    name = np.zeros((12, 30), dtype=np.uint8)
    name[3:9, 5:24] = 1
    other = np.zeros_like(name)
    other[5:9, 24:28] = 1
    monkeypatch.setattr(summon_vision, "ring_candidates", lambda *_: {})

    def replay(*, last=3840, prior_same=False, late_shape=name, split_gap=False, enabled=True, new_beam=False):
        observer = summon_vision.SummonObserver(dict(config, summon_beam_return=enabled))
        returned = 3360 if new_beam else 3120

        def beam(*_):
            textured = time >= returned or (split_gap and 1940 <= time < 2680 and time % 240 == 0)
            area = 850 if 1600 <= time <= 1760 or (new_beam and 3120 <= time <= 3280) else 0
            return {"L2": (area, 4 if area else 0, 32 if textured else 9, 0.4 if textured else 0.04)}

        def label(*_, **_options):
            if 1440 <= time <= 1680 or (prior_same and time == 720):
                return name
            return late_shape if time >= returned else None

        monkeypatch.setattr(summon_vision, "entry_beam_detail", beam)
        monkeypatch.setattr(summon_vision, "nameplate_shape", label)
        monkeypatch.setattr(summon_vision, "model_signature", lambda *_: np.array([1, 0] if time < returned else [0, 1]))
        for time in range(0, last + 1, 80):
            activity = time == 1600 or (new_beam and time == 3120)
            observer.observe(time, image, hsv, {"L2": 0.3 if activity else 0.0})
        return observer.events()

    event, = replay()
    assert (event["owner_id"], event["target_id"], event["time_ms"], event["decision_ready_ms"]) == (
        "L2", "L2_PET", 1600, 3840,
    )
    assert event["evidence"]["matching_name_on_return"] is True
    assert replay(last=3760) == []
    assert replay(prior_same=True) == []
    assert replay(late_shape=other) == []
    assert replay(late_shape=None) == []
    assert replay(split_gap=True) == []
    assert replay(enabled=False) == []
    next_entry, = replay(last=4640, new_beam=True)
    assert next_entry["time_ms"] == 3120
    assert "matching_name_on_return" not in next_entry["evidence"]


def test_beam_return_has_lower_priority_than_an_existing_observed_entry():
    pytest.importorskip("cv2")
    from battletracker.summon_vision import SummonObserver

    observer = SummonObserver({"width": 700, "height": 320, "pets": {}, "players": {}})
    fallback = dict(owner_id="L2", start_ms=1600, evidence={"matching_name_on_return": True})
    original = dict(owner_id="L2", start_ms=2720, evidence={"vacant_slot_gold_beam": True})
    separate = dict(original, start_ms=3200)
    observer._late_events = [fallback, original]
    assert observer.events() == [original]
    observer._late_events = [fallback, separate]
    assert observer.events() == [separate, fallback]


def test_identity_merge_uses_observed_ring_interval_not_a_broad_owner_window():
    pytest.importorskip("cv2")
    from battletracker.summon_vision import merge_summons

    def entry(owner, start, end):
        return dict(event_type="SUMMON_PET", owner_id=owner, target_id=f"{owner}_PET",
                    start_ms=start, end_ms=end, time_ms=end)

    template = entry("L2", 900, 1100)
    separate = entry("L2", 1900, 1940)
    neighbor = entry("L3", 900, 1100)
    update = entry("L2", 2100, 2100)
    update["evidence"] = dict(stable_name_shape_change=True, identity_ring_interval_ms=[980, 1140])

    assert merge_summons([template, separate, neighbor], [update]) == [neighbor, separate, update]
    del update["evidence"]["identity_ring_interval_ms"]
    assert merge_summons([template, separate], [update]) == [template, separate, update]
