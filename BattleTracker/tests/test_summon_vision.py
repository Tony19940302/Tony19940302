import json

import pytest


def observe_sequence(*, already_present=False, rays=True, actor_alive=True, last=1600,
                     owner="L2", participant_view=False):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from battletracker.summon_vision import SummonObserver

    detector = SummonObserver({"width": 700, "height": 320, "pets": {owner: (220, 135)},
                               "players": {owner: (175, 104)}, "participant_view": participant_view})
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


def test_ring_around_an_existing_pet_or_unfinished_tail_is_not_an_entry():
    assert observe_sequence(already_present=True) == []
    assert observe_sequence(last=1320) == []


def test_unaccompanied_arc_and_entry_with_downed_owner_remain_unconfirmed():
    assert observe_sequence(rays=False) == []
    assert observe_sequence(actor_alive=False) == []


def participant_sequence(*, red_bars=False, model_start=800, model_end=1760, move=True, rays=True,
                         last=1760):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from battletracker.summon_vision import SummonObserver

    observer = SummonObserver({"width": 700, "height": 320, "participant_view": True,
                               "pets": {"L2": (220, 135)}, "players": {"L2": (175, 104)}})
    assert observer.points == {}
    for time in range(0, last + 1, 80):
        image = np.zeros((320, 700, 3), np.uint8)
        if red_bars:
            cv2.rectangle(image, (161, 102), (188, 106), (230, 43, 43), -1)
            if time >= 800:
                cv2.rectangle(image, (206, 133), (233, 137), (230, 43, 43), -1)
        if 720 <= time <= 960:
            cv2.ellipse(image, (210, 129), (30, 10), 0, 0, 360, (255, 181, 43), 2)
            if rays:
                for x in (193, 202, 213, 225):
                    cv2.line(image, (x, 140), (x, 162), (255, 181, 43), 2)
        if model_start <= time <= model_end:
            cv2.ellipse(image, (219, 167), (12, 10), 0, 0, 360, (30, 230, 205), -1)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        observer.observe(time, image, hsv, {"L2": 0.30 if move and time == 720 else 0.0})
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
