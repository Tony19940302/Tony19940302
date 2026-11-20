from contextlib import nullcontext
from fractions import Fraction
from pathlib import Path

import pytest


def test_cue_groups_adjacent_matches_and_tolerates_a_missing_sample():
    pytest.importorskip("cv2")
    from battletracker.vision import CueTracker

    tracker = CueTracker(gap=200)
    assert tracker.update(100, {"L1": 0.66}) == []
    assert tracker.update(180, {}) == []
    assert tracker.update(260, {"L1": 0.91}) == []
    completed = tracker.update(500, {"L1": 0.71})
    assert [(key, cue.start, cue.end, cue.peak) for key, cue in completed] == [
        ("L1", 100, 260, 0.91)
    ]
    assert tracker.update(800, {}) == []


def test_participant_frames_hide_left_gauges_before_feature_extraction(monkeypatch):
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from battletracker import vision

    layout = {"width": 700, "height": 320, "players": {"L1": (10, 3), "R1": (500, 150)},
              "pets": {"L1": (170, 120), "R1": (480, 120)}}
    first = np.full((320, 700, 3), 64, dtype=np.uint8)
    first[145:164, 482:518] = (230, 43, 43)
    first[115:126, 462:498] = (230, 43, 43)
    changed = first.copy()
    changed[0:17, 0:28] = (40, 205, 240)
    changed[115:126, 152:188] = (255, 217, 30)

    class Frame:
        time_base = Fraction(1, 1000)
        is_corrupt = False

        def __init__(self, pts, image):
            self.pts, self.image = pts, image

        def reformat(self, *, width, height, format):
            assert (height, width, format) == (320, 700, "rgb24")
            return self

        def to_ndarray(self):
            return self.image.copy()

    class Video:
        def decode(self, *, video):
            assert video == 0
            return iter((Frame(0, first), Frame(80, changed)))

    monkeypatch.setattr(vision.av, "open", lambda _: nullcontext(Video()))
    replay = list(vision.frames("unused", layout, interval_ms=80))
    np.testing.assert_array_equal(replay[0][2], first)
    np.testing.assert_array_equal(replay[1][2], changed)
    participant = list(vision.frames("unused", dict(layout, participant_view=True), interval_ms=80))
    expected = first.copy()
    expected[0:17, 0:28] = 0
    expected[115:126, 152:188] = 0
    np.testing.assert_array_equal(participant[0][2], expected)
    np.testing.assert_array_equal(participant[1][2], expected)
    assert np.all(participant[1][2][145:164, 482:518] == (230, 43, 43))


@pytest.mark.parametrize("participant_view", [False, True])
def test_observe_uses_input_gauge_visibility_with_the_template_layout(monkeypatch, participant_view):
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from battletracker import vision

    layout = {"width": 700, "height": 320, "players": {"L1": (10, 3)}, "pets": {},
              "portal_centers": {}, "thresholds": {}, "sample_ms": 80,
              "participant_view": participant_view}
    template = dict(layout, players={"L1": (300, 50)}, participant_view=not participant_view)

    def extract(path, config):
        assert path == "template"
        assert config["players"] == template["players"]
        assert config["participant_view"] is participant_view
        return []

    monkeypatch.setattr(vision, "extract_patterns", extract)
    monkeypatch.setattr(vision, "frames", lambda *args, **kwargs:
                        [(0, [], np.zeros((320, 700, 3), dtype=np.uint8))])
    timeline, cues, heals = vision.observe("source", layout, "template", template)
    assert timeline == [(0, {}, {})]
    assert not cues and not heals


def test_participant_templates_skip_hidden_skill_words_and_only_degenerate_portals(monkeypatch):
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from battletracker import vision

    death_skill = {"time_ms": 80, "rect": (100, 46, 140, 54), "name": "死亡召唤"}
    visible_skill = {"time_ms": 80, "rect": (500, 100, 517, 111), "name": "番天印"}
    hidden_portal = {"time_ms": 80, "rect": (86, 48, 99, 60)}
    visible_portal = {"time_ms": 80, "rect": (112, 56, 140, 64)}
    layout = {"width": 700, "height": 320, "players": {"L1": (100, 50)}, "pets": {},
              "portal_templates": [hidden_portal, visible_portal],
              "skill_templates": [death_skill, visible_skill]}
    image = np.zeros((320, 700, 3), dtype=np.uint8)
    image[49:52, 104:109] = 255
    image[49:52, 126:128] = 255
    image[103:107, 507:510] = 255
    image[51:55, 93:97] = (230, 165, 20)
    image[59:61, 130:134] = (230, 165, 20)

    def sampled_frame(_, config, *, requested):
        sample = image.copy()
        if config.get("participant_view"):
            sample[45:64, 82:118] = 0
        yield 80, requested, sample

    monkeypatch.setattr(vision, "frames", sampled_frame)
    replay = vision.extract_patterns("source", layout)
    assert [entry for entry, _ in replay] == [hidden_portal, visible_portal, death_skill, visible_skill]
    assert any(vision.glyph_mask(vision.cv2.cvtColor(image[46:54, 118:140],
                                                    vision.cv2.COLOR_RGB2HSV)).ravel())
    participant = vision.extract_patterns("source", dict(layout, participant_view=True))
    assert [entry for entry, _ in participant] == [visible_portal, visible_skill]
    blank_portal = {"time_ms": 80, "rect": (420, 270, 430, 280)}
    for participant_view in (False, True):
        with pytest.raises(ValueError, match="empty or solid visual template"):
            vision.extract_patterns("source", dict(layout, participant_view=participant_view,
                                                   portal_templates=[blank_portal]))


def test_evaluation_keeps_one_to_one_matches_and_unreviewed_candidates(monkeypatch):
    pytest.importorskip("cv2")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from evaluate import evaluate

    def pred(i, time, owner="L1"):
        return dict(prediction_id=i, time_ms=time, event_type="SUMMON_PET",
                    source_asset="clean", source_sha256="digest", battle_id="battle",
                    owner_id=owner, actor_id=owner, target_id=None)

    ref = dict(sample_id="sample", mapped_counter_interval_ms=[1100, 1200],
               event_type="SUMMON_PET", owner_id="L1", actor_id="L1", target_id=None,
               source_video_id="clean", source_sha256="digest", battle_id="battle",
               annotation_status="reviewed")
    preds = [pred(1, 1150, "R1"), pred(2, 1170), pred(3, 5000)]
    report = evaluate(preds, [ref], 500)
    assert [m["prediction"]["prediction_id"] for m in report["matches"]] == [1]
    assert report["metrics"]["SUMMON_PET"]["correct_owner"] == 0
    assert report["metrics"]["SUMMON_PET"]["scoped_false_positives"] == 1
    assert report["unverified_prediction_count"] == 1
    with pytest.raises(ValueError, match="different replay"):
        evaluate([dict(preds[0], source_sha256="wrong")], [ref], 500)
    empty = evaluate([], [dict(ref, actor_id=None)], 500)
    assert empty["metrics"]["SUMMON_PET"]["misses"] == 1
    assert empty["metrics"]["SUMMON_PET"]["reviewed_recall"] == 0
