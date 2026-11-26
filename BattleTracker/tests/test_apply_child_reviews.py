from hashlib import sha256
from pathlib import Path

import pytest


def test_child_review_binds_source_and_entry_rather_than_prediction_number(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from apply_child_reviews import apply_reviews

    clip = tmp_path / "review.mp4"
    clip.write_bytes(b"evidence")
    source = dict(source_asset="ordinary", source_sha256="ordinary-hash", battle_id="battle")
    case = dict(event_type="SUMMON_CHILD", status="reviewed", sample_id="child-one", owner_id="R4",
                arrival_slot_id="R4_PET",
                display_name="小星星", first_visible_entry_ms=1000, source_window_ms=[500, 1600],
                review_media=str(clip), review_media_sha256=sha256(clip.read_bytes()).hexdigest(),
                basis_prediction_id=170)
    review = dict(source_video_id="ordinary", source_sha256="ordinary-hash", battle_id="battle", cases=[case])
    candidate = dict(source, prediction_id=173, event_type="SUMMON_PET", actor_id="R4", owner_id="R4",
                     target_id="R4_PET", time_ms=1450, start_ms=1016, evidence={"ring": True},
                     formation_slots={"actor": 4, "owner": 4, "target": 4})
    other = dict(candidate, prediction_id=174, start_ms=1870, time_ms=2300)
    rows, applied = apply_reviews([candidate, other], source, review, tolerance_ms=200)
    child = rows[0]
    assert applied[0]["prediction_id"] == 173
    assert child["event_type"] == "SUMMON_CHILD" and child["status"] == "confirmed"
    assert child["payload"]["child_name"] == "小星星" and child["target_id"] == "R4_CHILD"
    assert child["evidence"]["arrival_slot_id"] == "R4_PET"
    assert child["evidence"]["arrival_formation_slot"] == 4 and "target" not in child["formation_slots"]
    assert rows[1] == other and candidate["event_type"] == "SUMMON_PET"
    assert apply_reviews(rows, source, review, tolerance_ms=200) == (rows, applied)

    with pytest.raises(ValueError, match="share the reviewed replay"):
        apply_reviews([dict(candidate, battle_id="other")], source, review, tolerance_ms=200)
    with pytest.raises(ValueError, match="found 2"):
        apply_reviews([candidate, dict(candidate, prediction_id=175)], source, review, tolerance_ms=200)
    with pytest.raises(ValueError, match="slot differs"):
        apply_reviews([dict(candidate, target_id="R1_PET")], source, review, tolerance_ms=200)
    with pytest.raises(ValueError, match="existing child correction differs"):
        apply_reviews(rows, source, dict(review, cases=[dict(case, display_name="其他名字")]), tolerance_ms=200)
    clip.write_bytes(b"modified")
    with pytest.raises(ValueError, match="review clip does not match"):
        apply_reviews([candidate], source, review, tolerance_ms=200)
