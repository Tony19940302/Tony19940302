from dataclasses import replace

import pytest

from battletracker.alignment import Anchor, TimeAlignment, check_grouping


def test_piecewise_mapping_and_endpoints():
    first = Anchor("a", 1000, 2100, "first hit", 1)
    second = Anchor("b", 2000, 3100, "heal", 1)
    last = Anchor("c", 3000, 3600, "round changed", 2)
    alignment = TimeAlignment([first, second, last])
    assert alignment.map(999) is alignment.map(3001) is None
    assert [alignment.map(t) for t in (1000, 1500, 2000, 2500, 3000)] == [
        2100, 2600, 3100, 3350, 3600]
    checkpoint = Anchor("check", 2500, 3370, "damage number", 1)
    assert alignment.validate([checkpoint])[0]["error_ms"] == -20
    with pytest.raises(ValueError, match="repeats"):
        alignment.validate([second])
    with pytest.raises(ValueError, match="increase"):
        TimeAlignment([first, replace(second, clean_ms=2000)])


def test_dataset_rejects_leakage_and_detailed_clip():
    def asset(asset_id, role, split="development", battle="match"):
        return dict(asset_id=asset_id, role=role, status="available", sha256=asset_id,
                    battle_id=battle, dataset_split=split)

    clean = asset("clean", "clean_match_replay")
    detailed = asset("detail", "detailed_match_replay")
    clip = dict(source_video_id="clean", battle_id="match", dataset_split="development")
    assert check_grouping([clean, detailed], [clip]) == {"match": "development"}
    with pytest.raises(ValueError, match="crosses"):
        check_grouping([clean, dict(detailed, dataset_split="test")], [])
    with pytest.raises(ValueError, match="same content"):
        check_grouping([clean, dict(detailed, sha256="clean", battle_id="other")], [])
    with pytest.raises(ValueError, match="clean replay"):
        check_grouping([clean, detailed], [dict(clip, source_video_id="detail")])
    with pytest.raises(ValueError, match="does not match"):
        check_grouping([clean, detailed], [dict(clip, dataset_split="train")])

def test_official_source_and_output_digest_are_bound_to_detailed_video(tmp_path, monkeypatch):
    import hashlib
    import json
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from build_dataset import build

    detailed = dict(asset_id="detail", role="detailed_match_replay", status="available",
                    sha256="detailed-digest", battle_id="match", dataset_split="development")
    clean = dict(asset_id="clean", role="clean_match_replay", status="available", path="video.mp4",
                 sha256="clean-digest", battle_id="match", dataset_split="development")
    report = dict(sources={"detail": detailed["sha256"], "clean": clean["sha256"]}, version="test")
    path = tmp_path / "official_events.jsonl"
    metadata = path.with_name("counter_extraction_report.json")
    event = dict(battle_id="match", owner_id="L1", event_type="USE_MEDICINE",
                 earliest_timestamp_ms=3000, latest_timestamp_ms=3250,
                 earliest_round=2, latest_round=2, before="0/10", after="1/10", delta=1)
    path.write_text(json.dumps(event) + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    data = dict(battle_id="match", source_asset="detail", source_sha256=detailed["sha256"],
                official_events_sha256=digest)
    alignment = TimeAlignment([Anchor("a", 1000, 2000, "entry", 1),
                               Anchor("b", 2000, 3000, "heal", 2)])
    reviews = dict(version="test", changes={}, hard_negatives=[])
    args = (alignment, report, [detailed, clean], detailed, clean, path, reviews)

    metadata.write_text(json.dumps(dict(data, source_asset="clean", source_sha256=clean["sha256"])),
                        encoding="utf-8")
    with pytest.raises(ValueError, match="paired detailed"):
        build(*args)

    metadata.write_text(json.dumps(data), encoding="utf-8")
    _, unmatched, _ = build(*args)
    assert len(unmatched) == 1
    path.write_text(json.dumps(dict(event, owner_id="R1")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="event log hash"):
        build(*args)

    event.update(owner_id="L4", earliest_timestamp_ms=1300, latest_timestamp_ms=1500,
                 needs_review=False)
    path.write_text(json.dumps(event) + "\n", encoding="utf-8")
    data["official_events_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata.write_text(json.dumps(data), encoding="utf-8")
    changes = {"L4:1500:USE_MEDICINE": dict(kind="pet", actor_id="L2_PET", target_id="L1")}
    args = (*args[:-1], dict(reviews, changes=changes))
    layout = dict(source_asset="clean", players={key: {} for key in ("L1", "L2", "L4")},
                  pets={key: {} for key in ("L2", "L4")}, formation_slots={"L": {
                      "players": {"L1": 5, "L2": 1, "L4": 2}, "pets": {"L2": 2, "L4": 3},
                      "verified_pets": ["L2"],
                  }})
    samples, _, _ = build(*args, vision_layout=layout)
    sample, = samples
    assert sample["formation_slots"] == {"actor": 2, "owner": 2, "target": 5}
    with pytest.raises(ValueError, match="paired clean"):
        build(*args, vision_layout=dict(layout, source_asset="other"))
    with pytest.raises(ValueError, match="screen-slot owner"):
        build(*args)
    changes["L4:1500:USE_MEDICINE"]["actor_id"] = "L4_PET"
    with pytest.raises(ValueError, match="screen-slot owner"):
        build(*args, vision_layout=layout)
