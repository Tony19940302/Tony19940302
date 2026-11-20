from dataclasses import replace
from pathlib import Path

import pytest

from battletracker.battle import BattleEvent, BattleStateEngine, EventType as Kind
from battletracker.formation import BattleLayout, PET_ORDER, load_formations


FORMATIONS = load_formations(
    Path(__file__).resolve().parents[1] / "configs/formations/formations.json"
)


@pytest.fixture
def layout():
    return BattleLayout.build(FORMATIONS["hu_yi"], FORMATIONS["long_fei"])


def event(event_id, kind=Kind.USE_MEDICINE, time=20, **fields):
    return BattleEvent(event_id, "match", 1, time, kind, **fields)


@pytest.fixture
def engine(layout):
    engine = BattleStateEngine("match", layout)
    engine.record(event("start", Kind.BATTLE_START, 0, payload={
        "players": {"A1": {"medicine_used": 0, "medicine_limit": 10,
                           "pet_used": 1, "pet_limit": 6},
                    "B2": {"medicine_used": 0}},
        "initial_pets": {"A1": {"instance_id": "first"}},
    }))
    return engine


@pytest.mark.parametrize("formation", FORMATIONS.values())
def test_formations(formation):
    layout = BattleLayout.build(formation, formation)
    assert len(layout.players) == len(layout.pets) == 10
    for side in ("A", "B"):
        slots = [slot for slot in layout.pets.values() if slot.slot_id.startswith(side)]
        assert tuple(slot.number for slot in slots) == PET_ORDER
        assert [slot.owner_id for slot in slots] == [f"{side}{n}" for n in PET_ORDER]
        assert all(slot.position is None for slot in slots)


def test_calibration_views_are_independent():
    formation = FORMATIONS["long_fei"]
    layout = BattleLayout.build(formation, formation, {
        "calibration_version": "test",
        "teams": {"B": {"formation_id": "long_fei", "pets": {"1": [0.2, 0.8]}}},
    })
    assert layout.pets["B_PET_1"].position == (0.2, 0.8)
    assert layout.pets["A_PET_1"].position is None
    with pytest.raises(ValueError):
        BattleLayout.build(formation, formation, {
            "teams": {"A": {"formation_id": "long_fei", "players": {"1": [1.1, 0]}}}
        })


def test_use_and_target_at_same_time(engine):
    engine.record_many([
        event("person", actor_id="A1", target_id="B2"),
        event("pet", actor_id="A_PET_1", target_id="A2"),
        event("other", actor_id="B2", target_id="A1"),
    ])
    players = engine.state.players
    assert (players["A1"].medicine_used, players["B2"].medicine_used) == (2, 1)
    assert players["A2"].observed_medicine_delta == 0
    assert players["A1"].medicine_remaining == 8


def test_initial_pet_and_replacement(engine):
    assert engine.state.players["A1"].pet_used == 1
    engine.record(event("dead", Kind.PET_DEPARTURE, 20, actor_id="first",
                        payload={"leaves_slot": False}))
    engine.record(event("revive", Kind.PET_REVIVAL, 30, actor_id="first"))
    assert engine.state.players["A1"].pet_used == 1
    assert engine.state.pets["first"].alive
    engine.record(event("new", Kind.SUMMON_PET, 40, actor_id="A1"))
    engine.record(event("depart", Kind.PET_DEPARTURE, 50, actor_id="new"))
    state = engine.state
    assert state.players["A1"].pet_used == 2
    assert state.players["A1"].current_pet_id is None
    assert state.pet_slots["A_PET_1"].owner_id == "A1"
    with pytest.raises(ValueError):
        engine.record(event("revive_old", Kind.PET_REVIVAL, 60, actor_id="first"))


def test_caps_can_change(engine):
    engine.record_many([
        event("pet_cap", Kind.PET_LIMIT_CHANGE, actor_id="A1", payload={"limit": 7}),
        event("medicine_cap", Kind.MEDICINE_LIMIT_CHANGE, actor_id="A1", payload={"limit": 11}),
    ])
    player = engine.state.players["A1"]
    assert (player.pet_remaining, player.medicine_remaining) == (6, 11)
    assert player.resource_sources["pet_limit"] == "pet_cap"


def test_unknown_initial_consumption(layout):
    engine = BattleStateEngine("match", layout)
    engine.record_many([
        event("medicine", actor_id="B2"),
        event("summon", Kind.SUMMON_PET, actor_id="B2"),
        event("limit", Kind.MEDICINE_LIMIT_CHANGE, actor_id="B2", payload={"limit": 8}),
    ])
    player = engine.state.players["B2"]
    assert player.medicine_used is player.pet_used is player.medicine_remaining is None
    assert player.observed_medicine_delta == player.observed_pet_delta == 1


def test_duplicate_and_candidate(engine):
    use = event("use", actor_id="A1")
    engine.record_many([use, use, event("candidate", actor_id="A1", status="candidate")])
    assert engine.state.players["A1"].medicine_used == 1
    with pytest.raises(ValueError):
        engine.record(replace(use, actor_id="B2"))
    assert len(engine.state.event_history) == 3


def test_snapshot_same_tick_pet_link_and_later_retraction(engine):
    engine.record_many([
        event("use", actor_id="first", status="candidate"),
        event("snapshot", Kind.OFFICIAL_SNAPSHOT, payload={
            "players": {"A1": {"medicine_used": 1}}, "covers_event_ids": ["use"],
        }),
    ])
    engine.record(event("confirm", Kind.MANUAL_CORRECTION, 30, revision_of="use",
                        payload={"changes": {"status": "confirmed"}}))
    assert engine.state.players["A1"].medicine_used == 1
    assert not any("snapshot_link" in flag for flag in engine.state.uncertainty_flags)
    engine.record(event("retract", Kind.EVENT_RETRACTED, 40, revision_of="use"))
    assert engine.state.players["A1"].medicine_used == 1
    assert "snapshot:use:unresolved_snapshot_link" in engine.state.uncertainty_flags


def test_correction_retraction_and_persistence(engine, layout, tmp_path):
    engine.record(event("use", actor_id="A1"))
    engine.record(event("correct", Kind.MANUAL_CORRECTION, 30, revision_of="use",
                        payload={"changes": {"actor_id": "B2"}}))
    assert engine.state.players["A1"].medicine_used == 0
    assert engine.state.players["B2"].medicine_used == 1
    engine.record(event("retract", Kind.EVENT_RETRACTED, 40, revision_of="use"))
    assert engine.state.players["B2"].medicine_used == 0
    journal = tmp_path / "match.jsonl"
    engine.save(journal)
    assert BattleStateEngine.load(journal, "match", layout).state == engine.state


def test_invalid_resource_is_atomic_and_over_limit_is_visible(engine):
    before = engine.state
    with pytest.raises(ValueError):
        engine.record_many([event("use", actor_id="A1"),
                            event("bad", Kind.MEDICINE_LIMIT_CHANGE, actor_id="A1",
                                  payload={"limit": -2})])
    assert engine.state == before
    engine.record_many([
        event("use", actor_id="A1"),
        event("limit", Kind.MEDICINE_LIMIT_CHANGE, actor_id="A1", payload={"limit": 0}),
    ])
    assert engine.state.players["A1"].medicine_remaining is None
    assert "A1:medicine:over_limit" in engine.state.uncertainty_flags


def test_inconsistent_ownership_preserves_state(engine):
    with pytest.raises(ValueError):
        engine.record(event("invalid", actor_id="A_PET_1", owner_id="B2"))
    assert engine.state.players["A1"].medicine_used == 0


def test_linked_event_cannot_move_past_snapshot_without_unlinking(engine):
    engine.record_many([
        event("use", actor_id="A1"),
        event("snapshot", Kind.OFFICIAL_SNAPSHOT, 30, payload={
            "players": {"A1": {"medicine_used": 1}}, "covers_event_ids": ["use"],
        }),
    ])
    correction = event("move", Kind.MANUAL_CORRECTION, 45, revision_of="use",
                       payload={"changes": {"timestamp_ms": 40}})
    with pytest.raises(ValueError, match="occurs after"):
        engine.record(correction)
    assert engine.state.players["A1"].medicine_used == 1
    engine.record_many([
        correction,
        event("unlink", Kind.MANUAL_CORRECTION, 45, revision_of="snapshot",
              payload={"changes": {"payload": {"players": {"A1": {"medicine_used": 0}}}}}),
    ])
    assert engine.state.players["A1"].medicine_used == 1


def test_snapshot_payload_normalizes_for_duplicate_after_loading(engine, layout, tmp_path):
    engine.record(event("use", actor_id="A1"))
    snapshot = event("snapshot", Kind.OFFICIAL_SNAPSHOT, payload={
        "players": {"A1": {"medicine_used": 1}}, "covers_event_ids": ("use",),
    })
    engine.record(snapshot)
    path = tmp_path / "match.jsonl"
    engine.save(path)
    loaded = BattleStateEngine.load(path, "match", layout)
    assert loaded.record(snapshot) == engine.state


def test_second_battle_start_cannot_reset_consumption(engine):
    engine.record(event("use", actor_id="A1"))
    with pytest.raises(ValueError, match="first effective event"):
        engine.record(event("restart", Kind.BATTLE_START, 30, payload={
            "players": {"A1": {"medicine_used": 0}},
        }))
    assert engine.state.players["A1"].medicine_used == 1


def test_replaced_instance_cannot_use_medicine(engine):
    engine.record(event("new", Kind.SUMMON_PET, 20, actor_id="A1"))
    with pytest.raises(ValueError, match="active living pet"):
        engine.record(event("stale", time=30, actor_id="first"))
    assert engine.state.players["A1"].medicine_used == 0


def test_known_dead_pet_cannot_use_medicine_by_slot(engine):
    engine.record(event("dead-slot", Kind.PET_DEPARTURE, 20, actor_id="first",
                        payload={"leaves_slot": False}))
    with pytest.raises(ValueError, match="active living pet"):
        engine.record(event("dead-use", time=30, actor_id="A_PET_1"))
    assert engine.state.players["A1"].medicine_used == 0
    engine.record(event("unknown-occupancy", time=30, actor_id="B_PET_2"))
    assert engine.state.players["B2"].medicine_used == 1


def test_departed_pet_slot_rejects_use_until_a_new_pet_arrives(engine, layout, tmp_path):
    engine.record(event("depart", Kind.PET_DEPARTURE, 20, actor_id="first"))
    journal = tmp_path / "departed.jsonl"
    engine.save(journal)
    loaded = BattleStateEngine.load(journal, "match", layout)
    before = loaded.state
    with pytest.raises(ValueError, match="empty pet slot"):
        loaded.record(event("empty-use", time=30, actor_id="A_PET_1"))
    assert loaded.state == before
    loaded.record(event("next", Kind.SUMMON_PET, 30, actor_id="A1"))
    loaded.record(event("valid-use", time=40, actor_id="A_PET_1"))
    assert loaded.state.players["A1"].medicine_used == 1


def test_correcting_pet_arrival_to_child_replays_separate_ledger(engine, layout, tmp_path):
    original = event("arrival", Kind.SUMMON_PET, 20, actor_id="A1", target_id="A_PET_1")
    engine.record(original)
    assert engine.state.players["A1"].pet_used == 2
    assert engine.state.players["A1"].child_remaining is None

    engine.record(event("identify-child", Kind.MANUAL_CORRECTION, 30, revision_of="arrival", payload={
        "changes": {"event_type": "SUMMON_CHILD", "payload": {"child_name": "小星星", "slot_id": "A_PET_1"}},
    }))
    state = engine.state
    player = state.players["A1"]
    assert (player.child_used, player.child_limit, player.child_remaining) == (1, 1, 0)
    assert player.resource_sources["child_used"] == "arrival"
    assert (player.pet_used, player.pet_remaining, player.current_pet_id) == (1, 5, "first")
    assert set(state.pets) == {"first"}
    assert state.pets["first"].departure_round is None
    child = state.children["arrival"]
    assert (child.owner_id, child.slot_id, child.summon_round, child.child_name) == ("A1", "A_PET_1", 1, "小星星")
    assert state.event_history[-2] == original

    journal = tmp_path / "child-arrival.jsonl"
    engine.save(journal)
    loaded = BattleStateEngine.load(journal, "match", layout)
    assert loaded.state == state
    loaded.record(event("retract", Kind.EVENT_RETRACTED, 40, revision_of="arrival"))
    assert loaded.state.players["A1"].child_used is None
    assert not loaded.state.children


def test_child_confirmation_and_covering_snapshot_count_once(engine):
    arrival = event("child", Kind.SUMMON_CHILD, 20, actor_id="A1", status="candidate")
    engine.record(arrival)
    assert engine.state.players["A1"].child_used is None
    assert not engine.state.children

    snapshot = event("snapshot", Kind.OFFICIAL_SNAPSHOT, 20, payload={
        "players": {"A1": {"child_used": 1, "pet_used": 1}, "B2": {"child_used": 0}},
        "covers_event_ids": ["child"],
    })
    engine.record(snapshot)
    assert engine.state.players["B2"].child_remaining == 1
    assert not engine.state.children
    assert "snapshot:child:unresolved_snapshot_link" in engine.state.uncertainty_flags
    engine.record(event("confirm", Kind.MANUAL_CORRECTION, 30, revision_of="child",
                        payload={"changes": {"status": "confirmed"}}))
    state = engine.record(snapshot)
    assert (state.players["A1"].child_used, state.players["A1"].pet_used) == (1, 1)
    assert not any("snapshot_link" in flag or "snapshot_resource_mismatch" in flag for flag in state.uncertainty_flags)
    assert state.children["child"].slot_id is None
    engine.record(event("retract", Kind.EVENT_RETRACTED, 40, revision_of="child"))
    assert engine.state.players["A1"].child_used == 1
    assert not engine.state.children


def test_child_limit_and_owner_slot_are_enforced_atomically(engine):
    illegal_arrivals = [
        event("as-pet", Kind.SUMMON_CHILD, actor_id="first"),
        event("other-owner", Kind.SUMMON_CHILD, actor_id="A1", owner_id="B2"),
        event("other-slot", Kind.SUMMON_CHILD, actor_id="A1", payload={"slot_id": "B_PET_2"}),
        event("as-owner-target", Kind.SUMMON_CHILD, actor_id="A1", target_id="A1"),
        event("pet-id", Kind.SUMMON_CHILD, actor_id="A1", payload={"instance_id": "first"}),
    ]
    for arrival in illegal_arrivals:
        before = engine.state
        with pytest.raises(ValueError):
            engine.record(arrival)
        assert engine.state == before

    child = event("child", Kind.SUMMON_CHILD, actor_id="A1")
    engine.record_many([child, child])
    before = engine.state
    with pytest.raises(ValueError, match="child summon limit"):
        engine.record(event("next-child", Kind.SUMMON_CHILD, 30, actor_id="A1"))
    with pytest.raises(ValueError, match="reserved"):
        engine.record(event("clashing-pet", Kind.SUMMON_PET, 30, actor_id="A1", payload={"instance_id": "child"}))
    assert engine.state == before
    engine.record(event("another-owner-child", Kind.SUMMON_CHILD, 30, owner_id="B2"))
    assert engine.state.players["B2"].child_used == 1


def test_snapshot_cannot_overwrite_a_known_child_charge_with_an_invalid_value(engine):
    engine.record(event("snapshot", Kind.OFFICIAL_SNAPSHOT, 20, payload={"players": {"A1": {"child_used": 1}}}))
    engine.record(event("unknown", Kind.OFFICIAL_SNAPSHOT, 30, payload={"players": {"A1": {"child_used": None}}}))
    assert engine.state.players["A1"].child_used == 1
    with pytest.raises(ValueError, match="child summon limit"):
        engine.record(event("late-child", Kind.SUMMON_CHILD, 40, actor_id="A1"))
    for used in (0, 2, True):
        with pytest.raises(ValueError):
            engine.record(event("invalid", Kind.OFFICIAL_SNAPSHOT, 40,
                                payload={"players": {"A1": {"child_used": used}}}))
    assert engine.state.players["A1"].child_used == 1
