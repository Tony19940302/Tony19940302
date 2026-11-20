from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from .formation import BattleLayout, PetSlot


class EventType(StrEnum):
    BATTLE_START = "BATTLE_START"
    ROUND_START = "ROUND_START"
    USE_MEDICINE = "USE_MEDICINE"
    SUMMON_PET = "SUMMON_PET"
    SUMMON_CHILD = "SUMMON_CHILD"
    PET_DEPARTURE = "PET_DEPARTURE"
    PET_REVIVAL = "PET_REVIVAL"
    PLAYER_DEATH = "PLAYER_DEATH"
    PLAYER_REVIVAL = "PLAYER_REVIVAL"
    MEDICINE_LIMIT_CHANGE = "MEDICINE_LIMIT_CHANGE"
    PET_LIMIT_CHANGE = "PET_LIMIT_CHANGE"
    OFFICIAL_SNAPSHOT = "OFFICIAL_SNAPSHOT"
    MANUAL_CORRECTION = "MANUAL_CORRECTION"
    EVENT_RETRACTED = "EVENT_RETRACTED"
    UNKNOWN_ACTION = "UNKNOWN_ACTION"
    BATTLE_END = "BATTLE_END"


@dataclass(frozen=True)
class BattleEvent:
    event_id: str
    battle_id: str
    round: int
    timestamp_ms: int
    event_type: EventType
    actor_id: str | None = None
    owner_id: str | None = None
    target_id: str | None = None
    confidence: float | None = None
    source: str = "manual"
    status: str = "confirmed"
    start_ms: int | None = None
    end_ms: int | None = None
    evidence: tuple[str, ...] = ()
    algorithm_version: str | None = None
    revision_of: str | None = None
    payload: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict):
        values = dict(data)
        values["event_type"] = EventType(values["event_type"])
        values["evidence"] = tuple(values.get("evidence", ()))
        return cls(**values)


@dataclass
class PlayerState:
    player_id: str
    formation_slot: int
    medicine_used: int | None = None
    medicine_limit: int | None = None
    pet_used: int | None = None
    pet_limit: int | None = None
    current_pet_id: str | None = None
    alive: bool | None = None
    observed_medicine_delta: int = 0
    observed_pet_delta: int = 0
    resource_sources: dict[str, str] = field(default_factory=dict)
    updated_ms: int | None = None
    child_used: int | None = None

    @property
    def medicine_remaining(self) -> int | None:
        return _remaining(self.medicine_limit, self.medicine_used)

    @property
    def pet_remaining(self) -> int | None:
        return _remaining(self.pet_limit, self.pet_used)

    @property
    def child_limit(self) -> int:
        return CHILD_LIMIT

    @property
    def child_remaining(self) -> int | None:
        return _remaining(self.child_limit, self.child_used)


def _remaining(limit: int | None, used: int | None) -> int | None:
    if limit is None or used is None or used > limit:
        return None
    return limit - used


@dataclass
class PetState:
    pet_instance_id: str
    owner_id: str
    slot_id: str
    summon_round: int
    pet_name: str | None = None
    alive: bool | None = None
    departure_round: int | None = None


@dataclass(frozen=True)
class ChildState:
    child_instance_id: str
    owner_id: str
    summon_round: int
    child_name: str | None = None
    slot_id: str | None = None


@dataclass
class BattleState:
    battle_id: str
    battle_mode: str | None
    players: dict[str, PlayerState]
    pet_slots: dict[str, PetSlot]
    formations: dict[str, str]
    pets: dict[str, PetState] = field(default_factory=dict)
    current_round: int = 0
    current_timestamp: int = 0
    ended: bool = False
    event_history: tuple[BattleEvent, ...] = ()
    uncertainty_flags: tuple[str, ...] = ()
    children: dict[str, ChildState] = field(default_factory=dict)


CHILD_LIMIT = 1
RESOURCE_FIELDS = {"medicine_used", "medicine_limit", "pet_used", "pet_limit", "child_used"}
REVISIONS = {EventType.MANUAL_CORRECTION, EventType.EVENT_RETRACTED}
SNAPSHOTS = {EventType.BATTLE_START, EventType.OFFICIAL_SNAPSHOT}
RESOURCE_EVENT_FIELDS = {
    EventType.USE_MEDICINE: "medicine_used",
    EventType.SUMMON_PET: "pet_used",
    EventType.SUMMON_CHILD: "child_used",
    EventType.MEDICINE_LIMIT_CHANGE: "medicine_limit",
    EventType.PET_LIMIT_CHANGE: "pet_limit",
}
EVENT_ORDER = {
    EventType.BATTLE_START: -1,
    EventType.OFFICIAL_SNAPSHOT: 1,
    EventType.BATTLE_END: 2,
}


def _natural(value) -> bool:
    return type(value) is int and value >= 0


class BattleStateEngine:
    """Fold an append-only journal into a battle state."""

    def __init__(self, battle_id: str, layout: BattleLayout, battle_mode: str | None = None,
                 events=()):
        if not battle_id:
            raise ValueError("battle ID is required")
        self.battle_id = battle_id
        self._layout = deepcopy(layout)
        self.battle_mode = battle_mode
        self._events = self._unique(deepcopy(tuple(events)))
        self._state = self._replay(self._events)

    @property
    def state(self) -> BattleState:
        return deepcopy(self._state)

    def record(self, event: BattleEvent) -> BattleState:
        self.record_many((event,))
        return self.state

    def record_many(self, events) -> None:
        combined = self._unique(self._events + deepcopy(tuple(events)))
        state = self._replay(combined)
        self._events, self._state = combined, state

    @staticmethod
    def _unique(events) -> tuple[BattleEvent, ...]:
        by_id = {}
        for event in events:
            try:
                event = BattleEvent.from_dict(json.loads(json.dumps(asdict(event), allow_nan=False)))
            except (TypeError, ValueError) as error:
                raise ValueError("events must have JSON-compatible fields") from error
            if event.event_id in by_id and by_id[event.event_id] != event:
                raise ValueError(f"conflicting event ID: {event.event_id}")
            by_id[event.event_id] = event
        return tuple(by_id.values())

    def _validate(self, event: BattleEvent) -> None:
        if event.battle_id != self.battle_id or not event.event_id:
            raise ValueError("invalid battle or event ID")
        if event.status not in {"candidate", "confirmed"} or not event.source:
            raise ValueError("invalid event status or source")
        if not _natural(event.round) or not _natural(event.timestamp_ms):
            raise ValueError("invalid round or timestamp")
        if event.confidence is not None and (
            type(event.confidence) not in (int, float) or not math.isfinite(event.confidence)
            or not 0 <= event.confidence <= 1
        ):
            raise ValueError("invalid confidence")
        if event.start_ms is not None and (
            not _natural(event.start_ms) or event.start_ms > event.timestamp_ms
        ):
            raise ValueError("invalid start timestamp")
        if event.end_ms is not None and (
            not _natural(event.end_ms) or event.end_ms < event.timestamp_ms
        ):
            raise ValueError("invalid end timestamp")
        if not isinstance(event.event_type, EventType) or not isinstance(event.payload, dict):
            raise ValueError("invalid event type or payload")

    def _effective(self, events) -> list[BattleEvent]:
        original = {}
        effective = {}
        for event in events:
            self._validate(event)
            if event.event_type not in REVISIONS:
                original[event.event_id] = effective[event.event_id] = event
                continue
            base = original.get(event.revision_of)
            if base is None or event.status != "confirmed":
                raise ValueError("revision requires an earlier original event and confirmation")
            if event.event_type == EventType.EVENT_RETRACTED:
                effective[event.revision_of] = None
                continue
            changes = event.payload.get("changes", {})
            allowed = {
                "event_type", "actor_id", "owner_id", "target_id", "round", "timestamp_ms",
                "start_ms", "end_ms", "payload", "status", "confidence", "evidence",
                "algorithm_version",
            }
            if not isinstance(changes, dict) or set(changes) - allowed:
                raise ValueError("invalid correction fields")
            changes = dict(changes)
            if "event_type" in changes:
                changes["event_type"] = EventType(changes["event_type"])
            if "evidence" in changes:
                changes["evidence"] = tuple(changes["evidence"])
            current = effective[event.revision_of]
            revised = replace(current or base, **changes, source=event.source)
            self._validate(revised)
            if revised.event_type in REVISIONS:
                raise ValueError("cannot revise a normal event into a revision")
            effective[event.revision_of] = revised
        return sorted(
            (item for item in effective.values() if item and item.status == "confirmed"),
            key=lambda item: (item.round, item.timestamp_ms, EVENT_ORDER.get(item.event_type, 0)),
        )

    def _replay(self, events) -> BattleState:
        state = BattleState(
            self.battle_id, self.battle_mode,
            {key: PlayerState(key, slot.number) for key, slot in self._layout.players.items()},
            deepcopy(self._layout.pets),
            {side.value: item.formation_id for side, item in self._layout.formations.items()},
            event_history=events,
        )
        actions = self._effective(events)
        indexed = {event.event_id: event for event in events}
        action_order = {event.event_id: index for index, event in enumerate(actions)}
        resource_events = {}
        flags = []
        for index, event in enumerate(actions):
            if index and event.event_type == EventType.BATTLE_START:
                raise ValueError("battle start must be the first effective event")
            if index and event.timestamp_ms < state.current_timestamp:
                raise ValueError("timestamp goes backward across rounds")
            if state.ended:
                raise ValueError("confirmed event after battle end")
            state.current_round = event.round
            state.current_timestamp = event.timestamp_ms
            if event.event_type in SNAPSHOTS:
                flags.extend(self._snapshot_link_flags(event, indexed, resource_events, action_order))
            self._apply(state, event)
            if event.event_type in RESOURCE_EVENT_FIELDS:
                resource_events[event.event_id] = (
                    self._owner(state, event).player_id, RESOURCE_EVENT_FIELDS[event.event_type]
                )
        for player in state.players.values():
            for resource in ("medicine", "pet"):
                used = getattr(player, f"{resource}_used")
                limit = getattr(player, f"{resource}_limit")
                if used is None or limit is None:
                    flags.append(f"{player.player_id}:{resource}:unknown")
                elif used > limit:
                    flags.append(f"{player.player_id}:{resource}:over_limit")
        state.uncertainty_flags = tuple(flags)
        return state

    @staticmethod
    def _snapshot_link_flags(event, indexed, resource_events, action_order):
        links = event.payload.get("covers_event_ids", ())
        if not isinstance(links, (tuple, list)) or any(not isinstance(link, str) for link in links):
            raise ValueError("snapshot links must be event IDs")
        for event_id in links:
            original = indexed.get(event_id)
            if original is None or original.event_type in REVISIONS or event_id == event.event_id:
                raise ValueError("snapshot link must reference an original event")
            if action_order.get(event_id, -1) > action_order[event.event_id]:
                raise ValueError("confirmed event linked to a snapshot occurs after it")
            resource = resource_events.get(event_id)
            if resource is None:
                yield f"{event.event_id}:{event_id}:unresolved_snapshot_link"
                continue
            owner, field_name = resource
            if event.payload.get("players", {}).get(owner, {}).get(field_name) is None:
                yield f"{event.event_id}:{event_id}:snapshot_resource_mismatch"
    @staticmethod
    def _owner(state: BattleState, event: BattleEvent) -> PlayerState:
        actor = event.actor_id
        if actor in state.pet_slots:
            owner = state.pet_slots[actor].owner_id
        elif actor in state.pets:
            owner = state.pets[actor].owner_id
        else:
            owner = actor or event.owner_id
        if owner not in state.players or event.owner_id not in (None, owner):
            raise ValueError("unknown or inconsistent resource owner")
        return state.players[owner]

    @staticmethod
    def _set(player: PlayerState, name: str, value, event: BattleEvent) -> None:
        if value is not None and not _natural(value):
            raise ValueError(f"invalid resource value for {name}")
        if name == "child_used":
            if value is None:
                return
            if value > CHILD_LIMIT or (player.child_used is not None and value < player.child_used):
                raise ValueError("child count is outside its limit or regresses")
        setattr(player, name, value)
        player.resource_sources[name] = event.event_id
        player.updated_ms = event.timestamp_ms

    def _increment(self, player: PlayerState, resource: str, event: BattleEvent) -> None:
        name = f"{resource}_used"
        used = getattr(player, name)
        if used is None:
            delta = f"observed_{resource}_delta"
            setattr(player, delta, getattr(player, delta) + 1)
            player.updated_ms = event.timestamp_ms
        else:
            self._set(player, name, used + 1, event)

    def _snapshot(self, state: BattleState, event: BattleEvent) -> None:
        for player_id, values in event.payload.get("players", {}).items():
            if player_id not in state.players or set(values) - RESOURCE_FIELDS:
                raise ValueError("invalid snapshot player or resource field")
            player = state.players[player_id]
            for name, value in values.items():
                self._set(player, name, value, event)
                if name in {"medicine_used", "pet_used"} and value is not None:
                    setattr(player, f"observed_{name.removesuffix('_used')}_delta", 0)

    def _pet_slot(self, state: BattleState, player: PlayerState) -> PetSlot:
        return state.pet_slots[f"{player.player_id[0]}_PET_{player.formation_slot}"]

    def _new_pet(self, state: BattleState, player: PlayerState, event: BattleEvent,
                 instance_id: str, name: str | None) -> None:
        if not isinstance(instance_id, str) or not instance_id or (
            instance_id in state.pets or instance_id in state.pet_slots or instance_id in state.players
            or instance_id in state.children
        ):
            raise ValueError("duplicate, reserved or empty pet instance ID")
        if player.current_pet_id is not None:
            previous = state.pets[player.current_pet_id]
            previous.departure_round = event.round
        slot = self._pet_slot(state, player)
        state.pets[instance_id] = PetState(
            instance_id, player.player_id, slot.slot_id, event.round, name, True
        )
        player.current_pet_id = instance_id
        player.updated_ms = event.timestamp_ms

    def _summon_child(self, state: BattleState, player: PlayerState, event: BattleEvent) -> None:
        if event.actor_id is not None and event.actor_id != player.player_id:
            raise ValueError("child summon requires a player actor")
        if player.child_used == CHILD_LIMIT:
            raise ValueError("child summon limit reached")
        instance_id = event.payload.get("instance_id", event.event_id)
        if not isinstance(instance_id, str) or not instance_id or (
            instance_id in state.children or instance_id in state.pets
            or instance_id in state.pet_slots or instance_id in state.players
        ):
            raise ValueError("duplicate, reserved or empty child instance ID")
        slot_id = event.payload.get("slot_id")
        if slot_id is not None:
            slot = state.pet_slots.get(slot_id) if isinstance(slot_id, str) else None
            if slot is None or slot.owner_id != player.player_id:
                raise ValueError("child arrival slot does not belong to the player")
        if event.target_id not in (None, instance_id, slot_id):
            raise ValueError("child summon target must be the child instance or specified slot")
        child_name = event.payload.get("child_name")
        if child_name is not None and not isinstance(child_name, str):
            raise ValueError("child name must be text")
        state.children[instance_id] = ChildState(instance_id, player.player_id, event.round, child_name, slot_id)
        # A successful arrival consumes the single charge even if we joined partway through.
        self._set(player, "child_used", CHILD_LIMIT, event)

    def _apply(self, state: BattleState, event: BattleEvent) -> None:
        kind = event.event_type
        if kind in SNAPSHOTS:
            self._snapshot(state, event)
            if kind == EventType.BATTLE_START:
                for player_id, pet in event.payload.get("initial_pets", {}).items():
                    if player_id not in state.players:
                        raise ValueError("unknown initial pet owner")
                    self._new_pet(state, state.players[player_id], event,
                                  pet["instance_id"], pet.get("name"))
            return
        if kind == EventType.BATTLE_END:
            state.ended = True
            return
        if kind in {EventType.ROUND_START, EventType.UNKNOWN_ACTION}:
            return
        player = self._owner(state, event)
        if kind == EventType.USE_MEDICINE:
            instance_id = event.actor_id
            if instance_id in state.pet_slots:
                instance_id = player.current_pet_id
                if instance_id is None and any(
                    pet.owner_id == player.player_id for pet in state.pets.values()
                ):
                    raise ValueError("medicine actor is an empty pet slot")
            pet = state.pets.get(instance_id)
            if pet is not None and (pet.alive is False or player.current_pet_id != pet.pet_instance_id):
                raise ValueError("medicine actor is not the active living pet")
            self._increment(player, "medicine", event)
        elif kind == EventType.SUMMON_PET:
            if event.actor_id and event.actor_id != player.player_id:
                raise ValueError("only a player can summon in this event")
            self._new_pet(state, player, event, event.payload.get("instance_id", event.event_id),
                          event.payload.get("pet_name"))
            self._increment(player, "pet", event)
        elif kind == EventType.SUMMON_CHILD:
            self._summon_child(state, player, event)
        elif kind in {EventType.PET_DEPARTURE, EventType.PET_REVIVAL}:
            instance_id = event.payload.get("instance_id") or player.current_pet_id
            if instance_id is None or instance_id not in state.pets:
                raise ValueError("pet event requires a known instance")
            pet = state.pets[instance_id]
            if pet.owner_id != player.player_id:
                raise ValueError("pet belongs to another player")
            if kind == EventType.PET_DEPARTURE:
                pet.alive = False
                if event.payload.get("leaves_slot", True):
                    pet.departure_round = event.round
                    if player.current_pet_id == instance_id:
                        player.current_pet_id = None
            else:
                if player.current_pet_id != instance_id or pet.alive is not False:
                    raise ValueError("revival requires the current dead pet in its slot")
                pet.alive = True
                pet.departure_round = None
                player.current_pet_id = instance_id
            player.updated_ms = event.timestamp_ms
        elif kind in {EventType.PLAYER_DEATH, EventType.PLAYER_REVIVAL}:
            if event.actor_id and event.actor_id != player.player_id:
                raise ValueError("player life event requires a player")
            player.alive = kind == EventType.PLAYER_REVIVAL
            player.updated_ms = event.timestamp_ms
        else:
            field_name = "medicine_limit" if kind == EventType.MEDICINE_LIMIT_CHANGE else "pet_limit"
            self._set(player, field_name, event.payload["limit"], event)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = None
        try:
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                    delete=False) as stream:
                staged = Path(stream.name)
                for event in self._events:
                    stream.write(json.dumps(asdict(event), ensure_ascii=False, allow_nan=False) + "\n")
            os.replace(staged, path)
        finally:
            if staged is not None:
                staged.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path, battle_id: str, layout: BattleLayout, battle_mode=None):
        with path.open(encoding="utf-8") as stream:
            events = [BattleEvent.from_dict(json.loads(line)) for line in stream if line.strip()]
        return cls(battle_id, layout, battle_mode, events)






