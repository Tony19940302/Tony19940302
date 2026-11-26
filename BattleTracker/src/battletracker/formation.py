from dataclasses import dataclass
from enum import StrEnum
import json
import math
from pathlib import Path

PET_ORDER = (5, 3, 1, 2, 4)
Point = tuple[float, float]


def vision_slot_numbers(config):
    """Translate calibrated image IDs to formation numbers without moving their anchors."""
    numbers = {}
    for side, slots in config.get("formation_slots", {}).items():
        if side not in ("L", "R"):
            raise ValueError("unknown screen side in formation slot map")
        for kind, suffix in (("players", ""), ("pets", "_PET")):
            table = slots[kind]
            if (any(key not in config.get(kind, {}) or not key.startswith(side) for key in table)
                    or any(type(value) is not int or value not in PET_ORDER for value in table.values())
                    or len(set(table.values())) != len(table)):
                raise ValueError(f"invalid {kind} formation slots for side {side}")
            numbers.update((f"{key}{suffix}", number) for key, number in table.items())
    return numbers


def vision_pet_owners(config):
    """Map fixed pet geometry keys to player geometry IDs, with unresolved owners as None."""
    sides = config.get("formation_slots", {})
    numbers = vision_slot_numbers(config)
    players = {(player[0], number): player
               for player in config.get("players", {}) if (number := numbers.get(player)) is not None}
    for side, table in sides.items():
        if set(table["verified_pets"]) - table["pets"].keys():
            raise ValueError(f"verified pet is missing a formation number for side {side}")
    legacy_unverified = set(config.get("unverified_pet_owners", ()))
    owners = {}
    for slot in config.get("pets", {}):
        side = slot[0]
        if side in sides:
            if slot not in sides[side]["verified_pets"]:
                owners[slot] = None
                continue
            owner = players.get((side, numbers[f"{slot}_PET"]))
            if owner is None:
                raise ValueError(f"verified pet {slot} has no player with the same formation number")
            owners[slot] = owner
        else:
            owners[slot] = None if slot in legacy_unverified else slot
    return owners


class TeamSide(StrEnum):
    A = "A"
    B = "B"


@dataclass(frozen=True)
class PlayerSlot:
    player_id: str
    number: int
    position: Point | None = None


@dataclass(frozen=True)
class PetSlot:
    slot_id: str
    owner_id: str
    number: int
    position: Point | None = None


@dataclass(frozen=True)
class Formation:
    formation_id: str
    formation_name: str


def load_formations(path: Path) -> dict[str, Formation]:
    formations = {}
    for row in json.loads(path.read_text(encoding="utf-8")):
        formation = Formation(**row)
        if not formation.formation_id or not formation.formation_name:
            raise ValueError("formation ID and name are required")
        if formation.formation_id in formations:
            raise ValueError(f"duplicate formation ID: {formation.formation_id}")
        formations[formation.formation_id] = formation
    return formations


def _position(positions: dict, number: int) -> Point | None:
    point = positions.get(str(number))
    if point is None:
        return None
    if not isinstance(point, list) or len(point) != 2 or any(
        type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
        for value in point
    ):
        raise ValueError(f"invalid normalized position for slot {number}")
    return tuple(point)


@dataclass(frozen=True)
class BattleLayout:
    formations: dict[TeamSide, Formation]
    players: dict[str, PlayerSlot]
    pets: dict[str, PetSlot]
    calibration_version: str | None = None

    @classmethod
    def build(cls, ally: Formation, enemy: Formation, calibration: dict | None = None):
        calibration = {} if calibration is None else calibration
        if calibration.get("coordinate_system", "normalized_screen") != "normalized_screen":
            raise ValueError("unsupported coordinate system")
        teams = calibration.get("teams", {})
        if set(teams) - {"A", "B"}:
            raise ValueError("unknown calibration side")
        formations = {TeamSide.A: ally, TeamSide.B: enemy}
        players, pets = {}, {}
        for side, formation in formations.items():
            team = teams.get(side, {})
            if team and team.get("formation_id") != formation.formation_id:
                raise ValueError(f"calibration formation does not match side {side}")
            player_positions = team.get("players", {})
            pet_positions = team.get("pets", {})
            allowed = {"1", "2", "3", "4", "5"}
            if (set(player_positions) | set(pet_positions)) - allowed:
                raise ValueError("unknown calibration slot")
            for number in range(1, 6):
                player_id = f"{side}{number}"
                players[player_id] = PlayerSlot(
                    player_id, number, _position(player_positions, number)
                )
            for number in PET_ORDER:
                slot_id = f"{side}_PET_{number}"
                pets[slot_id] = PetSlot(
                    slot_id, f"{side}{number}", number, _position(pet_positions, number)
                )
        return cls(formations, players, pets, calibration.get("calibration_version"))
