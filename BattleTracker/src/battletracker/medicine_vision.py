"""Observe timed, local evidence for medicine candidates in a clean battle replay."""
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from statistics import median

import cv2
import numpy as np


@dataclass(frozen=True)
class FloatingNumber:
    time_ms: int
    x: float
    y: float
    area: int


@dataclass
class NumberTrack:
    samples: list[FloatingNumber] = field(default_factory=list)


@dataclass(frozen=True)
class MedicineFrame:
    time_ms: int
    activity: dict
    health: dict
    amber: dict


def floating_numbers(hsv, time_ms, players):
    mask = cv2.inRange(hsv, (42, 55, 100), (83, 255, 255))
    grouped = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 5), np.uint8))
    _, _, components, _ = cv2.connectedComponentsWithStats(grouped)
    numbers = []
    for x, y, width, height, area in components[1:]:
        if not (11 <= width <= 46 and 4 <= height <= 13 and area >= 95):
            continue
        cx, cy = x + width / 2, y + height / 2
        if any(abs(cx - px) <= 25 and -36 <= cy - py <= 42 for px, py in players.values()):
            numbers.append(FloatingNumber(time_ms, cx, cy, int(area)))
    return numbers


def health_width(red, center):
    """Length of the strongest horizontal red segment near a calibrated HP bar."""
    x, y = center
    height, width = red.shape
    tile = red[max(0, y - 3):min(height, y + 4), max(0, x - 17):min(width, x + 17)]
    if not tile.size:
        return 0
    bands = np.pad(tile > 0, ((0, 0), (1, 1)))
    best = 0
    for band in bands:
        boundaries = np.flatnonzero(band[1:] != band[:-1])
        if boundaries.size:
            best = max(best, int(np.max(boundaries[1::2] - boundaries[::2])))
    return best


class MedicineObserver:
    """Combine a rising number, a target-side change, and a preceding caster action."""

    def __init__(self, config):
        self.players = config["players"]
        self.health_points = {owner: point for owner, point in self.players.items()
                              if not (config.get("participant_view") is True and owner.startswith("L"))}
        self.actor_owners = {actor: actor for actor in self.players}
        self.actor_owners.update({f"{owner}_PET": owner for owner in config["pets"]})
        self.unverified_actors = {f"{owner}_PET" for owner in config.get("unverified_pet_owners", [])}
        self.allow_weak_amber = config.get("medicine_allow_weak_amber") is True
        self.active = []
        self.completed = []
        self.frames = []
        self.times = []

    def observe(self, time, image, hsv, activity):
        red = cv2.bitwise_or(cv2.inRange(hsv, (0, 139, 115), (7, 255, 255)),
                             cv2.inRange(hsv, (173, 139, 115), (179, 255, 255)))
        amber = cv2.inRange(hsv, (3, 126, 212), (31, 255, 255))
        health = {owner: health_width(red, point) for owner, point in self.health_points.items()}
        impact = {owner: self._amber_coverage(amber, point) for owner, point in self.players.items()}
        self.times.append(time)
        self.frames.append(MedicineFrame(time, activity, health, impact))
        self._track_numbers(time, floating_numbers(hsv, time, self.players))

    @staticmethod
    def _amber_coverage(mask, center):
        x, y = center
        height, width = mask.shape
        tile = mask[max(0, y + 7):min(height, y + 55), max(0, x - 19):min(width, x + 19)]
        return np.count_nonzero(tile) / tile.size if tile.size else 0.0

    def _track_numbers(self, time, numbers):
        active = []
        for track in self.active:
            if time - track.samples[-1].time_ms > 250:
                self.completed.append(track)
            else:
                active.append(track)
        self.active = active
        options = []
        for index, track in enumerate(active):
            last = track.samples[-1]
            if time - track.samples[0].time_ms > 1150:
                continue
            for hit, number in enumerate(numbers):
                dx, dy = abs(number.x - last.x), number.y - last.y
                if dx <= 6 and -10 <= dy <= 3:
                    options.append((2 * dx + abs(dy), index, hit))
        used_tracks, used_hits = set(), set()
        for _, index, hit in sorted(options):
            if index not in used_tracks and hit not in used_hits:
                active[index].samples.append(numbers[hit])
                used_tracks.add(index)
                used_hits.add(hit)
        active.extend(NumberTrack([number]) for hit, number in enumerate(numbers) if hit not in used_hits)

    def _window(self, start, end):
        return self.frames[bisect_left(self.times, start):bisect_right(self.times, end)]

    def _rank_targets(self, first):
        before = self._window(first.time_ms - 520, first.time_ms - 260)
        after = self._window(first.time_ms - 220, first.time_ms + 220)
        settled = self._window(first.time_ms, first.time_ms + 310)
        if not before or not after or not settled:
            return []
        ranking = []
        for owner, (x, y) in self.players.items():
            dx, dy = abs(first.x - x), first.y - y
            if dx > 25 or not -36 <= dy <= 42:
                continue
            baseline = median(frame.activity.get(owner, 0) for frame in before)
            impact = max(frame.activity.get(owner, 0) for frame in after) - baseline
            amber_baseline = median(frame.amber[owner] for frame in before)
            amber_delta = max(frame.amber[owner] for frame in after) - amber_baseline
            amber_onset = next((frame.time_ms for frame in self._window(
                first.time_ms - 370, first.time_ms + 120)
                if frame.amber[owner] >= amber_baseline + 0.045), first.time_ms)
            health_gain = None
            if owner in self.health_points:
                health_gain = median(frame.health[owner] for frame in settled) - median(
                    frame.health[owner] for frame in before)
            geometry = 0.12 - dx / 140 - abs(dy + 7) / 520
            health_score = min(0.23, max(0, health_gain) * 0.026) if health_gain is not None else 0.0
            score = (geometry + health_score
                     + min(0.14, max(0, amber_delta) * 0.70)
                     + min(0.18, max(0, impact) * 0.64))
            ranking.append((score, owner, impact, amber_delta, health_gain, amber_onset))
        return sorted(ranking, reverse=True)

    def _rank_actors(self, when, target, amber_onset):
        prior = self._window(when - 1320, when - 800)
        action = self._window(when - 760, min(when - 120, amber_onset - 40))
        if len(action) < 3:
            return []
        allied = [actor for actor, owner in self.actor_owners.items() if owner[0] == target[0]]
        ranking = []
        for actor in allied:
            idle = median(frame.activity.get(actor, 0) for frame in prior) if prior else 0
            contrasts = []
            for frame in action:
                other = max((frame.activity.get(candidate, 0) for candidate in allied
                             if candidate not in {actor, target}), default=0)
                contrasts.append(frame.activity.get(actor, 0) - 0.45 * other)
            peak, second, third = sorted(contrasts, reverse=True)[:3]
            score = 0.50 * peak + 0.30 * second + 0.20 * third - 0.25 * min(idle, 0.28)
            if actor == target:
                score -= 0.07  # Only a distinct earlier animation supports self-use.
            ranking.append((score, actor))
        return sorted(ranking, reverse=True)

    def _persistent_hp_return(self, when, target):
        before = self._window(when - 520, when - 260)
        settled = self._window(when, when + 310)
        return (len(before) >= 3 and len(settled) >= 3
                and all(frame.health[target] <= 1 for frame in before)
                and all(frame.health[target] >= 2 for frame in settled))

    def classify(self, skills, *, summons=()):
        events = []
        for track in (*self.completed, *self.active):
            event = self._classify_track(track, skills, summons)
            if event is not None:
                events.append(event)
        events.sort(key=lambda row: (row["time_ms"], row["target_id"] or ""))
        return self._resolve_nearby_tracks(events)

    def _classify_track(self, track, skills, summons):
        first = track.samples[0]
        decision = bisect_left(self.times, first.time_ms + 380)
        if decision == len(self.times):
            return None
        samples = [item for item in track.samples if item.time_ms <= self.times[decision]]
        if len(samples) < 3 or first.y - min(item.y for item in samples[1:]) < 2.5:
            return None
        targets = self._rank_targets(first)
        if not targets:
            return None
        score, target, impact, amber, health, amber_onset = targets[0]
        target_gap = score - targets[1][0] if len(targets) > 1 else score
        actors = self._rank_actors(first.time_ms, target, amber_onset)
        top, actor = actors[0] if actors else (0, None)
        actor_gap = top - actors[1][0] if len(actors) > 1 else top
        source_strong = top >= 0.14 and actor_gap >= 0.053
        health_increase = health is not None and health >= 1
        receiver_strong = target_gap >= 0.034 and amber >= 0.075 and (health_increase or impact >= 0.07)
        conflict = [name for performer, name, cue in skills
                    if performer == actor and actor is not None
                    and cue.start <= first.time_ms + 30 and cue.end >= first.time_ms - 480]
        summoning = any(
            row["event_type"] == "SUMMON_PET" and row.get("actor_id") == actor
            # A summon can be dated at ring onset while its animation still overlaps the heal.
            and first.time_ms - 540 <= row.get("end_ms", row["time_ms"]) <= first.time_ms + 60
            and row.get("decision_ready_ms") is not None
            and row["decision_ready_ms"] <= first.time_ms + 660
            for row in summons
        )
        source_eligible = (source_strong and not conflict and not summoning
                           and actor not in self.unverified_actors)
        weak_amber_recovery = (
            self.allow_weak_amber and not receiver_strong and source_eligible
            and actor.endswith("_PET") and top >= 0.28 and actor_gap >= 0.11
            and target_gap >= 0.034 and health is not None and health >= 2 and impact >= 0.25
            and self._persistent_hp_return(first.time_ms, target)
        )
        receiver_strong = receiver_strong or weak_amber_recovery
        evidence = dict(
            medicine_number_rise_px=round(first.y - min(item.y for item in samples), 2),
            heal_area=max(item.area for item in samples),
            actor_motion_candidates=[[key, round(value, 3)] for value, key in actors[:3]],
            target_motion_candidates=[[row[1], round(row[0], 3)] for row in targets[:3]],
            receiver_health_gain_px=health, receiver_motion=round(impact, 3),
            receiver_amber_growth=round(amber, 3), nearby_skill_templates=conflict,
            overlapping_summon=summoning, unverified_source_mapping=actor in self.unverified_actors,
        )
        if weak_amber_recovery:
            evidence["persistent_hp_return_during_weak_amber"] = True
        accepted = source_eligible and receiver_strong
        event_type = "USE_MEDICINE" if accepted else ("NON_MEDICINE" if conflict else "UNKNOWN")
        return dict(event_type=event_type, actor_id=actor if accepted else None,
                    owner_id=self.actor_owners[actor] if accepted else None,
                    target_id=target if receiver_strong else None,
                    time_ms=first.time_ms, start_ms=first.time_ms, end_ms=samples[-1].time_ms,
                    status="candidate", decision_ready_ms=self.times[decision], evidence=evidence)

    def _resolve_nearby_tracks(self, events):
        """One caster and one action must not turn nearby splashes into multiple uses."""
        groups = {}
        resolved = []
        for event in events:
            actor = event["actor_id"]
            if actor is None:
                resolved.append(event)
                continue
            runs = groups.setdefault(actor, [])
            if not runs or event["time_ms"] - runs[-1][0]["time_ms"] > 280:
                runs.append([])
            runs[-1].append(event)
        for runs in groups.values():
            for group in runs:
                first = group[0]
                decision = bisect_left(self.times, first["time_ms"] + 380 + 280)
                if decision == len(self.times):
                    first.update(event_type="UNKNOWN", actor_id=None, owner_id=None)
                    first["evidence"]["recording_ended_during_disambiguation"] = True
                else:
                    first["decision_ready_ms"] = self.times[decision]
                if len({event["target_id"] for event in group}) > 1:
                    first.update(event_type="UNKNOWN", actor_id=None, owner_id=None)
                    first["evidence"]["overlapping_receivers"] = True
                    first["evidence"]["receiver_observations"] = [
                        {"target_id": event["target_id"], "time_ms": event["time_ms"]}
                        for event in group
                    ]
                resolved.append(first)
        return sorted(resolved, key=lambda row: (row["time_ms"], row["target_id"] or ""))
