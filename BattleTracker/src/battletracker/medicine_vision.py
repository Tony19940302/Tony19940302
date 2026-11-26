"""Observe timed, local evidence for medicine candidates in a clean battle replay."""
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import median

import cv2
import numpy as np

from .battle import MedicineConsumption
from .formation import vision_pet_owners


MEDICINE_TRIGGER_SKILLS = frozenset({"寸草心"})


def _matches_medicine_trigger(performer, name, cue, actor, recovery_ms):
    return (actor is not None and actor.endswith("_PET") and performer == actor
            and name in MEDICINE_TRIGGER_SKILLS and -80 <= cue.start - recovery_ms <= 160)


def medicine_skill_context(skills, actor, start_ms, end_ms, *, recovery_ms=None):
    conflicts = []
    triggers = []
    if actor is not None:
        for performer, name, cue in skills:
            if performer != actor or cue.end < start_ms:
                continue
            if name in MEDICINE_TRIGGER_SKILLS:
                if recovery_ms is not None and _matches_medicine_trigger(performer, name, cue, actor, recovery_ms):
                    triggers.append(name)
            elif cue.start <= end_ms:
                conflicts.append(name)
    return conflicts, triggers


def mark_medicine_trigger(event, triggers):
    if not triggers:
        if (event["event_type"] != "USE_MEDICINE" and "payload" in event
                and event["evidence"].get("ambiguous_medicine_trigger")):
            event["payload"].pop("medicine_consumption", None)
            if not event["payload"]:
                del event["payload"]
        return event
    hints = sorted(set(triggers))
    if event["event_type"] == "USE_MEDICINE":
        event.setdefault("payload", {}).update(medicine_consumption=MedicineConsumption.UNKNOWN.value,
                                               medicine_trigger_hints=hints)
    else:
        if "payload" in event:
            event["payload"].pop("medicine_consumption", None)
            event["payload"].pop("medicine_trigger_hints", None)
            if not event["payload"]:
                del event["payload"]
        event["evidence"]["medicine_trigger_hints"] = hints
    return event


def attach_medicine_trigger_hints(events, skills):
    """Link passive caption onsets to finalized candidates without confirming consumption."""
    matches = defaultdict(list)
    for performer, name, cue in skills:
        if name not in MEDICINE_TRIGGER_SKILLS:
            continue
        possible = [i for i, event in enumerate(events)
                    if event["event_type"] == "USE_MEDICINE"
                    and event["status"] == "candidate"
                    and _matches_medicine_trigger(performer, name, cue, event["actor_id"], event["time_ms"])]
        for i in possible:
            matches[i].append((name, len(possible)))
    for i, captions in matches.items():
        event = events[i]
        if len(captions) == 1 and captions[0][1] == 1:
            mark_medicine_trigger(event, [captions[0][0]])
        else:
            event.setdefault("payload", {})["medicine_consumption"] = MedicineConsumption.UNKNOWN.value
            event["evidence"]["ambiguous_medicine_trigger"] = sorted({name for name, _ in captions})
    return events


@dataclass(frozen=True)
class FloatingNumber:
    time_ms: int
    x: float
    y: float
    area: int


NUMBER_DECISION_MS = 380


@dataclass
class NumberTrack:
    samples: list[FloatingNumber] = field(default_factory=list)
    pending_overlap: list[FloatingNumber] = field(default_factory=list)
    recovered_from_weak_seed: bool = False


@dataclass(frozen=True)
class MedicineFrame:
    time_ms: int
    activity: dict
    health: dict
    amber: dict
    pet_body_detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CasterMotion:
    actor: str
    start_ms: int
    end_ms: int


def _number_components(hsv, upper_hue):
    mask = cv2.inRange(hsv, (42, 55, 100), (upper_hue, 255, 255))
    grouped = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 5), np.uint8))
    _, labels, components, _ = cv2.connectedComponentsWithStats(grouped)
    return labels, components


def _number_shape(width, height, area):
    return 11 <= width <= 46 and 4 <= height <= 13 and area >= 95


def _number_near_player(cx, cy, point):
    px, py = point
    return abs(cx - px) <= 25 and -36 <= cy - py <= 42


def floating_numbers(hsv, time_ms, players):
    numbers = []
    _, components = _number_components(hsv, 83)
    for x, y, width, height, area in components[1:]:
        if not _number_shape(width, height, area):
            continue
        cx, cy = x + width / 2, y + height / 2
        if any(_number_near_player(cx, cy, point) for point in players.values()):
            numbers.append(FloatingNumber(time_ms, cx, cy, int(area)))
    return numbers


def _continue_overlapping_numbers(hsv, time, tracks, used_tracks, players, *, previous_frame_ms=None):
    """Continue rising tracks when a cyan effect merges into an oversized green region."""
    eligible = {}
    for index, track in enumerate(tracks):
        if index in used_tracks:
            continue
        first, last = track.samples[0], track.samples[-1]
        pending = getattr(track, "pending_overlap", [])
        if pending and time - pending[-1].time_ms > 160:
            pending.clear()
        rising = len(track.samples) >= 2 and first.y - last.y >= 1.5
        recent = pending[-1] if pending else last
        age = time - first.time_ms
        usable = (time - recent.time_ms <= 160 and (
            (rising and age <= 1150) or (recent.time_ms == previous_frame_ms and age <= NUMBER_DECISION_MS)
        ))
        if usable:
            regions = [point for point in players.values() if _number_near_player(first.x, first.y, point)]
            eligible[index] = (first, recent, regions, rising, pending)
        else:
            pending.clear()
    if not eligible:
        return
    wide_labels, wide = _number_components(hsv, 83)
    tall = {index for index, (_, _, width, height, area) in enumerate(wide[1:], 1)
            if 11 <= width <= 46 and height > 13 and area >= 95}
    if not tall:
        return
    possible = defaultdict(list)
    narrow_labels, narrow = _number_components(hsv, 76)
    for index, (x, y, width, height, area) in enumerate(narrow[1:], 1):
        if not _number_shape(width, height, area):
            continue
        bounds = np.s_[y:y + height, x:x + width]
        enclosing = set(wide_labels[bounds][narrow_labels[bounds] == index])
        if len(enclosing) != 1 or not enclosing <= tall:
            continue
        cx, cy = x + width / 2, y + height / 2
        matches = [index for index, (first, last, regions, _, _) in eligible.items()
                   if abs(cx - first.x) <= 6 and abs(cx - last.x) <= 6 and -10 <= cy - last.y <= 1
                   and any(_number_near_player(cx, cy, point) for point in regions)]
        for index in matches:
            possible[index].append((FloatingNumber(time, cx, cy, int(area)), len(matches)))
    for index, candidates in possible.items():
        track = tracks[index]
        first, _, _, rising, pending = eligible[index]
        if len(candidates) != 1 or candidates[0][1] != 1:
            pending.clear()
            continue
        number = candidates[0][0]
        if rising:
            track.samples.append(number)
        else:
            pending.append(number)
            track.pending_overlap = pending
            if len(pending) >= 2 and first.y - number.y >= 2.5:
                track.samples.extend(pending)
                pending.clear()
                track.recovered_from_weak_seed = True


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
        pet_owners = vision_pet_owners(config)
        # Keep a side for ranking unresolved visual slots without publishing a guessed owner.
        self.actor_owners.update({f"{slot}_PET": owner or slot for slot, owner in pet_owners.items()})
        self.unverified_actors = {f"{slot}_PET" for slot, owner in pet_owners.items() if owner is None}
        self.allow_weak_amber = config.get("medicine_allow_weak_amber") is True
        self.action_first = config.get("medicine_action_first") is True
        self.recover_cyan_overlap = config.get("medicine_recover_cyan_overlap") is True
        self.pet_body_points = ({f"{owner}_PET": xy for owner, xy in config["pets"].items()}
                                if config.get("medicine_check_pet_body") is True else {})
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
        body_detail = self._observe_pet_body(image) if image is not None and getattr(self, "pet_body_points", None) else {}
        self.times.append(time)
        self.frames.append(MedicineFrame(time, activity, health, impact, body_detail))
        self._track_numbers(time, floating_numbers(hsv, time, self.players),
                            overlap_hsv=hsv if getattr(self, "recover_cyan_overlap", False) else None)

    def _observe_pet_body(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        observed = {}
        for actor, (x, y) in self.pet_body_points.items():
            tile = gray[max(0, y + 8):y + 33, max(0, x - 10):x + 10]
            if tile.shape == (25, 20):
                gx = cv2.Sobel(tile, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(tile, cv2.CV_32F, 0, 1, ksize=3)
                observed[actor] = (float(np.mean(cv2.magnitude(gx, gy) > 70)), float(tile.std()))
        return observed

    def _pet_body_exclusions(self, track, target, summons, ready):
        if not getattr(self, "pet_body_points", None) or not getattr(track, "recovered_from_weak_seed", False):
            return set()
        when = track.samples[0].time_ms
        prior = self._window(when - 1320, when - 800)
        arrivals = {row.get("owner_id") for row in summons
                    if row["event_type"] in ("SUMMON_PET", "SUMMON_CHILD")
                    and when - 1320 <= row.get("end_ms", row["time_ms"])
                    and row.get("start_ms", row["time_ms"]) <= when
                    and (row.get("decision_ready_ms") is None or row["decision_ready_ms"] <= ready)}
        exclusions = set()
        for actor in self.pet_body_points:
            if actor[0] != target[0] or self.actor_owners[actor] in arrivals:
                continue
            details = [getattr(frame, "pet_body_detail", {}).get(actor) for frame in prior]
            values = [detail for detail in details if detail is not None]
            if (len(values) >= 4 and median(edges for edges, _ in values) < 0.08
                    and median(deviation for _, deviation in values) < 16):
                exclusions.add(actor)
        return exclusions

    @staticmethod
    def _amber_coverage(mask, center):
        x, y = center
        height, width = mask.shape
        tile = mask[max(0, y + 7):min(height, y + 55), max(0, x - 19):min(width, x + 19)]
        return np.count_nonzero(tile) / tile.size if tile.size else 0.0

    def _track_numbers(self, time, numbers, *, overlap_hsv=None):
        previous = self.times[-2] if len(self.times) >= 2 else None
        active = []
        for track in self.active:
            last_time = track.samples[-1].time_ms
            broken_recovery = getattr(track, "recovered_from_weak_seed", False) and last_time != previous
            if time - last_time > 250 or broken_recovery:
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
                if getattr(active[index], "pending_overlap", None):
                    active[index].pending_overlap.clear()
                used_tracks.add(index)
                used_hits.add(hit)
        if overlap_hsv is not None:
            _continue_overlapping_numbers(overlap_hsv, time, active, used_tracks, self.players,
                                          previous_frame_ms=previous)
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

    def _rank_actors(self, when, target, amber_onset, *, excluded=()):
        prior = self._window(when - 1320, when - 800)
        action = self._window(when - 760, min(when - 120, amber_onset - 40))
        if len(action) < 3:
            return []
        allied = [actor for actor, owner in self.actor_owners.items()
                  if owner[0] == target[0] and actor not in excluded]
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
        resolved = self._resolve_nearby_tracks(events, skills)
        if self.action_first:
            additional = self._classify_caster_motion(resolved, skills, summons)
            resolved = sorted([*resolved, *additional], key=lambda row: (row["time_ms"], row["target_id"] or ""))
        return attach_medicine_trigger_hints(resolved, skills)

    def _caster_motion(self):
        """Propose coherent motion from fixed actor regions before looking at replies."""
        allied = {side: [actor for actor in self.actor_owners if actor.startswith(side)]
                  for side in ("L", "R")}
        active = {}
        motions = []
        for sample in self.frames:
            before = self._window(sample.time_ms - 760, sample.time_ms - 240)
            if len(before) < 3:
                continue
            for actor in self.actor_owners:
                idle = median(frame.activity.get(actor, 0.0) for frame in before)
                nearby = max((sample.activity.get(other, 0.0)
                              for other in allied[actor[0]] if other != actor), default=0.0)
                contrast = sample.activity.get(actor, 0.0) - 0.45 * nearby - 0.25 * min(idle, 0.28)
                if contrast < 0.14:
                    continue
                start, previous = active.get(actor, (sample.time_ms, sample.time_ms))
                if sample.time_ms - previous > 160:
                    if previous - start >= 60:
                        motions.append(CasterMotion(actor, start, previous))
                    start = sample.time_ms
                active[actor] = (start, sample.time_ms)
        motions.extend(CasterMotion(actor, start, end) for actor, (start, end) in active.items()
                       if end - start >= 60)
        return motions

    def _classify_caster_motion(self, existing, skills, summons):
        motions = self._caster_motion()
        recoveries = sorted((track.samples[0].time_ms, index, track)
                            for index, track in enumerate((*self.completed, *self.active)))
        recovery_times = [time for time, _, _ in recoveries]
        candidates = {}
        for motion in motions:
            if motion.actor in self.unverified_actors:
                continue
            owner = self.actor_owners[motion.actor]
            end = min(motion.start_ms + 860, motion.end_ms + 680)
            first = bisect_left(recovery_times, motion.start_ms + 160)
            last = bisect_right(recovery_times, end)
            for when, index, track in recoveries[first:last]:
                # One sustained animation must lead the recovery, not flicker throughout it.
                if when < motion.end_ms:
                    continue
                targets = self._rank_targets(track.samples[0])
                if not targets:
                    continue
                score, target, impact, amber, hp, onset = targets[0]
                gap = score - targets[1][0] if len(targets) > 1 else score
                if (target[0] != owner[0] or gap < 0.034 or onset <= motion.start_ms
                        or amber < 0.075):
                    continue
                # Weak or stationary number tracks require independently visible recovery.
                visible_numbers = sum(sample.time_ms <= when + 380 for sample in track.samples)
                if (hp is None or hp < 2) and (impact < 0.10 or visible_numbers < 2):
                    continue
                candidates.setdefault(index, []).append((motion, target, hp, amber))

        by_motion = {}
        credited = [row for row in existing if row["event_type"] == "USE_MEDICINE"]
        for when, index, track in recoveries:
            options = candidates.get(index, [])
            if len(options) != 1:
                continue
            motion, target, hp, amber = options[0]
            actor = motion.actor
            # A nearby default candidate can take 660 ms to finish its own disambiguation.
            end = min(motion.start_ms + 860, motion.end_ms + 680)
            decision = bisect_left(self.times, max(end, when + 280) + 660)
            if decision == len(self.times):
                continue
            ready = self.times[decision]
            if actor in self._pet_body_exclusions(track, target, summons, ready):
                continue
            conflict, _ = medicine_skill_context(skills, actor, motion.start_ms, when)
            if conflict:
                continue
            # Legacy template summons lack confirmation timestamps and veto overlaps offline.
            if any(row["event_type"] == "SUMMON_PET" and row.get("actor_id") == actor
                   and (row.get("decision_ready_ms") is None or row["decision_ready_ms"] <= ready)
                   and row.get("end_ms", row["time_ms"]) >= motion.start_ms and row["time_ms"] <= when
                   for row in summons):
                continue
            owner = self.actor_owners[actor]
            if any(row["decision_ready_ms"] <= ready and (
                (row["actor_id"] == actor and motion.start_ms <= row["time_ms"] <= end)
                or (row["owner_id"][0] == owner[0] and abs(row["time_ms"] - when) <= 280)
            ) for row in credited):
                continue
            event = dict(
                event_type="USE_MEDICINE", actor_id=actor, owner_id=owner, target_id=target,
                time_ms=when, start_ms=motion.start_ms, end_ms=when, status="candidate",
                decision_ready_ms=ready, evidence={"action_first": True, "motion_end_ms": motion.end_ms,
                    "receiver_health_gain_px": hp, "receiver_amber_growth": round(amber, 3)},
            )
            by_motion.setdefault(motion, []).append(event)
        events = [rows[0] for rows in by_motion.values()
                  if len({row["target_id"] for row in rows}) == 1]
        events.sort(key=lambda row: (row["time_ms"], row["target_id"]))
        return self._resolve_nearby_tracks(events, skills)

    def _classify_track(self, track, skills, summons):
        first = track.samples[0]
        decision = bisect_left(self.times, first.time_ms + NUMBER_DECISION_MS)
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
        conflict, triggers = medicine_skill_context(skills, actor, first.time_ms - 480,
                                                    first.time_ms + 30, recovery_ms=first.time_ms)
        ready = self.times[decision]
        excluded = set()
        if not conflict and not triggers and actor in self._pet_body_exclusions(track, target, (), ready):
            body_decision = bisect_left(self.times, first.time_ms + NUMBER_DECISION_MS + 280)
            if body_decision < len(self.times):
                ready = self.times[body_decision]
                candidates = self._pet_body_exclusions(track, target, summons, ready)
                if actor in candidates:
                    excluded = candidates
                    actors = self._rank_actors(first.time_ms, target, amber_onset, excluded=excluded)
                    top, actor = actors[0] if actors else (0, None)
                    conflict, triggers = medicine_skill_context(skills, actor, first.time_ms - 480,
                                                                first.time_ms + 30, recovery_ms=first.time_ms)
        actor_gap = top - actors[1][0] if len(actors) > 1 else top
        source_strong = top >= 0.14 and actor_gap >= 0.053
        health_increase = health is not None and health >= 1
        receiver_strong = target_gap >= 0.034 and amber >= 0.075 and (health_increase or impact >= 0.07)
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
        if excluded:
            evidence["low_detail_pet_slots"] = sorted(excluded)
        accepted = source_eligible and receiver_strong
        event_type = "USE_MEDICINE" if accepted else ("NON_MEDICINE" if conflict else "UNKNOWN")
        return mark_medicine_trigger(dict(
            event_type=event_type, actor_id=actor if accepted else None,
            owner_id=self.actor_owners[actor] if accepted else None,
            target_id=target if receiver_strong else None,
            time_ms=first.time_ms, start_ms=first.time_ms, end_ms=samples[-1].time_ms,
            status="candidate", decision_ready_ms=ready, evidence=evidence,
        ), [] if accepted else triggers)

    def _resolve_nearby_tracks(self, events, skills=()):
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
                actor = first["actor_id"]
                triggers = [trigger for event in group for trigger in event.get("payload", {}).get("medicine_trigger_hints", ())]
                decision = bisect_left(self.times, first["time_ms"] + NUMBER_DECISION_MS + 280)
                if decision == len(self.times):
                    first.update(event_type="UNKNOWN", actor_id=None, owner_id=None)
                    first["evidence"]["recording_ended_during_disambiguation"] = True
                else:
                    first["decision_ready_ms"] = max(self.times[decision], max(row["decision_ready_ms"] for row in group))
                if len({event["target_id"] for event in group}) > 1:
                    first.update(event_type="UNKNOWN", actor_id=None, owner_id=None)
                    first["evidence"]["overlapping_receivers"] = True
                    first["evidence"]["receiver_observations"] = [
                        {"target_id": event["target_id"], "time_ms": event["time_ms"]}
                        for event in group
                    ]
                if first["event_type"] == "UNKNOWN":
                    triggers.extend(name for performer, name, cue in skills
                                    if any(_matches_medicine_trigger(performer, name, cue, actor, event["time_ms"])
                                           for event in group))
                resolved.append(mark_medicine_trigger(first, triggers))
        return sorted(resolved, key=lambda row: (row["time_ms"], row["target_id"] or ""))
