"""Corroborate pet entry effects using the signals visible in each view."""
from bisect import bisect_left, bisect_right
from collections import deque
from dataclasses import dataclass, field
from statistics import median

import cv2
import numpy as np

from .formation import vision_pet_owners


REFERENCE_SIZE = (700, 320)
LOOK_AHEAD_MS = 620
MODEL_LOOK_AHEAD_MS = 960
IDENTITY_LOOK_AHEAD_MS = 1520
BEAM_RETURN_LOOK_AHEAD_MS = 2240
BEAM_RETURN_PRIOR_MS = 1420
RING_FLOOR = 0.33
PROBE_DURATION_MS = 550
MIN_NEW_BAR_WIDTH = 21
BEAM_AREA_FLOOR = 400
BEAM_AREA_PEAK = 700
EXPANDED_NAME_SHAPE = (49, 62)
MAX_RETAINED_NAME_GAP_MS = 550


def ring_patterns():
    patterns = []
    for horizontal, vertical in ((14, 4), (15, 5), (15, 6), (16, 5), (17, 6)):
        x, y = horizontal + 3, vertical + 3
        pattern = np.zeros((2 * y + 1, 2 * x + 1), np.uint8)
        cv2.ellipse(pattern, (x, y), (horizontal, vertical), 0, 0, 360, 255, 1, cv2.LINE_AA)
        patterns.append((pattern, x, y))
    return patterns


def entry_rays(gold, x, y):
    below = gold[y + 6:y + 47, max(0, x - 23):x + 24]
    if below.shape[0] < 13 or not below.shape[1]:
        return False
    lines = cv2.morphologyEx(below, cv2.MORPH_OPEN, np.ones((13, 1), np.uint8))
    columns = np.count_nonzero(lines, axis=0) >= 13
    bands = np.diff(np.pad(columns.astype(np.int8), (1, 1))) == 1
    return bool(np.count_nonzero(lines) >= 180 and np.count_nonzero(bands) >= 3)


def ring_candidates(hsv, patterns, pets):
    gold = cv2.inRange(hsv, (12, 70, 150), (39, 255, 255))
    bright = cv2.inRange(hsv, (12, 70, 170), (37, 255, 255))
    bright = cv2.resize(bright, (350, 160), interpolation=cv2.INTER_AREA)
    best = {}
    for pattern, center_x, center_y in patterns:
        matched = cv2.matchTemplate(bright, pattern, cv2.TM_CCOEFF_NORMED)
        height, width = matched.shape
        for owner, (x, y) in pets.items():
            x, y = round((x - 10) / 2) - center_x, round((y - 7) / 2) - center_y
            left, right = max(0, x - 10), min(width, x + 11)
            top, bottom = max(0, y - 11), min(height, y + 8)
            if left >= right or top >= bottom:
                continue
            _, strength, _, (dx, dy) = cv2.minMaxLoc(matched[top:bottom, left:right])
            if strength >= RING_FLOOR and (owner not in best or strength > best[owner][0]):
                best[owner] = (strength, 2 * (left + dx + center_x), 2 * (top + dy + center_y))
    return {owner: (strength, x, y, entry_rays(gold, x, y))
            for owner, (strength, x, y) in best.items()}


def health_bars(hsv, points):
    red = cv2.inRange(hsv, (0, 128, 135), (9, 255, 255))
    red |= cv2.inRange(hsv, (174, 128, 135), (179, 255, 255))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((1, 4), np.uint8))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((1, 12), np.uint8))
    _, _, stats, centers = cv2.connectedComponentsWithStats(red)
    bars = {}
    for (left, _, width, height, _), (x, y) in zip(stats[1:], centers[1:], strict=True):
        if not (12 <= width <= 38 and 2 <= height <= 7):
            continue
        # The filled part is left-aligned. A partially filled bar has an
        # off-center centroid, so estimate its anchor from its left edge.
        x = max(x, left + 13)
        distances = [(abs(x - px) + 1.2 * abs(y - py), owner)
                     for owner, (px, py) in points.items()
                     if abs(x - px) <= 13 and abs(y - py) <= 12]
        if distances:
            distance, owner = min(distances)
            bar = (distance, float(x), float(y), int(width))
            if owner not in bars or bar < bars[owner]:
                bars[owner] = bar
    return bars


def model_signature(hsv, anchor):
    x, y = map(round, anchor)
    # Sample the model below the gauge and the typical entry-ring center.
    tile = hsv[max(0, y + 20):y + 47, max(0, x - 18):x + 18]
    if not tile.size:
        return None
    hist = cv2.calcHist([tile], [0, 1, 2], None, [6, 3, 4], [0, 180, 0, 256, 0, 256])
    return (hist / hist.sum()).ravel()


def entry_beam_detail(image, hsv, points):
    """Measure fixed-slot gold columns and body texture without reading gauges."""
    gold = cv2.inRange(hsv, (12, 70, 150), (39, 255, 255))
    lines = cv2.morphologyEx(gold, cv2.MORPH_OPEN, np.ones((13, 1), np.uint8))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    detail = {}
    for slot, point in points.items():
        x, y = map(round, point)
        tile = lines[max(0, y - 1):y + 38, max(0, x - 33):x + 14]
        columns = np.count_nonzero(tile, axis=0) >= 13
        bands = np.count_nonzero(np.diff(np.pad(columns.astype(np.int8), (1, 1))) == 1)
        body = gray[y + 20:y + 47, max(0, x - 18):x + 18]
        if body.shape != (27, 36):
            continue
        gx = cv2.Sobel(body, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(body, cv2.CV_32F, 0, 1, ksize=3)
        edges = np.mean(cv2.magnitude(gx, gy) > 70)
        detail[slot] = (int(np.count_nonzero(tile)), int(bands), float(body.std()), float(edges))
    return detail


def nameplate_shape(hsv, anchor, owner, *, expanded=False):
    """A visible label cue, with no OCR or identity inference from a missing label."""
    x, y = map(round, anchor)
    top, bottom = (44, 93) if expanded else (54, 76)
    tile = hsv[y + top:y + bottom, x - 31:x + 31]
    if tile.shape[:2] != (bottom - top, 62):
        return None
    low, high = (79, 107) if owner.startswith("L") else (38, 79)
    mask = (cv2.inRange(tile, (low, 70, 75), (high, 255, 255)) != 0).astype(np.uint8)
    # Keep a narrow horizontal band to reduce colored model and spell spill.
    density = np.sum(mask, axis=1)
    radius = 4 if expanded else 3
    middle = int(np.argmax(np.convolve(density, np.ones(2 * radius + 1), mode="same")))
    mask[:max(0, middle - radius)] = 0
    mask[middle + radius + 1:] = 0
    min_pixels, max_pixels = (60, 420) if expanded else (12, 180)
    return mask if min_pixels <= int(mask.sum()) <= max_pixels else None


def shape_overlap(left, right):
    padded = cv2.copyMakeBorder(left, 3, 3, 4, 4, cv2.BORDER_CONSTANT, value=0)
    _, intersection, _, _ = cv2.minMaxLoc(cv2.matchTemplate(padded, right, cv2.TM_CCORR))
    intersection = round(intersection)
    return intersection / (int(left.sum()) + int(right.sum()) - intersection)


def observation_streak(times, min_samples, min_span):
    start = 0
    for i, time in enumerate(times):
        if i and time - times[i - 1] > 120:
            start = i
        if i - start + 1 >= min_samples and time - times[start] >= min_span:
            return True
    return False


@dataclass(frozen=True)
class VisiblePet:
    time: int
    since: int
    name: np.ndarray
    model: np.ndarray
    model_drift: float


@dataclass(frozen=True)
class Observation:
    time: int
    bars: dict
    rings: dict
    activity: dict
    models: dict
    names: dict
    beam_detail: dict = field(default_factory=dict)
    beam_names: dict = field(default_factory=dict)


class SummonObserver:
    def __init__(self, config):
        width, height = config["width"], config["height"]
        self.pets = {slot: (x * REFERENCE_SIZE[0] / width, y * REFERENCE_SIZE[1] / height)
                     for slot, (x, y) in config["pets"].items()}
        self.pet_owners = vision_pet_owners(config)
        participant = config.get("participant_view") is True
        self.track_identity = config.get("summon_track_identity") is True
        self.model_pets = {owner: point for owner, point in self.pets.items()
                           if participant and owner.startswith("L")}
        self.late_settle = (bool(self.model_pets) and not self.track_identity
                            and config.get("summon_late_settle") is True)
        self.vacant_beam = bool(self.model_pets) and config.get("summon_vacant_beam") is True
        self.beam_return = self.vacant_beam and config.get("summon_beam_return") is True
        self.guard_retained = participant and config.get("summon_guard_retained") is True
        points = {**self.pets, **{f"{owner}:actor": (x * REFERENCE_SIZE[0] / width,
                                                  y * REFERENCE_SIZE[1] / height)
                                for owner, (x, y) in config["players"].items()}}
        self.points = {actor: point for actor, point in points.items()
                       if not participant or not actor.startswith("L")}
        self.patterns = ring_patterns()
        self._history = deque()
        self._pending = deque()
        self._pending_models = deque()
        self._pending_late_models = deque()
        self._pending_beams = deque()
        self._pending_beam_returns = deque()
        self._events = []
        self._late_events = []
        self._last_entry = {}
        self._visible_pets = {owner: [] for owner in self.pets} if self.track_identity else {}
        self._retained_views = {slot: [] for slot in self.pets if self._pet_owner(slot) is not None
                                and self.guard_retained}

    def _pet_owner(self, slot):
        owners = getattr(self, "pet_owners", None)
        if owners is None:
            return slot if slot in self.pets else None
        return owners.get(slot)

    def observe(self, time, image, hsv, activity):
        if hsv.shape[1::-1] != REFERENCE_SIZE:
            image = cv2.resize(image, REFERENCE_SIZE, interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        model_points = self.pets if self.track_identity else self.model_pets
        models = {owner: model_signature(hsv, point) for owner, point in model_points.items()}
        names = ({owner: nameplate_shape(hsv, point, owner) for owner, point in self.pets.items()}
                 if self.track_identity else {})
        bars = health_bars(hsv, self.points) if self.points else {}
        beam_detail = entry_beam_detail(image, hsv, self.model_pets) if self.vacant_beam else {}
        beam_names = ({slot: nameplate_shape(hsv, point, slot, expanded=True)
                       for slot, point in self.model_pets.items()} if self.beam_return else {})
        for slot, views in self._retained_views.items():
            point = self.pets[slot]
            name = beam_names[slot] if slot in beam_names else nameplate_shape(hsv, point, slot, expanded=True)
            if name is None:
                continue
            model = models[slot] if slot in models else model_signature(hsv, point)
            if model is not None:
                views.append((time, np.packbits(name), model))
        item = Observation(time, bars,
                           ring_candidates(hsv, self.patterns, self.pets), activity, models, names, beam_detail, beam_names)
        self._history.append(item)
        self._pending.append(item)
        while self._pending and time >= self._pending[0].time + LOOK_AHEAD_MS:
            self._test_entry(self._pending.popleft(), time)
        if any(owner in item.rings for owner in self.model_pets):
            self._pending_models.append(item)
        model_window = IDENTITY_LOOK_AHEAD_MS if self.track_identity else MODEL_LOOK_AHEAD_MS
        while self._pending_models and time >= self._pending_models[0].time + model_window:
            self._test_model_entry(self._pending_models.popleft(), time)
        while (self._pending_late_models
               and time >= self._pending_late_models[0][0].time + IDENTITY_LOOK_AHEAD_MS):
            probe, owners = self._pending_late_models.popleft()
            self._test_model_entry(probe, time, late_owners=owners)
        if self.track_identity:
            self._track_pet_names(item)
        if self.vacant_beam:
            self._check_entry_beams(item)
        look_ahead = (IDENTITY_LOOK_AHEAD_MS if self.late_settle or self.vacant_beam else
                      model_window if self.model_pets or self.track_identity else LOOK_AHEAD_MS)
        history_ms = (BEAM_RETURN_LOOK_AHEAD_MS + BEAM_RETURN_PRIOR_MS + 250 if self.beam_return else
                      look_ahead + PROBE_DURATION_MS + 250)
        while self._history and time - self._history[0].time > history_ms:
            self._history.popleft()

    def _track_pet_names(self, item):
        recent = [probe for probe in self._history if item.time - 280 <= probe.time <= item.time]
        if len(recent) < 4 or item.time - recent[0].time < 220:
            return
        for slot, name in item.names.items():
            models = [probe.models[slot] for probe in recent]
            if name is None or any(model is None for model in models):
                continue
            if any(probe.names[slot] is None or shape_overlap(name, probe.names[slot]) < 0.48
                   for probe in recent[:-1]):
                continue
            observations = self._visible_pets[slot]
            prior = observations[-1] if observations else None
            signatures = np.asarray(models)
            model = np.median(signatures, axis=0)
            drift = float(np.abs(signatures - model).sum(axis=1).max())
            current = VisiblePet(item.time, recent[0].time, name, model, drift)
            observations.append(current)
            if (prior is None or current.since - prior.time < 160
                    or current.since - prior.time > 18000
                    or current.since - self._last_entry.get(slot, -2000) < 1250
                    or shape_overlap(prior.name, current.name) > 0.22):
                continue
            separation = float(np.abs(current.model - prior.model).sum())
            if separation < 0.30 or max(prior.model_drift, current.model_drift) > separation / 2:
                continue
            owner = self._pet_owner(slot)
            if owner is None:
                continue
            onset = [probe for probe in self._history
                     if current.since - 1200 <= probe.time <= current.since - 80]
            motion = max((probe.activity.get(owner, 0.0) for probe in onset), default=0.0)
            if motion < 0.18:
                continue
            # The entry time is the first frame of the new stable appearance,
            # while the event can only be emitted at the end of the stability window.
            self._last_entry[slot] = current.since
            evidence = {"stable_name_shape_change": True, "prior_name_visible_ms": prior.time,
                        "stable_name_frames": len(recent), "pet_appearance_change": round(separation, 3),
                        "player_motion": round(motion, 3),
                        "pet_model_drift": round(max(prior.model_drift, current.model_drift), 3)}
            # Merge with a template cue only when a single spatially consistent
            # ring episode was actually observed near this transition.
            ring = self._identity_ring(slot, onset, prior.time)
            if ring is not None:
                evidence["identity_ring_interval_ms"] = ring
            self._events.append(dict(
                event_type="SUMMON_PET", actor_id=owner, owner_id=owner, target_id=f"{slot}_PET",
                start_ms=current.since, end_ms=current.since, time_ms=current.since,
                decision_ready_ms=item.time, status="candidate", evidence=evidence))

    @staticmethod
    def _identity_ring(slot, onset, prior_time):
        samples = [(probe.time, probe.rings[slot]) for probe in onset
                   if probe.time > prior_time and slot in probe.rings and probe.rings[slot][0] >= 0.39]
        if len(samples) < 2:
            return None
        first, (_, x, y, _) = samples[0]
        if any(later - earlier > 160 or abs(rx - x) > 10 or abs(ry - y) > 8
               for (earlier, _), (later, (_, rx, ry, _)) in zip(samples, samples[1:])):
            return None
        return [first, samples[-1][0]]

    def _retained_pet(self, slot, start, end):
        # Require visible, consistent shapes throughout the interval. A gap is
        # unknown and must not be used to veto a same-looking new pet.
        views = [view for view in self._visible_pets.get(slot, ())
                 if start - 400 <= view.time <= end + 400]
        if not views or views[0].time > start - 160 or views[-1].time < end + 160:
            return False
        first = views[0]
        return first.model_drift <= 0.30 and all(
            right.time - left.time <= 160 and shape_overlap(first.name, right.name) >= 0.52
            and right.model_drift <= 0.30 and np.abs(first.model - right.model).sum() <= 0.30
            for left, right in zip(views, views[1:]))

    def filter_events(self, events):
        """Drop template-only summon cues with a continuously visible original pet."""
        def slot_for_event(row):
            target = row.get("target_id")
            if target is not None:
                return target.removesuffix("_PET") if target.endswith("_PET") else None
            owner = row["owner_id"]
            return next((slot for slot in self.pets if owner is not None and self._pet_owner(slot) == owner), None)

        return [row for row in events if row["event_type"] != "SUMMON_PET"
                or not self._retained_pet(slot_for_event(row), row["start_ms"], row["end_ms"])]

    def review_retained_entries(self, events):
        """Keep a weak summon cue as UNKNOWN when old-pet continuity is visible."""
        if not self.guard_retained:
            return events
        by_owner = {self._pet_owner(slot): slot for slot in self._retained_views}
        # Merging can keep the template row while dropping an equivalent late
        # beam row. Keep its independent entry evidence available to the review.
        beams = [row for row in self._late_events if row["evidence"].get("vacant_slot_gold_beam")]
        beams = _without_conflicting_skills(beams, events)
        result = []
        for row in events:
            evidence = row.get("evidence", {})
            if (row["event_type"] != "SUMMON_PET" or row.get("status") != "candidate"
                    or evidence.get("vacant_slot_gold_beam") or evidence.get("stable_name_shape_change")
                    or evidence.get("new_health_bar_samples") or evidence.get("entry_ray_samples", 0) >= 2):
                result.append(row)
                continue
            owner = row.get("owner_id")
            target = row.get("target_id")
            slot = target.removesuffix("_PET") if target and target.endswith("_PET") else by_owner.get(owner)
            if (slot not in self._retained_views or self._pet_owner(slot) != owner
                    or (target is not None and not target.endswith("_PET"))
                    or any(beam["owner_id"] == owner and beam["target_id"] == f"{slot}_PET"
                           and _intervals_near(row, beam) for beam in beams)):
                result.append(row)
                continue
            continuity = self._retained_name_evidence(slot, row["start_ms"], row["end_ms"])
            if continuity:
                row = dict(row, event_type="UNKNOWN", decision_ready_ms=max(
                    row.get("decision_ready_ms", row["end_ms"]), continuity["after_ms"][-1],
                    continuity["reference_ms"]),
                    evidence=dict(evidence, retained_pet=continuity))
            result.append(row)
        return result

    def _retained_name_evidence(self, slot, start, end):
        views = self._retained_views[slot]
        when = lambda view: view[0]
        low = bisect_left(views, start - 1800, key=when)
        high = bisect_right(views, end + 1150, key=when)
        prior, near, transition, later = [], [], [], []
        height, width = EXPANDED_NAME_SHAPE
        for time, packed_name, model in views[low:high]:
            # At the normal 80 ms cadence the preceding decoded frame can
            # be only 67 ms earlier. Use it when it still shows the old pet.
            if start - 350 < time <= start - 60:
                group = near
            elif time <= start - 350:
                group = prior
            elif time >= end + 400:
                group = later
            else:
                group = transition
            name = np.unpackbits(packed_name, count=height * width).reshape(EXPANDED_NAME_SHAPE)
            group.append((time, name, model))
        if not near or len(prior) < 3 or len(later) < 3:
            return None

        def matches(samples, name, model, name_floor, body_limit):
            return [time for time, sample, body in samples
                    if np.abs(model - body).sum() <= body_limit and shape_overlap(name, sample) >= name_floor]

        for reference_time, name, model in later:
            near_times = matches(near, name, model, 0.72, 0.65)
            if not near_times:
                continue
            prior_times = matches(prior, name, model, 0.82, 0.60)
            if not observation_streak(prior_times, 3, 140):
                continue
            later_times = matches(later, name, model, 0.82, 0.55)
            if not observation_streak(later_times, 3, 140):
                continue
            # Color can be washed out by the effect. Still require the label to
            # bridge the cue, with any interruption limited to a short interval.
            bridge = [near_times[-1],
                      *(time for time, sample, _ in transition if shape_overlap(name, sample) >= 0.72),
                      later_times[0]]
            if any(right - left > MAX_RETAINED_NAME_GAP_MS for left, right in zip(bridge, bridge[1:])):
                continue
            for i in range(2, len(later_times)):
                if observation_streak(later_times[:i + 1], 3, 140):
                    return {"slot_id": f"{slot}_PET", "reference_ms": reference_time, "before_ms": prior_times,
                            "near_cue_ms": near_times, "through_cue_ms": bridge[1:-1],
                            "after_ms": later_times[:i + 1]}
        return None

    def _test_entry(self, item, ready):
        before = [probe for probe in self._history
                  if item.time - PROBE_DURATION_MS <= probe.time < item.time]
        after = [probe for probe in self._history if item.time <= probe.time <= ready]
        if len(before) < 4 or len(after) < 4:
            return
        for slot, (_, x, y, width) in item.bars.items():
            owner = self._pet_owner(slot)
            if (owner is None or width < MIN_NEW_BAR_WIDTH
                    or item.time - self._last_entry.get(slot, -2000) < 1250):
                continue
            preceding = [probe for probe in before if slot in probe.bars]
            if len(preceding) > 1 or (preceding and preceding[0] is before[-1]):
                continue
            samples = [probe for probe in after if slot in probe.bars
                       and abs(probe.bars[slot][1] - x) <= 6 and abs(probe.bars[slot][2] - y) <= 6]
            if len(samples) < 4 or samples[-1].time - item.time < 340:
                continue
            candidates = [(probe.time, strength, rx, ry, rays) for probe in (*before, *after)
                          if probe.time >= item.time - 250 and slot in probe.rings
                          for strength, rx, ry, rays in (probe.rings[slot],)
                          if abs(rx + 10 - x) <= 11 and -21 <= ry - y <= 8]
            motion = max((probe.activity.get(owner, 0.0) for probe in (*before, item)), default=0.0)
            alive_frames = sum(f"{owner}:actor" in probe.bars for probe in (*before, item))
            if len(candidates) < 2 or motion < 0.07 or not alive_frames:
                continue
            _, peak, rx, ry, _ = max(candidates, key=lambda ring: ring[1])
            consistent = [sample for sample in candidates
                          if abs(sample[2] - rx) <= 10 and abs(sample[3] - ry) <= 8]
            ray_frames = sum(sample[4] for sample in consistent)
            if len(consistent) < 2 or ray_frames < 2 or peak < 0.39:
                continue
            start, end = consistent[0][0], consistent[-1][0]
            self._last_entry[slot] = item.time
            self._events.append(dict(
                event_type="SUMMON_PET", actor_id=owner, owner_id=owner, target_id=f"{slot}_PET",
                start_ms=min(start, item.time), end_ms=max(end, item.time),
                time_ms=max(end, item.time), decision_ready_ms=ready, status="candidate",
                evidence={"entry_ring_score": round(peak, 3), "entry_ring_samples": len(consistent),
                          "entry_ray_samples": ray_frames, "player_health_bar_samples": alive_frames,
                          "prior_health_bar_samples": len(preceding), "new_health_bar_samples": len(samples),
                          "health_bar_first_ms": item.time, "player_motion": round(motion, 3)}))

    @staticmethod
    def _stable_model_change(initial, settled):
        if len(settled) < 3 or settled[-1][0] - settled[0][0] < 160:
            return None
        previous = np.asarray(initial)
        current = np.asarray([signature for _, signature in settled])
        initial_model = np.median(previous, axis=0)
        settled_model = np.median(current, axis=0)
        separation = float(np.abs(settled_model - initial_model).sum())
        prior_drift = float(np.median(np.abs(previous - initial_model).sum(axis=1)))
        later_drift = float(np.max(np.abs(current - settled_model).sum(axis=1)))
        if separation < 0.18 or max(prior_drift, later_drift) > separation / 2:
            return None
        return separation

    def _test_model_entry(self, item, ready, *, late_owners=()):
        before = [probe for probe in self._history
                  if item.time - PROBE_DURATION_MS <= probe.time < item.time]
        after = [probe for probe in self._history if item.time <= probe.time <= ready]
        pending = []
        for slot, (strength, x, y, _) in item.rings.items():
            owner = self._pet_owner(slot)
            if owner is None or slot not in self.model_pets or strength < 0.39:
                continue
            if late_owners:
                if slot not in late_owners:
                    continue
            elif item.time - self._last_entry.get(slot, -2000) < 1250:
                continue
            rings = [(probe.time, score, rays) for probe in after if slot in probe.rings
                     for score, rx, ry, rays in (probe.rings[slot],)
                     if abs(rx - x) <= 10 and abs(ry - y) <= 8 and score >= 0.39]
            ray_frames = sum(rays for _, _, rays in rings)
            motion = max((probe.activity.get(owner, 0.0) for probe in (*before, item)), default=0.0)
            if len(rings) < 3 or not ray_frames or motion < 0.07:
                continue
            last_ring = rings[-1][0]
            initial = [probe.models[slot] for probe in before
                       if probe.time < item.time - 120 and probe.models[slot] is not None]
            if len(initial) < 3:
                continue
            settled = [(probe.time, probe.models[slot]) for probe in after
                       if probe.time >= last_ring + 120 and probe.models[slot] is not None][-4:]
            separation = self._stable_model_change(initial, settled)
            if separation is None:
                if self.late_settle and not late_owners:
                    pending.append(slot)
                continue
            if self.track_identity and self._retained_pet(slot, item.time, last_ring):
                continue
            # Use the first corroborated ring frame for the action timestamp.
            # The candidate is still only available after the new model settles.
            event = dict(
                event_type="SUMMON_PET", actor_id=owner, owner_id=owner, target_id=f"{slot}_PET",
                start_ms=item.time, end_ms=last_ring, time_ms=item.time, decision_ready_ms=ready,
                status="candidate", evidence={
                    "entry_ring_score": round(max(score for _, score, _ in rings), 3),
                    "entry_ring_samples": len(rings), "entry_ray_samples": ray_frames,
                    "pet_appearance_change": round(separation, 3), "settled_model_samples": len(settled),
                    "player_motion": round(motion, 3), "enemy_gauges_available": False})
            if late_owners:
                event["evidence"]["late_model_settle"] = True
                self._late_events.append(event)
            else:
                self._last_entry[slot] = item.time
                self._events.append(event)
        if pending:
            self._pending_late_models.append((item, tuple(pending)))

    def _check_entry_beams(self, item):
        recent = [probe for probe in self._history if item.time - 280 <= probe.time < item.time]
        if len(recent) >= 3:
            for slot, (area, _, _, _) in item.beam_detail.items():
                if (area >= BEAM_AREA_FLOOR and self._pet_owner(slot) is not None
                        and all(slot in probe.beam_detail and probe.beam_detail[slot][0] < BEAM_AREA_FLOOR
                                for probe in recent)):
                    self._pending_beams.append((item.time, slot))
        while self._pending_beams and item.time >= self._pending_beams[0][0] + IDENTITY_LOOK_AHEAD_MS:
            when, slot = self._pending_beams.popleft()
            self._test_vacant_beam(when, slot, item.time)
        while (self._pending_beam_returns
               and item.time >= self._pending_beam_returns[0][0] + BEAM_RETURN_LOOK_AHEAD_MS):
            when, slot = self._pending_beam_returns.popleft()
            self._test_vacant_beam(when, slot, item.time, returning=True)

    def _test_vacant_beam(self, when, slot, ready, *, returning=False):
        owner = self._pet_owner(slot)
        samples = [probe for probe in self._history if slot in probe.beam_detail]
        old = [probe for probe in samples if when - 550 <= probe.time <= when - 120]
        pulse = [probe for probe in samples if when <= probe.time <= when + 280
                 and probe.beam_detail[slot][0] >= BEAM_AREA_FLOOR]
        if len(old) < 4 or len(pulse) < 2:
            return
        peak = max(probe.beam_detail[slot][0] for probe in pulse)
        bands = max(probe.beam_detail[slot][1] for probe in pulse)
        motion = max((probe.activity.get(owner, 0.0) for probe in samples
                      if when - 520 <= probe.time <= when + 100), default=0.0)
        if peak < BEAM_AREA_PEAK or bands < 3 or motion < 0.16:
            return
        prior_deviation = median(probe.beam_detail[slot][2] for probe in old)
        prior_edges = median(probe.beam_detail[slot][3] for probe in old)
        if prior_deviation >= 16 or prior_edges >= 0.08:
            return
        settled = [probe for probe in samples if when + (1520 if returning else 950) <= probe.time <= ready
                   and probe.beam_detail[slot][0] < BEAM_AREA_FLOOR][-5:]
        if len(settled) < 4:
            if self.beam_return and not returning:
                self._pending_beam_returns.append((when, slot))
            return
        current_deviation = median(probe.beam_detail[slot][2] for probe in settled)
        current_edges = median(probe.beam_detail[slot][3] for probe in settled)
        change = self._stable_model_change([probe.models[slot] for probe in old],
                                           [(probe.time, probe.models[slot]) for probe in settled])
        if change is None or current_deviation < 25 or current_edges < 0.23:
            if self.beam_return and not returning:
                self._pending_beam_returns.append((when, slot))
            return
        return_evidence = self._returning_beam_model(when, slot, samples, settled) if returning else {}
        if returning and not return_evidence:
            return
        self._late_events.append(dict(
            event_type="SUMMON_PET", actor_id=owner, owner_id=owner, target_id=f"{slot}_PET",
            start_ms=when, end_ms=pulse[-1].time, time_ms=when, decision_ready_ms=ready,
            status="candidate", evidence={
                "vacant_slot_gold_beam": True, "beam_samples": len(pulse), "beam_peak": peak,
                "beam_bands": bands, "player_motion": round(motion, 3),
                "prior_texture": [round(prior_deviation, 3), round(prior_edges, 3)],
                "settled_texture": [round(current_deviation, 3), round(current_edges, 3)],
                "pet_appearance_change": round(change, 3), "enemy_gauges_available": False,
                **return_evidence}))

    @staticmethod
    def _returning_beam_model(when, slot, samples, settled):
        # Match the name near the beam to the stable, visible model on return.
        # Compare with earlier visible names so an old pet's return can veto it.
        prior = [probe for probe in samples if when - BEAM_RETURN_PRIOR_MS <= probe.time <= when - 400]
        if not observation_streak([probe.time for probe in prior], 8, 760):
            return None
        if any(probe.beam_detail[slot][0] >= BEAM_AREA_PEAK for probe in samples
               if when + 480 <= probe.time < settled[0].time):
            return None
        name = settled[-1].beam_names[slot]
        if name is None:
            return None

        def matches(probe):
            candidate = probe.beam_names.get(slot)
            return candidate is not None and shape_overlap(name, candidate) >= 0.82

        if any(matches(probe) for probe in prior) or not all(matches(probe) for probe in settled):
            return None
        if not observation_streak([probe.time for probe in settled], len(settled), 160):
            return None
        entry_names = [probe.time for probe in samples if when - 350 <= probe.time <= when + 180 and matches(probe)]
        gap = [probe.time for probe in samples if when + 310 <= probe.time <= when + 1100
               and probe.beam_detail[slot][2] < 16 and probe.beam_detail[slot][3] < 0.08 and not matches(probe)]
        if not observation_streak(entry_names, 2, 40) or not observation_streak(gap, 4, 320):
            return None
        return {"matching_name_on_return": True, "entry_name_samples": len(entry_names),
                "early_name_first_ms": entry_names[0], "vacancy_samples": len(gap),
                "return_confirmation_start_ms": settled[0].time}

    def events(self):
        # Resolve offline, once the default detector has considered every cue.
        # A late proposal must never occupy the default detector's cooldown.
        events = list(self._events)
        late = sorted(self._late_events, key=lambda event: bool(event["evidence"].get("matching_name_on_return")))
        for row in late:
            if not any(previous["owner_id"] == row["owner_id"]
                       and abs(previous["start_ms"] - row["start_ms"]) < 1250
                       for previous in events):
                events.append(row)
        return events


def _intervals_near(template, entry):
    return (abs(template["time_ms"] - entry["end_ms"]) <= 650
            or template["start_ms"] <= entry["end_ms"] and entry["start_ms"] <= template["end_ms"])


def _without_conflicting_skills(rows, events):
    exclusions = [row for row in events if row["event_type"] == "SKILL"
                  and row.get("skill") in {"死亡召唤", "高级神佑复生"}]
    return [row for row in rows if not any(
        skill.get("actor_id") in {None, row["target_id"]}
        and skill["start_ms"] <= row["end_ms"] + 320
        and skill["end_ms"] >= row["start_ms"] - 250 for skill in exclusions)]


def merge_summons(events, corroborated):
    """Keep other event types and collapse equivalent template/entry detections."""
    corroborated = _without_conflicting_skills(corroborated, events)
    fallback, primary = [], []
    for row in corroborated:
        evidence = row.get("evidence", {})
        (fallback if evidence.get("late_model_settle") or evidence.get("vacant_slot_gold_beam") else primary).append(row)
    corroborated = primary
    identity_matches = set()
    for update in corroborated:
        evidence = update.get("evidence", {})
        if not evidence.get("stable_name_shape_change"):
            continue
        start, end = evidence.get("identity_ring_interval_ms", (update["start_ms"], update["end_ms"]))
        candidates = [(abs(row["end_ms"] - end), i) for i, row in enumerate(events)
                      if i not in identity_matches and row["event_type"] == "SUMMON_PET"
                      and row["owner_id"] == update["owner_id"]
                      and row["start_ms"] <= end and start <= row["end_ms"]]
        if candidates:
            identity_matches.add(min(candidates)[1])
    kept = [row for i, row in enumerate(events) if i not in identity_matches
            and (row["event_type"] != "SUMMON_PET" or not any(
        # Template events are timestamped at the end of their visible cue.
        row["owner_id"] == updated["owner_id"]
        and abs(row["time_ms"] - updated["end_ms"]) <= 650
        for updated in corroborated if not updated.get("evidence", {}).get("stable_name_shape_change")))]
    # Preserve template-only output too when the late proposal covers the same entry.
    late = [update for update in fallback if not any(
        row["event_type"] == "SUMMON_PET" and row["owner_id"] == update["owner_id"]
        and (abs(row["time_ms"] - update["end_ms"]) <= 650
             or (update["evidence"].get("vacant_slot_gold_beam")
                 and row.get("target_id") in (None, update["target_id"])
                 and _intervals_near(row, update)))
        for row in kept)]
    return sorted([*kept, *corroborated, *late], key=lambda row: (row["time_ms"], row["event_type"]))
