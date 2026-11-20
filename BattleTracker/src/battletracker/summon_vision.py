"""Corroborate pet entry effects using the signals visible in each view."""
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


REFERENCE_SIZE = (700, 320)
LOOK_AHEAD_MS = 620
MODEL_LOOK_AHEAD_MS = 960
RING_FLOOR = 0.33
PROBE_DURATION_MS = 550
MIN_NEW_BAR_WIDTH = 21


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


@dataclass(frozen=True)
class Observation:
    time: int
    bars: dict
    rings: dict
    activity: dict
    models: dict


class SummonObserver:
    def __init__(self, config):
        width, height = config["width"], config["height"]
        self.pets = {owner: (x * REFERENCE_SIZE[0] / width, y * REFERENCE_SIZE[1] / height)
                     for owner, (x, y) in config["pets"].items()}
        participant = config.get("participant_view") is True
        self.model_pets = {owner: point for owner, point in self.pets.items()
                           if participant and owner.startswith("L")}
        points = {**self.pets, **{f"{owner}:actor": (x * REFERENCE_SIZE[0] / width,
                                                  y * REFERENCE_SIZE[1] / height)
                                for owner, (x, y) in config["players"].items()}}
        self.points = {actor: point for actor, point in points.items()
                       if not participant or not actor.startswith("L")}
        self.patterns = ring_patterns()
        self._history = deque()
        self._pending = deque()
        self._pending_models = deque()
        self._events = []
        self._last_entry = {}

    def observe(self, time, image, hsv, activity):
        if hsv.shape[1::-1] != REFERENCE_SIZE:
            scaled = cv2.resize(image, REFERENCE_SIZE, interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(scaled, cv2.COLOR_RGB2HSV)
        models = {owner: model_signature(hsv, point) for owner, point in self.model_pets.items()}
        bars = health_bars(hsv, self.points) if self.points else {}
        item = Observation(time, bars,
                           ring_candidates(hsv, self.patterns, self.pets), activity, models)
        self._history.append(item)
        self._pending.append(item)
        while self._pending and time >= self._pending[0].time + LOOK_AHEAD_MS:
            self._test_entry(self._pending.popleft(), time)
        if any(owner in item.rings for owner in self.model_pets):
            self._pending_models.append(item)
        while self._pending_models and time >= self._pending_models[0].time + MODEL_LOOK_AHEAD_MS:
            self._test_model_entry(self._pending_models.popleft(), time)
        look_ahead = MODEL_LOOK_AHEAD_MS if self.model_pets else LOOK_AHEAD_MS
        while self._history and time - self._history[0].time > look_ahead + PROBE_DURATION_MS + 250:
            self._history.popleft()

    def _test_entry(self, item, ready):
        before = [probe for probe in self._history
                  if item.time - PROBE_DURATION_MS <= probe.time < item.time]
        after = [probe for probe in self._history if item.time <= probe.time <= ready]
        if len(before) < 4 or len(after) < 4:
            return
        for owner, (_, x, y, width) in item.bars.items():
            if (owner not in self.pets or width < MIN_NEW_BAR_WIDTH
                    or item.time - self._last_entry.get(owner, -2000) < 1250):
                continue
            preceding = [probe for probe in before if owner in probe.bars]
            if len(preceding) > 1 or (preceding and preceding[0] is before[-1]):
                continue
            samples = [probe for probe in after if owner in probe.bars
                       and abs(probe.bars[owner][1] - x) <= 6 and abs(probe.bars[owner][2] - y) <= 6]
            if len(samples) < 4 or samples[-1].time - item.time < 340:
                continue
            candidates = [(probe.time, strength, rx, ry, rays) for probe in (*before, *after)
                          if probe.time >= item.time - 250 and owner in probe.rings
                          for strength, rx, ry, rays in (probe.rings[owner],)
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
            self._last_entry[owner] = item.time
            self._events.append(dict(
                event_type="SUMMON_PET", actor_id=owner, owner_id=owner, target_id=f"{owner}_PET",
                start_ms=min(start, item.time), end_ms=max(end, item.time),
                time_ms=max(end, item.time), decision_ready_ms=ready, status="candidate",
                evidence={"entry_ring_score": round(peak, 3), "entry_ring_samples": len(consistent),
                          "entry_ray_samples": ray_frames, "player_health_bar_samples": alive_frames,
                          "prior_health_bar_samples": len(preceding), "new_health_bar_samples": len(samples),
                          "health_bar_first_ms": item.time, "player_motion": round(motion, 3)}))

    def _test_model_entry(self, item, ready):
        before = [probe for probe in self._history
                  if item.time - PROBE_DURATION_MS <= probe.time < item.time]
        after = [probe for probe in self._history if item.time <= probe.time <= ready]
        for owner, (strength, x, y, _) in item.rings.items():
            if (owner not in self.model_pets or strength < 0.39
                    or item.time - self._last_entry.get(owner, -2000) < 1250):
                continue
            rings = [(probe.time, score, rays) for probe in after if owner in probe.rings
                     for score, rx, ry, rays in (probe.rings[owner],)
                     if abs(rx - x) <= 10 and abs(ry - y) <= 8 and score >= 0.39]
            ray_frames = sum(rays for _, _, rays in rings)
            motion = max((probe.activity.get(owner, 0.0) for probe in (*before, item)), default=0.0)
            if len(rings) < 3 or not ray_frames or motion < 0.07:
                continue
            last_ring = rings[-1][0]
            initial = [probe.models[owner] for probe in before
                       if probe.time < item.time - 120 and probe.models[owner] is not None]
            settled = [(probe.time, probe.models[owner]) for probe in after
                       if probe.time >= last_ring + 120 and probe.models[owner] is not None][-4:]
            if len(initial) < 3 or len(settled) < 3 or settled[-1][0] - settled[0][0] < 160:
                continue
            previous = np.asarray(initial)
            current = np.asarray([signature for _, signature in settled])
            initial_model = np.median(previous, axis=0)
            settled_model = np.median(current, axis=0)
            separation = float(np.abs(settled_model - initial_model).sum())
            prior_drift = float(np.median(np.abs(previous - initial_model).sum(axis=1)))
            later_drift = float(np.max(np.abs(current - settled_model).sum(axis=1)))
            if separation < 0.18 or max(prior_drift, later_drift) > separation / 2:
                continue
            self._last_entry[owner] = item.time
            # Use the first corroborated ring frame for the action timestamp.
            # The candidate is still only available after the new model settles.
            self._events.append(dict(
                event_type="SUMMON_PET", actor_id=owner, owner_id=owner, target_id=f"{owner}_PET",
                start_ms=item.time, end_ms=last_ring, time_ms=item.time, decision_ready_ms=ready,
                status="candidate", evidence={
                    "entry_ring_score": round(max(score for _, score, _ in rings), 3),
                    "entry_ring_samples": len(rings), "entry_ray_samples": ray_frames,
                    "pet_appearance_change": round(separation, 3), "settled_model_samples": len(settled),
                    "player_motion": round(motion, 3), "enemy_gauges_available": False}))

    def events(self):
        return list(self._events)


def merge_summons(events, corroborated):
    """Keep other event types and collapse equivalent template/entry detections."""
    exclusions = [row for row in events if row["event_type"] == "SKILL"
                  and row.get("skill") in {"死亡召唤", "高级神佑复生"}]
    corroborated = [row for row in corroborated if not any(
        skill.get("actor_id") in {None, row["target_id"]}
        and skill["start_ms"] <= row["end_ms"] + 320
        and skill["end_ms"] >= row["start_ms"] - 250 for skill in exclusions)]
    kept = [row for row in events if row["event_type"] != "SUMMON_PET" or not any(
        # Template events are timestamped at the end of their visible cue.
        row["owner_id"] == updated["owner_id"] and abs(row["time_ms"] - updated["end_ms"]) <= 650
        for updated in corroborated)]
    return sorted([*kept, *corroborated], key=lambda row: (row["time_ms"], row["event_type"]))
