"""Local, offline visual cues for the calibrated clean battle replay."""
from collections import deque
from dataclasses import dataclass

import av
import cv2
import numpy as np


def enemy_gauge_rectangles(config):
    """Approximate replay gauge regions using calibrated 700×320 slot offsets."""
    width, height = config["width"], config["height"]
    half_width = round(18 * width / 700)
    above = round(5 * height / 320)
    regions = []
    for slots, below in ((config["players"], 14), (config["pets"], 6)):
        below = round(below * height / 320)
        for owner, (x, y) in slots.items():
            if owner.startswith("L"):
                left, right = max(0, x - half_width), min(width, x + half_width)
                top, bottom = max(0, y - above), min(height, y + below)
                if left < right and top < bottom:
                    regions.append((left, top, right, bottom))
    return regions


def frames(path, config, interval_ms=None, requested=()):
    times = deque(sorted(set(requested)))
    tick = 0 if interval_ms else None
    origin = None
    gauge_rectangles = enemy_gauge_rectangles(config) if config.get("participant_view") is True else ()
    with av.open(str(path)) as video:
        for frame in video.decode(video=0):
            if frame.pts is None or frame.time_base is None or frame.is_corrupt:
                continue
            pts = frame.pts * frame.time_base
            if origin is None:
                origin = pts
            elapsed = (pts - origin) * 1000
            matched = []
            while times and elapsed >= times[0]:
                matched.append(times.popleft())
            regular = tick is not None and elapsed >= tick
            if regular:
                tick = (elapsed // interval_ms + 1) * interval_ms
            if matched or regular:
                image = frame.reformat(width=config["width"], height=config["height"],
                                       format="rgb24").to_ndarray()
                for x1, y1, x2, y2 in gauge_rectangles:
                    image[y1:y2, x1:x2] = 0
                yield round(elapsed), matched, image
            if tick is None and not times:
                break


def roi(image, center, half_width, top, bottom):
    x, y = center
    return image[max(0, y + top):y + bottom, max(0, x - half_width):x + half_width]


def gold_mask(hsv):
    return cv2.inRange(hsv, (12, 80, 172), (39, 255, 255))


def glyph_mask(hsv):
    return cv2.inRange(hsv, (0, 0, 184), (179, 135, 255))


def score(region, pattern):
    height, width = pattern.shape
    if region.shape[0] < height or region.shape[1] < width:
        return 0.0
    return float(cv2.matchTemplate(region, pattern, cv2.TM_CCOEFF_NORMED).max())


@dataclass
class Cue:
    start: int
    end: int
    peak: float
    hits: int = 1


class CueTracker:
    def __init__(self, gap=200, minimum_hits=2):
        self.active = {}
        self.gap = gap
        self.minimum_hits = minimum_hits

    def update(self, time, hits):
        ended = []
        for key in tuple(self.active):
            cue = self.active[key]
            if time - cue.end > self.gap:
                self.active.pop(key)
                if cue.hits >= self.minimum_hits:
                    ended.append((key, cue))
        for key, strength in hits.items():
            cue = self.active.get(key)
            if cue is None:
                self.active[key] = Cue(time, time, strength)
            else:
                cue.end = time
                cue.peak = max(cue.peak, strength)
                cue.hits += 1
        return ended


def heal_glyphs(mask):
    grouped = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 5), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(grouped)
    return [(x + w / 2, y + h / 2, int(area)) for x, y, w, h, area in stats[1:]
            if 11 <= w <= 46 and 3 <= h <= 13]


def color_signature(image, center):
    tile = roi(image, center, 18, 3, 31)
    channels = [np.histogram(tile[:, :, i], bins=8, range=(0, 256))[0] for i in range(3)]
    values = np.concatenate(channels)
    return values / values.sum()


def motion_onset(before, after, center):
    a = roi(before, center, 18, 5, 34)
    b = roi(after, center, 18, 5, 34)
    return float(np.mean(cv2.absdiff(a, b) > 28))


def extract_patterns(path, config):
    gauge_rectangles = enemy_gauge_rectangles(config) if config.get("participant_view") is True else ()

    def intersects_mask(entry):
        left, top, right, bottom = entry["rect"]
        return any(left < x2 and right > x1 and top < y2 and bottom > y1
                   for x1, y1, x2, y2 in gauge_rectangles)

    # A partly hidden skill name can resemble a different action. Entry
    # templates can still use the visible part of an overlapping effect.
    entries = config["portal_templates"] + [entry for entry in config["skill_templates"]
                                          if not intersects_mask(entry)]
    source = {request: cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
              for _, matches, image in frames(path, config, requested=[x["time_ms"] for x in entries])
              for request in matches}
    patterns = []
    for entry in entries:
        x1, y1, x2, y2 = entry["rect"]
        crop = source[entry["time_ms"]][y1:y2, x1:x2]
        pattern = glyph_mask(crop) if "name" in entry else gold_mask(crop)
        if not pattern.any() or pattern.all():
            if intersects_mask(entry):
                continue
            raise ValueError(f"empty or solid visual template: {entry}")
        patterns.append((entry, pattern))
    return patterns


def observe(path, config, template_path=None, template_config=None, *, observers=()):
    template_layout = dict(template_config or config,
                           participant_view=config.get("participant_view") is True)
    patterns = extract_patterns(template_path or path, template_layout)
    thresholds = config["thresholds"]
    players, pets = config["players"], config["pets"]
    actor_points = {**players, **{f"{owner}_PET": xy for owner, xy in pets.items()}}
    portal_centers = {**pets, **config["portal_centers"]}
    skill_patterns = [(item, pattern) for item, pattern in patterns if "name" in item]
    portal_patterns = [pattern for item, pattern in patterns if "name" not in item]
    timeline = []
    cues = CueTracker()
    skill_cues = CueTracker(gap=380, minimum_hits=1)
    heals = CueTracker(gap=220)
    completed, healing = [], []
    previous = None
    for time, _, image in frames(path, config, interval_ms=config["sample_ms"]):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        signature = {owner: color_signature(image, center) for owner, center in pets.items()}
        activity = {} if previous is None else {
            actor: motion_onset(previous, gray, point) for actor, point in actor_points.items()
        }
        previous = gray
        timeline.append((time, activity, signature))
        for observer in observers:
            observer.observe(time, image, hsv, activity)
        gold, glyphs = gold_mask(hsv), glyph_mask(hsv)
        hits = {}
        for owner, center in portal_centers.items():
            mask = roi(gold, center, 37, -19, 17)
            strength = max((score(mask, pattern) for pattern in portal_patterns), default=0.0)
            if strength >= thresholds["portal"]:
                hits["portal", owner, None] = strength
        text_hits = {}
        for item, pattern in skill_patterns:
            x1, y1, x2, y2 = item["rect"]
            mask = glyphs[max(0, y1 - 10):y2 + 10, x1 - 14:x2 + 14]
            strength = score(mask, pattern)
            if strength >= thresholds["skill"]:
                text_hits["skill", item["actor_id"], item["name"]] = strength
        completed.extend(cues.update(time, hits))
        completed.extend(skill_cues.update(time, text_hits))

        green = cv2.inRange(hsv[:224], (43, 65, 180), (88, 255, 255))
        targets = {}
        for x, y, area in heal_glyphs(green):
            if area < thresholds["heal_area"]:
                continue
            possible = [(abs(x - px) + abs(y - py + 8), owner)
                        for owner, (px, py) in players.items()
                        if abs(x - px) <= 25 and -26 <= y - py <= 1]
            if possible:
                target = min(possible)[1]
                targets[target] = max(targets.get(target, 0), area)
        healing.extend(heals.update(time, targets))
    if not timeline:
        raise ValueError("clean replay produced no sampled frames")
    finish = timeline[-1][0] + 1000
    completed.extend(cues.update(finish, {}))
    completed.extend(skill_cues.update(finish, {}))
    healing.extend(heals.update(finish, {}))
    return timeline, completed, healing


def classify(timeline, cues, healing, config, *, include_medicine=True):
    events = []
    skills = [(actor, name, cue) for (kind, actor, name), cue in cues if kind == "skill"]
    for actor, name, cue in skills:
        events.append(dict(event_type="SKILL", skill=name, actor_id=actor,
                           owner_id=actor.removesuffix("_PET") if actor else None,
                           target_id=None, start_ms=cue.start, end_ms=cue.end, time_ms=cue.start,
                           status="candidate", evidence={"glyph_template_score": round(cue.peak, 3)}))

    for (kind, owner, _), cue in cues:
        if kind != "portal":
            continue
        before = [sample[2][owner] for sample in timeline
                  if cue.start - 500 <= sample[0] < cue.start - 90]
        after = [sample[2][owner] for sample in timeline
                 if cue.end + 160 <= sample[0] <= cue.end + 650]
        difference = (float(np.abs(np.median(before, axis=0) - np.median(after, axis=0)).sum())
                      if before and after else 0.0)
        actor_motion = max((sample[1].get(owner, 0) for sample in timeline
                            if cue.start - 450 <= sample[0] <= cue.start + 80), default=0)
        exclusion = [name for actor, name, skill in skills
                     if name in {"死亡召唤", "高级神佑复生"}
                     and actor in {None, f"{owner}_PET"}
                     and skill.start <= cue.end + 120 and skill.end >= cue.start - 120]
        corroborated = (difference >= config["thresholds"]["pet_change"]
                        and actor_motion >= 0.07 and not exclusion)
        events.append(dict(event_type="SUMMON_PET" if corroborated else "UNKNOWN",
                           actor_id=owner if corroborated else None, owner_id=owner,
                           target_id=f"{owner}_PET", start_ms=cue.start, end_ms=cue.end,
                           time_ms=cue.end, status="candidate",
                           evidence={"portal_score": round(cue.peak, 3),
                                     "pet_appearance_change": round(difference, 3),
                                     "player_motion": round(actor_motion, 3),
                                     "excluded_by_text": exclusion}))

    for target, cue in healing if include_medicine else ():
        motion = {}
        for time, activity, _ in timeline:
            if cue.start - 640 <= time <= cue.start - 130:
                for actor, value in activity.items():
                    motion[actor] = max(motion.get(actor, 0), value)
        ranking = sorted(motion.items(), key=lambda item: (-item[1], item[0]))[:3]
        blocked = [name for _, name, skill in skills if skill.start <= cue.start + 80
                   and skill.end >= cue.start - 550]
        top = ranking[0][1] if ranking else 0
        margin = top - ranking[1][1] if len(ranking) >= 2 else 0
        actor = ranking[0][0] if top >= 0.14 and margin >= 0.07 and not blocked else None
        event_type = "NON_MEDICINE" if blocked else ("USE_MEDICINE" if actor else "UNKNOWN")
        events.append(dict(event_type=event_type, actor_id=actor,
                           owner_id=actor.removesuffix("_PET") if actor else None,
                           target_id=target, start_ms=cue.start, end_ms=cue.end, time_ms=cue.start,
                           status="candidate", evidence={"heal_area": cue.peak,
                           "actor_motion_candidates": [[key, round(value, 3)] for key, value in ranking],
                           "nearby_skill_templates": blocked}))
    return sorted(events, key=lambda row: (row["time_ms"], row["event_type"]))



