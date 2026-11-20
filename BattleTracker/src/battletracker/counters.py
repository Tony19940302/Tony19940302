"""Extract observation intervals from a calibrated detailed replay."""
from dataclasses import dataclass
from fractions import Fraction

import av
import cv2
import numpy as np


FIELDS = ("medicine", "pet")


def sampled_regions(path, config, requested=(), step=None):
    """Decode in presentation order and emit only small crops with source PTS."""
    points = iter(sorted({Fraction(str(time)) for time in requested}))
    next_point = next(points, None)
    next_regular = Fraction(0) if step is not None else None
    if step is not None and step <= 0:
        raise ValueError("step must be positive")
    with av.open(str(path)) as video:
        stream = video.streams.video[0]
        if (stream.width, stream.height) != (config["width"], config["height"]):
            raise ValueError("source dimensions do not match this calibration")
        origin = None
        for frame in video.decode(stream):
            if frame.pts is None or frame.time_base is None or frame.is_corrupt:
                continue
            pts = frame.pts * frame.time_base
            if origin is None:
                origin = pts
            elapsed = pts - origin
            labels = []
            while next_point is not None and elapsed >= next_point:
                labels.append(float(next_point))
                next_point = next(points, None)
            regular = next_regular is not None and elapsed >= next_regular
            if regular:
                next_regular = (elapsed // step + 1) * step
            if not labels and not regular:
                if step is None and next_point is None:
                    break
                continue
            rgb = frame.to_ndarray(format="rgb24")

            def crop(rect):
                x1, y1, x2, y2 = (2 * value for value in rect)
                return rgb[y1:y2, x1:x2].copy()

            regions = {(None, "round"): crop(config["round"])}
            for side, formation in config.get("formations", {}).items():
                regions[side, "formation"] = crop(formation["region"])
            for player, rect in config["players"].items():
                both = crop(rect)
                regions[player, "medicine"] = both[:36, :]
                regions[player, "pet"] = both[36:, :]
            yield dict(timestamp_ms=round(elapsed * 1000), pts=frame.pts,
                       time_base=str(frame.time_base), requests=labels), regions


def feature(rgb, field):
    if field == "round":
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return cv2.threshold(gray, 168, 255, cv2.THRESH_BINARY)[1]
    r, g, b = np.moveaxis(rgb.astype(np.int16), -1, 0)
    gold = (r >= 92) & (r - g >= 9) & (r - g <= 100) & (g - b >= 18)
    return gold.astype(np.uint8) * 255


class TemplateReader:
    def __init__(self, labels, frames, template_bounds=None):
        self.templates = {}
        template_bounds = template_bounds or {}
        for time, player, field, value in labels:
            image = feature(frames[time][player, field], field)
            # Keep the numeric text and enough surrounding background to reject partial reads.
            if field in template_bounds:
                x1, y1, x2, y2 = template_bounds[field]
                height, width = image.shape
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError(f"template bounds outside {field} region")
                image = image[y1:y2, x1:x2]
            elif field == "round":
                image = image[7:79, 6:74]
            else:
                x2 = 76 if field == "medicine" else 65
                image = image[2:-2, 6:x2]
            if not image.any():
                raise ValueError(f"empty template {time} {player} {field}")
            self.templates.setdefault(field, []).append((value, image))

    def read(self, rgb, field):
        query = feature(rgb, field)
        scores = {}
        for label, template in self.templates[field]:
            match = cv2.matchTemplate(query, template, cv2.TM_CCOEFF_NORMED)
            score = float(match.max())
            scores[label] = max(scores.get(label, -1), score)
        ranks = sorted(scores, key=scores.get, reverse=True)
        label = ranks[0]
        score = scores[label]
        margin = score - scores[ranks[1]] if len(ranks) > 1 else score
        threshold = {"round": 0.80, "medicine": 0.64, "pet": 0.62}[field]
        minimum_margin = 0.055 if field == "medicine" else 0.08
        return {
            "value": label if score >= threshold and margin >= minimum_margin else None,
            "best": label, "score": round(score, 3), "margin": round(margin, 3),
        }


@dataclass
class StableValue:
    value: object = None
    last_seen: int | None = None
    pending: object = None
    pending_start: int | None = None

    def observe(self, value, timestamp):
        """Require two consecutive observations. Unknown resets pending evidence."""
        if value is None:
            self.pending = self.pending_start = None
            return None, None
        if value == self.value:
            self.last_seen = timestamp
            self.pending = self.pending_start = None
            return self.value, None
        if value != self.pending:
            self.pending, self.pending_start = value, timestamp
            return None, None
        change = (self.value, value, self.last_seen, self.pending_start)
        self.value, self.last_seen = value, timestamp
        self.pending = self.pending_start = None
        return self.value, change


def changes_for(player, field, change, frame, round_number, prior_round):
    before, after, left, right = change
    if before is None:
        return []
    old_used, old_limit = map(int, before.split("/"))
    new_used, new_limit = map(int, after.split("/"))
    common = dict(owner_id=player, actor_id=None, earliest_timestamp_ms=left,
                  latest_timestamp_ms=right, confirmed_timestamp_ms=frame["timestamp_ms"],
                  earliest_round=prior_round, latest_round=round_number,
                  source="official_counter", before=before, after=after)
    result = []
    if new_used != old_used:
        kind = "USE_MEDICINE" if field == "medicine" else "SUMMON_PET"
        result.append(dict(common, event_type=kind, delta=new_used - old_used,
                           needs_review=new_used - old_used != 1))
    if new_limit != old_limit:
        result.append(dict(common, event_type=f"{field.upper()}_LIMIT_CHANGE",
                           delta=new_limit - old_limit, needs_review=True))
    return result


