"""Extract observation intervals from a calibrated detailed replay."""
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from fractions import Fraction
from itertools import groupby

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


@dataclass(frozen=True)
class MedicineReading:
    timestamp_ms: int
    round_number: int | None
    display: str | None


@dataclass(frozen=True)
class _StableMedicineDisplay:
    round_number: int
    used: int
    limit: int
    start_ms: int
    end_ms: int


def _stable_medicine_displays(readings, max_gap_ms):
    stable = []
    last_time = -1
    for (round_number, display), group in groupby(readings, key=lambda read: (read.round_number, read.display)):
        if round_number is not None and (type(round_number) is not int or round_number < 1):
            raise ValueError("invalid medicine reading round")
        start, count = None, 0
        used, limit = (None, None) if display is None else map(int, display.split("/"))
        if display is not None and not 0 <= used <= limit:
            raise ValueError("medicine reading outside its limit")
        for read in group:
            time = read.timestamp_ms
            if type(time) is not int or time <= last_time:
                raise ValueError("medicine readings must have increasing nonnegative timestamps")
            if start is None or time - last_time > max_gap_ms:
                if round_number is not None and display is not None and count >= 2:
                    stable.append(_StableMedicineDisplay(round_number, used, limit, start, last_time))
                start, count = time, 0
            count += 1
            last_time = time
        if round_number is not None and display is not None and count >= 2:
            stable.append(_StableMedicineDisplay(round_number, used, limit, start, last_time))
    return stable


def settle_medicine_waiver(readings, *, action_ms, round_number, max_gap_ms):
    """Pair a reviewed waiver with stable display evidence through the following round.

    The action anchor is supplied independently of the bottle values. This function
    never promotes a numerical regression to an action or edits the source evidence.
    """
    if type(max_gap_ms) is not int or max_gap_ms <= 0:
        raise ValueError("sample gap must be a positive integer")
    if type(action_ms) is not int or action_ms < 0 or type(round_number) is not int or round_number < 1:
        raise ValueError("invalid reviewed medicine action")
    readings = tuple(readings)
    displays = _stable_medicine_displays(readings, max_gap_ms)
    clock = [read.timestamp_ms for read in readings]
    before = [item for item in displays if item.round_number == round_number and item.start_ms < action_ms]
    if not before or before[-1].end_ms < action_ms - max_gap_ms:
        return {"status": "unresolved", "reason": "no_stable_baseline"}
    baseline = before[-1]
    following = [item for item in displays if item.start_ms > baseline.start_ms]
    if not following or following[0].start_ms > action_ms + max_gap_ms:
        return {"status": "unresolved", "reason": "no_anchored_increase"}
    rise = following[0]
    if (rise.round_number != round_number or rise.start_ms - baseline.end_ms > max_gap_ms
            or rise.limit != baseline.limit or rise.used != baseline.used + 1):
        return {"status": "unresolved", "reason": "no_anchored_increase"}

    other_charges = []
    unread_intervals = []
    last = rise
    for display in following[1:]:
        if display.start_ms - last.end_ms > max_gap_ms:
            same_plateau = (display.round_number == last.round_number == round_number
                            and display.used == last.used and display.limit == last.limit)
            section = readings[bisect_left(clock, last.end_ms):bisect_right(clock, display.start_ms)]
            continuous = all(right.timestamp_ms - left.timestamp_ms <= max_gap_ms
                             for left, right in zip(section, section[1:]))
            round_reads = [read.timestamp_ms for read in section if read.round_number == round_number]
            compatible = all(read.round_number in (None, round_number)
                             and read.display in (None, f"{last.used}/{last.limit}") for read in section)
            short_round_gaps = all(right - left <= max_gap_ms for left, right in zip(round_reads, round_reads[1:]))
            if not same_plateau or not continuous or not compatible or not short_round_gaps:
                return {"status": "unresolved", "reason": "missing_display_evidence"}
            unread_intervals.append([last.end_ms, display.start_ms])
        if display.limit != baseline.limit:
            return {"status": "unresolved", "reason": "limit_changed"}
        if display.round_number == round_number + 1:
            if display.used != last.used - 1:
                return {"status": "unresolved", "reason": "no_stable_rollback_at_round_change"}
            return {
                "status": "settled",
                "waived_attempts": 1,
                "net_waiver_delta": 0,
                "baseline_used": baseline.used,
                "temporary_first_ms": rise.start_ms,
                "rollback_first_ms": display.start_ms,
                "settled_used": display.used,
                "limit": display.limit,
                "other_charges": other_charges,
                "unread_same_value_intervals_ms": unread_intervals,
            }
        if display.round_number != round_number or display.used - last.used not in (0, 1):
            return {"status": "unresolved", "reason": "ambiguous_intervening_change"}
        if display.used > last.used:
            other_charges.append({"before_ms": last.end_ms, "after_ms": display.start_ms,
                                  "used_before": last.used - 1, "used_after": display.used - 1})
        last = display
    return {"status": "unresolved", "reason": "no_stable_rollback_at_round_change"}


def _medicine_review_windows(readings, action_ms, round_number, max_gap_ms):
    windows = [(max(0, action_ms - 400), action_ms + 350)]
    lower_reads = []
    round_edge = None
    last_round_time = None
    later_round = None
    previous = None
    for read in readings:
        time = read.timestamp_ms
        if time < action_ms:
            continue
        if read.round_number is not None and read.round_number > round_number + 1:
            if read.round_number == later_round:
                break
            later_round = read.round_number
            previous = None
            continue
        later_round = None
        if read.round_number == round_number:
            last_round_time = time
        if (round_edge is None and read.round_number == round_number + 1
                and last_round_time is not None and time - last_round_time <= max_gap_ms):
            round_edge = (max(0, last_round_time - 200), time + 250)
        if read.display is None:
            continue
        used, limit = map(int, read.display.split("/"))
        if previous and read.round_number in (None, round_number + 1):
            last_time, last_round, last_used, last_limit = previous
            if (last_round in (None, round_number) and time - last_time <= max_gap_ms
                    and limit == last_limit and used == last_used - 1):
                lower_reads.append((max(0, last_time - 200), time + 250))
        previous = (time, read.round_number, used, limit)
    windows.extend(lower_reads)
    if round_edge is not None:
        windows.append(round_edge)
    merged = []
    for start, end in sorted(windows):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def settle_medicine_waiver_adaptively(readings, read_native, *, action_ms, round_number, max_gap_ms):
    """Recheck a sparse replay around the reviewed action and visible count or round changes.

    Read every available source frame within each requested inclusive window, using
    the same timestamp and OCR rules as the sparse readings. A result covers the
    observed frames, so an unsampled change between windows can still be missed.
    Use this for offline review without updating the ledger.
    """
    readings = tuple(readings)
    arguments = dict(action_ms=action_ms, round_number=round_number, max_gap_ms=max_gap_ms)
    coarse = settle_medicine_waiver(readings, **arguments)
    windows = _medicine_review_windows(readings, action_ms, round_number, max_gap_ms)
    combined = {read.timestamp_ms: read for read in readings}
    reread_times = set()
    for start, end in windows:
        for read in read_native(start, end):
            time = read.timestamp_ms
            if not start <= time <= end:
                raise ValueError("native reading is outside its requested window")
            if time in combined and combined[time] != read:
                raise ValueError("native reading conflicts with an existing frame")
            combined[time] = read
            reread_times.add(time)
    clock = sorted(combined)
    result = settle_medicine_waiver((combined[time] for time in clock), **arguments)
    if result["status"] == "settled":
        for edge in ("temporary_first_ms", "rollback_first_ms"):
            index = bisect_left(clock, result[edge])
            if not set(clock[index:index + 2]) <= reread_times:
                result = {"status": "unresolved", "reason": "native_edge_not_confirmed"}
                break
    return {"coarse": coarse, "result": result, "native_windows_ms": windows, "samples": len(combined)}


