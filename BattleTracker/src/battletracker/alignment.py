"""Piecewise alignment of two recordings of a battle."""
from bisect import bisect_right
from dataclasses import dataclass


@dataclass(frozen=True)
class Anchor:
    anchor_id: str
    detailed_ms: int
    clean_ms: int
    cue: str
    round: int


class TimeAlignment:
    def __init__(self, anchors):
        self.anchors = tuple(anchors)
        self._times = tuple(anchor.detailed_ms for anchor in self.anchors)
        if len(self.anchors) < 2:
            raise ValueError("alignment requires at least two anchors")
        if len({anchor.anchor_id for anchor in self.anchors}) != len(self.anchors):
            raise ValueError("anchor IDs must be unique")
        for anchor in self.anchors:
            if min(anchor.detailed_ms, anchor.clean_ms) < 0 or anchor.round < 1 or not anchor.cue:
                raise ValueError("anchor requires nonnegative times and a battle cue")
        for left, right in zip(self.anchors, self.anchors[1:]):
            if (right.detailed_ms <= left.detailed_ms or right.clean_ms <= left.clean_ms
                    or right.round < left.round):
                raise ValueError("anchors must increase on both timelines")

    def map(self, detailed_ms):
        if not self._times[0] <= detailed_ms <= self._times[-1]:
            return None
        index = min(bisect_right(self._times, detailed_ms) - 1, len(self.anchors) - 2)
        left, right = self.anchors[index:index + 2]
        fraction = (detailed_ms - left.detailed_ms) / (right.detailed_ms - left.detailed_ms)
        return round(left.clean_ms + fraction * (right.clean_ms - left.clean_ms))

    def validate(self, checks):
        fitted = set(self._times)
        results = []
        for check in checks:
            if check.detailed_ms in fitted:
                raise ValueError("checkpoint repeats a fitting anchor")
            predicted = self.map(check.detailed_ms)
            if predicted is None:
                raise ValueError("checkpoint is outside the overlap")
            results.append(dict(anchor_id=check.anchor_id, round=check.round, cue=check.cue,
                                detailed_ms=check.detailed_ms, clean_ms=check.clean_ms,
                                predicted_clean_ms=predicted, error_ms=predicted - check.clean_ms))
        return results


def check_grouping(assets, samples):
    registered = {}
    groups = {}
    hashes = {}
    for asset in assets:
        asset_id = asset["asset_id"]
        if asset_id in registered:
            raise ValueError(f"duplicate asset ID {asset_id}")
        registered[asset_id] = asset
        battle, split = asset.get("battle_id"), asset.get("dataset_split")
        if not battle and not split:
            continue
        if not battle or not split:
            raise ValueError("asset group requires both battle ID and dataset split")
        if groups.setdefault(battle, split) != split:
            raise ValueError(f"battle crosses dataset splits: {battle}")
        digest = asset["sha256"]
        if not digest or hashes.setdefault(digest, (battle, split)) != (battle, split):
            raise ValueError("same content assigned to different battle groups")
    for sample in samples:
        asset = registered.get(sample["source_video_id"])
        if not asset or asset["role"] != "clean_match_replay" or asset["status"] != "available":
            raise ValueError("dataset clip requires a registered clean replay")
        if (not asset["battle_id"] or sample["battle_id"] != asset["battle_id"] or
                sample["dataset_split"] != asset["dataset_split"]):
            raise ValueError("clip group does not match its source replay")
    return groups

