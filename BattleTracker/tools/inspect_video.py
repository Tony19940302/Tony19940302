"""Inspect a video and extract PNGs at decoded presentation timestamps."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import av


def time_value(value: str) -> Fraction:
    """Parse seconds exactly, including decimal input."""
    try:
        seconds = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError("expected seconds as a decimal or fraction") from exc
    if seconds < 0:
        raise argparse.ArgumentTypeError("seconds must be nonnegative")
    return seconds


def inspect_video(
    path: Path, requests: list[Fraction], output: Path | None
) -> dict:
    """Decode to EOF and sample the first frame at or after each requested time."""
    with path.open("rb") as source:
        sha256 = hashlib.file_digest(source, "sha256").hexdigest()

    samples = []
    requests = sorted(set(requests))
    next_request = 0
    frames = 0
    missing_pts = 0
    corrupt_frames = 0
    origin = None
    last_pts = None
    last_export = None

    with av.open(str(path), mode="r") as video:
        if not video.streams.video:
            raise ValueError("source has no video stream")
        stream = video.streams.video[0]
        if stream.disposition & av.stream.Disposition.attached_pic:
            raise ValueError("first video stream is an attachment")
        for index, frame in enumerate(video.decode(stream)):
            frames += 1
            corrupt_frames += int(frame.is_corrupt)
            if frame.pts is None or frame.time_base is None:
                missing_pts += 1
                continue

            pts_seconds = frame.pts * frame.time_base
            if origin is None:
                origin = pts_seconds
            last_pts = pts_seconds
            timestamp = pts_seconds - origin
            while next_request < len(requests) and timestamp >= requests[next_request]:
                assert output is not None
                image = output / f"{path.stem}-{sha256[:12]}-frame-{index:08d}.png"
                if last_export != index:
                    output.mkdir(parents=True, exist_ok=True)
                    frame.to_image().save(image)
                    last_export = index
                samples.append({
                    "request_seconds": float(requests[next_request]),
                    "frame_index": index,
                    "pts": frame.pts,
                    "time_base": str(frame.time_base),
                    "pts_seconds": float(pts_seconds),
                    "timestamp_seconds": float(timestamp),
                    "image": str(image),
                })
                next_request += 1

        if not frames:
            raise ValueError("source produced no decoded frames")
        duration = stream.duration * stream.time_base if (
            stream.duration is not None and stream.time_base is not None
        ) else None
        return {
            "path": str(path),
            "filename": path.name,
            "sha256": sha256,
            "size_bytes": path.stat().st_size,
            "format": video.format.name,
            "codec": stream.codec_context.name,
            "stream_index": stream.index,
            "width": stream.width,
            "height": stream.height,
            "average_frame_rate": str(stream.average_rate) if stream.average_rate else None,
            "base_frame_rate": str(stream.base_rate) if stream.base_rate else None,
            "stream_time_base": str(stream.time_base) if stream.time_base else None,
            "stream_start_pts": stream.start_time,
            "declared_frames": stream.frames or None,
            "decoded_frames": frames,
            "stream_duration_seconds": float(duration) if duration is not None else None,
            "container_duration_seconds": (
                video.duration / av.time_base if video.duration is not None else None
            ),
            "first_pts_seconds": float(origin) if origin is not None else None,
            "last_pts_seconds": float(last_pts) if last_pts is not None else None,
            "frames_without_timestamp": missing_pts,
            "corrupt_frames": corrupt_frames,
            "samples": samples,
            "unmatched_request_seconds": [float(t) for t in requests[next_request:]],
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Times are relative to the first decoded frame with a presentation timestamp.",
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--at", type=time_value, action="append", default=[], metavar="SECONDS")
    parser.add_argument(
        "--range", nargs=3, type=time_value, metavar=("START", "END", "STEP"),
        help="sample this inclusive time grid in seconds",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/video-inspection"))
    args = parser.parse_args(argv)

    requests = args.at
    if args.range:
        start, end, step = args.range
        if start > end or step <= 0:
            parser.error("--range requires START <= END and STEP > 0")
        points = int((end - start) // step) + 1
        requests.extend(start + index * step for index in range(points))

    path = args.video.resolve()
    if not path.is_file():
        parser.error(f"video file not found: {path}")

    try:
        result = inspect_video(path, requests, args.output.resolve())
    except (av.FFmpegError, OSError, ValueError) as exc:
        print(f"Video inspection failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return int(bool(result["unmatched_request_seconds"] or result["corrupt_frames"]))


if __name__ == "__main__":
    raise SystemExit(main())
