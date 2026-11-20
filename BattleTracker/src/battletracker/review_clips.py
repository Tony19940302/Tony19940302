"""Export small, silent review copies of clean-video source intervals."""
from contextlib import ExitStack
from fractions import Fraction

import av


def export_review_clips(video, samples, output):
    clips = output / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    requests = iter(sorted(samples, key=lambda row: row["clip_start_ms"]))
    upcoming = next(requests, None)
    active = []
    completed = set()
    origin = None
    rate = 30
    clock = Fraction(1, rate)
    with av.open(str(video)) as source, ExitStack() as cleanup:
        for frame in source.decode(video=0):
            if frame.pts is None or frame.time_base is None or frame.is_corrupt:
                continue
            pts = frame.pts * frame.time_base
            if origin is None:
                origin = pts
            timestamp = (pts - origin) * 1000
            while upcoming is not None and upcoming["clip_start_ms"] <= timestamp:
                # Sample IDs may contain colons, which Windows does not allow in file names.
                name = f"clip-{len(completed) + len(active) + 1:03}.mp4"
                writer = cleanup.enter_context(av.open(str(clips / name), mode="w"))
                stream = writer.add_stream("libx264", rate=rate, options={"preset": "fast"})
                stream.width = 960
                stream.height = 2 * round(960 * frame.height / frame.width / 2)
                stream.pix_fmt = "yuv420p"
                upcoming["review_media"] = f"clips/{name}"
                active.append((upcoming, writer, stream, -1))
                upcoming = next(requests, None)

            scaled = None
            for item in active[:]:
                row, writer, stream, last = item
                if timestamp >= row["clip_end_ms"]:
                    for packet in stream.encode():
                        writer.mux(packet)
                    writer.close()
                    active.remove(item)
                    if last < 0:
                        raise ValueError(f"source produced no frame for {row['sample_id']}")
                    completed.add(row["sample_id"])
                    continue
                index = int((timestamp - row["clip_start_ms"]) / 1000 * rate)
                if index <= last:
                    continue
                if scaled is None:
                    scaled = frame.reformat(width=stream.width, height=stream.height,
                                            format="rgb24").to_ndarray()
                review_frame = av.VideoFrame.from_ndarray(scaled, format="rgb24")
                review_frame.pts, review_frame.time_base = index, clock
                for packet in stream.encode(review_frame):
                    writer.mux(packet)
                active[active.index(item)] = (row, writer, stream, index)
            if upcoming is None and not active:
                break
    if len(completed) != len(samples):
        raise ValueError("clean video ended before all review intervals finished")

    for row in samples:
        with av.open(str(output / row["review_media"])) as clip:
            frames = sum(1 for _ in clip.decode(video=0))
        if frames == 0:
            raise ValueError(f"review clip is not decodable: {row['sample_id']}")
        row["review_media_frames"] = frames
