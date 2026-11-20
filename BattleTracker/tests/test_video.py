from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import av
import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "inspect_video.py"


@pytest.fixture
def variable_rate_video(tmp_path):
    path = tmp_path / "irregular timing.mkv"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("ffv1", rate=25)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = Fraction(1, 1000)
        for pts in (120, 200, 480, 520):
            frame = av.VideoFrame(32, 32, "yuv420p")
            for index, plane in enumerate(frame.planes):
                plane.update(bytes([90 if index == 0 else 128]) * plane.buffer_size)
            frame.pts = pts
            frame.time_base = Fraction(1, 1000)
            output.mux(stream.encode(frame))
        output.mux(stream.encode())
    return path


def run_inspection(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_extract_point_and_interval_using_native_pts(variable_rate_video, tmp_path):
    output = tmp_path / "frames"
    result = run_inspection(variable_rate_video, "--at", "0", "--range", ".04", ".4", ".12",
                            "--output", output)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["sha256"] == hashlib.sha256(variable_rate_video.read_bytes()).hexdigest()
    assert report["decoded_frames"] == 4
    assert report["first_pts_seconds"] == pytest.approx(0.12)
    assert (report["width"], report["height"]) == (32, 32)
    assert [s["frame_index"] for s in report["samples"]] == [0, 1, 2, 2, 3]
    assert [s["pts"] for s in report["samples"]] == [120, 200, 480, 480, 520]
    assert [s["time_base"] for s in report["samples"]] == ["1/1000"] * 5
    assert len(list(output.glob("*.png"))) == 4
    for sample in report["samples"]:
        with Image.open(sample["image"]) as image:
            assert image.size == (32, 32)
    assert report["unmatched_request_seconds"] == []


def test_unmatched_request_is_reported(variable_rate_video, tmp_path):
    result = run_inspection(variable_rate_video, "--at", "5", "--output", tmp_path / "frames")
    assert result.returncode == 1
    assert json.loads(result.stdout)["unmatched_request_seconds"] == [5.0]


def test_invalid_interval_fails(variable_rate_video):
    result = run_inspection(variable_rate_video, "--range", "0", "1", "0")
    assert result.returncode != 0
    assert "STEP > 0" in result.stderr


def test_nonvideo_file_fails(tmp_path):
    path = tmp_path / "invalid.mp4"
    path.write_bytes(b"not a video")
    result = run_inspection(path)
    assert result.returncode != 0
    assert "Video inspection failed" in result.stderr
