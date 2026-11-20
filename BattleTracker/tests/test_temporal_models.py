"""Guard pooling compatibility and dynamic-batch ONNX deployment."""
import numpy as np
import pytest

from battletracker.runtime import bootstrap_native_runtime

bootstrap_native_runtime()
pytest.importorskip("torch")

from battletracker.model import features_from_sequence
from battletracker.temporal_models import export_cnn, fit_cnn, predict


def test_pooling_retains_original_temporal_features():
    import cv2

    sequence = np.random.default_rng(42).random((12, 3, 32, 72), dtype=np.float32)
    original_grid = np.stack([
        cv2.resize(frame.transpose(1, 2, 0), (12, 6), interpolation=cv2.INTER_AREA)
        for frame in sequence
    ])
    pooled = [chunk.mean(axis=0) for chunk in np.array_split(original_grid, 3)]
    motion = [chunk.mean(axis=0) for chunk in np.array_split(np.abs(np.diff(original_grid, axis=0)), 3)]
    expected = np.concatenate([chunk.ravel() for chunk in pooled + motion])
    np.testing.assert_array_equal(features_from_sequence(sequence), expected)


def test_cnn_repeated_fit_and_dynamic_batch_export(tmp_path):
    sequence = np.random.default_rng(42).random((3, 12, 3, 32, 72), dtype=np.float32)
    labels = np.arange(3)
    first = fit_cnn(sequence, labels, seed=42, steps=3)
    repeated = fit_cnn(sequence, labels, seed=42, steps=3)
    np.testing.assert_array_equal(predict(first, sequence), predict(repeated, sequence))
    result = export_cnn(first, sequence, tmp_path / "temporal.onnx")
    assert result["runtime_status"] == "PASS"
    assert result["max_logit_error"] < 1e-5
