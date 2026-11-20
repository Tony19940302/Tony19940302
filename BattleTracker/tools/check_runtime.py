"""Verify native runtime and exported linear model probabilities on CPU."""
import json
from pathlib import Path
import sys

from battletracker.runtime import bootstrap_native_runtime


def main():
    runtime = bootstrap_native_runtime()
    import numpy as np
    import onnx
    import onnxruntime as ort

    root = Path(__file__).resolve().parents[1]
    directory = root / "outputs/phase-05"
    with np.load(directory / "model.npz") as model:
        weight, bias = model["weight"], model["bias"]
    features = np.random.default_rng(42).normal(size=(17, weight.shape[0])).astype(np.float32)
    logits = features @ weight + bias
    exponent = np.exp(logits - logits.max(axis=1, keepdims=True))
    expected = exponent / exponent.sum(axis=1, keepdims=True)
    path = directory / "model.onnx"
    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"features": features})[0]
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)
    report = dict(status="PASS", python=sys.version, runtime=runtime,
                  onnx_version=onnx.__version__, onnxruntime_version=ort.__version__,
                  providers=session.get_providers(), seed=42, input_shape=list(features.shape),
                  onnx_checker="PASS", atol=1e-5,
                  max_probability_error=float(np.abs(expected - actual).max()))
    (directory / "runtime-check.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
