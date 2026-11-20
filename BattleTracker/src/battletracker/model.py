"""Reproducible clip classification for Phase 5 offline experiments."""
import av
import cv2
import numpy as np

LABELS = ("NON_RESOURCE", "USE_MEDICINE", "SUMMON_PET")
FRAME_COUNT = 12
WIDTH, HEIGHT = 96, 48


def clip_sequence(path, start_ms, end_ms):
    """Read equally spaced source-time frames as float32 [12, 3, 32, 72]."""
    if not 0 <= start_ms < end_ms:
        raise ValueError("invalid clip interval")
    positions = np.linspace(start_ms, end_ms, FRAME_COUNT, endpoint=False)
    samples = []
    with av.open(str(path)) as video:
        stream = video.streams.video[0]
        origin = next(frame.pts * frame.time_base for frame in video.decode(stream)
                      if frame.pts is not None and frame.time_base is not None)
        video.seek(int((origin + start_ms / 1000) / stream.time_base), stream=stream)
        for frame in video.decode(stream):
            if frame.pts is None or frame.time_base is None or frame.is_corrupt:
                continue
            time = float(frame.pts * frame.time_base - origin) * 1000
            if time >= end_ms:
                break
            if time < positions[len(samples)]:
                continue
            # Keep the battle field and exclude the round header and replay controls.
            rgb = frame.reformat(width=WIDTH, height=HEIGHT, format="rgb24").to_ndarray()
            field = rgb[7:39, 12:84].astype(np.float32) / 255
            while len(samples) < FRAME_COUNT and time >= positions[len(samples)]:
                samples.append(field.transpose(2, 0, 1))
            if len(samples) == FRAME_COUNT:
                break
    if len(samples) < FRAME_COUNT:
        raise ValueError(f"clip has insufficient timestamp coverage: {start_ms}-{end_ms}")
    return np.stack(samples)


def features_from_sequence(frames):
    """Apply the original 12x6 spatial and three-bin temporal pooling."""
    frames = np.asarray(frames, dtype=np.float32)
    if frames.shape != (FRAME_COUNT, 3, 32, 72):
        raise ValueError("expected sequence shape [12, 3, 32, 72]")
    sequence = np.stack([
        cv2.resize(frame.transpose(1, 2, 0), (12, 6), interpolation=cv2.INTER_AREA)
        for frame in frames
    ])
    pooled = [part.mean(axis=0) for part in np.array_split(sequence, 3)]
    motion = [part.mean(axis=0) for part in np.array_split(np.abs(np.diff(sequence, axis=0)), 3)]
    return np.concatenate([item.ravel() for item in pooled + motion]).astype(np.float32)


def clip_features(path, start_ms, end_ms):
    """Use equally spaced source timestamps and pool appearance and motion."""
    return features_from_sequence(clip_sequence(path, start_ms, end_ms))


def softmax(logits):
    exponent = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exponent / exponent.sum(axis=1, keepdims=True)


def fit(features, targets, *, seed, steps, rate, regularization):
    center = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 0.05)
    normalized = (features - center) / scale
    rng = np.random.default_rng(seed)
    weight = rng.normal(0, 0.001, (features.shape[1], len(LABELS))).astype(np.float32)
    bias = np.zeros(len(LABELS), np.float32)
    expected = np.eye(len(LABELS), dtype=np.float32)[targets]
    for _ in range(steps):
        residual = (softmax(normalized @ weight + bias) - expected) / len(features)
        weight -= rate * (normalized.T @ residual + regularization * weight)
        bias -= rate * residual.sum(axis=0)
    # Fold preprocessing into the linear classifier for reproducible deployment.
    weight /= scale[:, None]
    return weight, bias - center @ weight


def metrics(expected, predicted):
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    np.add.at(confusion, (expected, predicted), 1)
    result = {}
    for i, label in enumerate(LABELS):
        tp = int(confusion[i, i])
        actual = int(confusion[i].sum())
        emitted = int(confusion[:, i].sum())
        result[label] = dict(support=actual, precision=tp / emitted if emitted else None,
                             recall=tp / actual if actual else None)
    return dict(accuracy=float((expected == predicted).mean()),
                confusion_matrix=confusion.tolist(), classes=result)

def export_and_compare(path, features, weight, bias, tolerance):
    from battletracker.runtime import bootstrap_native_runtime
    bootstrap_native_runtime()
    import onnx


    h = onnx.helper
    tensor = onnx.TensorProto.FLOAT
    graph = h.make_graph(
        [h.make_node("Gemm", ["features", "weight", "bias"], ["logits"]),
         h.make_node("Softmax", ["logits"], ["probabilities"], axis=1)],
        "battletracker_clip_classifier",
        [h.make_tensor_value_info("features", tensor, [None, weight.shape[0]])],
        [h.make_tensor_value_info("probabilities", tensor, [None, bias.size])],
        [onnx.numpy_helper.from_array(weight, name="weight"),
         onnx.numpy_helper.from_array(bias, name="bias")],
    )
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)], ir_version=8)

    onnx.save(model, str(path))
    expected = softmax(features @ weight + bias)
    serialized = onnx.load(str(path))

    assert [node.op_type for node in serialized.graph.node] == ["Gemm", "Softmax"]
    arrays = {item.name: onnx.numpy_helper.to_array(item) for item in serialized.graph.initializer}
    reference = softmax(features @ arrays["weight"] + arrays["bias"])
    np.testing.assert_allclose(reference, expected, atol=tolerance, rtol=0)
    parity = {
        "serialized_weights_max_probability_error": float(np.max(np.abs(reference - expected))),
        "runtime_status": "SKIPPED",
        "runtime_reason": None,
    }
    try:
        import onnxruntime as ort
    except ImportError as error:
        parity["runtime_reason"] = str(error)
        return parity
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    output = session.run(None, {"features": features})[0]
    np.testing.assert_allclose(output, expected, atol=tolerance, rtol=0)
    return dict(parity, runtime_status="PASS",
                runtime_max_probability_error=float(np.max(np.abs(output - expected))))


