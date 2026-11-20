"""Small CPU video classifier and a nearest-centroid feature comparator."""
from pathlib import Path

import numpy as np

from battletracker.runtime import bootstrap_native_runtime

bootstrap_native_runtime()

import torch
from torch import nn

from battletracker.model import FRAME_COUNT, clip_sequence




class TinyTemporalCNN(nn.Module):
    """Shared per-frame CNN followed by three temporal bins and motion pooling."""

    def __init__(self):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8, 12, 3, stride=2, padding=1), nn.ReLU(),
            nn.AvgPool2d((4, 6)),
        )
        self.classifier = nn.Linear(72 * 4, 3)

    def forward(self, sequence):
        frames = sequence.reshape(-1, 3, 32, 72)
        encoded = self.spatial(frames).reshape(-1, FRAME_COUNT, 72)
        temporal = encoded.reshape(-1, 3, 4, 72).mean(dim=2).flatten(start_dim=1)
        motion = torch.abs(encoded[:, 1:] - encoded[:, :-1]).mean(dim=1)
        return self.classifier(torch.cat((temporal, motion), dim=1))


def fit_cnn(sequences, targets, *, seed=42, steps=300, rate=0.01, regularization=0.01):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = TinyTemporalCNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=rate, weight_decay=regularization)
    inputs = torch.from_numpy(np.asarray(sequences, dtype=np.float32))
    expected = torch.from_numpy(np.asarray(targets, dtype=np.int64))
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(inputs), expected)
        loss.backward()
        optimizer.step()
    return model.eval()


def predict(model, sequences):
    with torch.no_grad():
        inputs = torch.from_numpy(np.asarray(sequences, dtype=np.float32))
        return model(inputs).softmax(dim=1).numpy()


def export_cnn(model, sequences, path, tolerance=1e-5):
    import onnxruntime as ort

    path = Path(path)
    sample = torch.from_numpy(sequences[:1])
    torch.onnx.export(model, (sample,), str(path), dynamo=False, opset_version=13,
                      input_names=["sequences"], output_names=["logits"],
                      dynamic_axes={"sequences": {0: "batch"}, "logits": {0: "batch"}})
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    with torch.no_grad():
        expected = model(torch.from_numpy(sequences)).numpy()
    actual = session.run(None, {"sequences": sequences})[0]
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0)
    return {"runtime_status": "PASS", "max_logit_error": float(np.max(np.abs(actual - expected))),
            "onnxruntime_version": ort.__version__, "torch_version": torch.__version__}


def fit_centroids(features, targets):
    center = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 0.05)
    normalized = (features - center) / scale
    centroids = np.stack([normalized[targets == label].mean(axis=0) for label in range(3)])
    return center, scale, centroids


def predict_centroids(model, features):
    center, scale, centroids = model
    normalized = (features - center) / scale
    distance = ((normalized[:, None] - centroids[None]) ** 2).mean(axis=2)
    exponent = np.exp(-distance - (-distance).max(axis=1, keepdims=True))
    return exponent / exponent.sum(axis=1, keepdims=True)
