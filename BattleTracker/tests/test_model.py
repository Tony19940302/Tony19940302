from pathlib import Path

import pytest


def test_smoke_ignores_unreviewed_labels_and_rejects_single_battle_validation(monkeypatch):
    pytest.importorskip("cv2")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from train_model import split_clips

    battle = dict(battle_id="battle", dataset_split="development")
    asset = dict(battle, asset_id="clean", role="clean_match_replay", sha256="clean",
                 status="available")
    rows = [dict(battle, sample_id=i, source_video_id="clean", event_type=event,
                 annotation_status=status) for i, (event, status) in enumerate([
        ("USE_MEDICINE", "reviewed"), ("SUMMON_PET", "reviewed"),
        ("PET_REVIVAL", "reviewed_hard_negative"), ("USE_MEDICINE", "candidate")
    ])]
    train, validation = split_clips(rows, [asset], smoke=True)
    assert [label for _, label in train] == [1, 2, 0]
    assert validation == []
    with pytest.raises(ValueError, match="separate groups"):
        split_clips(rows, [asset], smoke=False)
    with pytest.raises(ValueError, match="duplicate samples"):
        split_clips(rows + rows[:1], [asset], smoke=True)


def test_classifier_reproducibility():
    pytest.importorskip("cv2")
    import numpy as np
    from battletracker.model import fit, softmax

    features = np.eye(3, dtype=np.float32)
    targets = np.arange(3)
    params = dict(seed=42, steps=100, rate=0.1, regularization=0.01)
    weight, bias = fit(features, targets, **params)
    repeated = fit(features, targets, **params)
    np.testing.assert_array_equal(weight, repeated[0])
    np.testing.assert_array_equal(bias, repeated[1])
    np.testing.assert_array_equal(softmax(features @ weight + bias).argmax(axis=1), targets)
