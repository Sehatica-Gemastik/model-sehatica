from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("data/processed/final/ml_ready")
MODELS = ["lifestyle", "clinical"]
TARGETS = ["diabetes", "hypertension", "heart_disease", "stroke"]


def assert_finite(name: str, arr: np.ndarray) -> None:
    if not np.isfinite(arr).all():
        raise AssertionError(f"{name} contains NaN/inf")


def validate_one(model: str, target: str) -> None:
    d = ROOT / model / target
    features = json.loads((d / "features.json").read_text(encoding="utf-8"))
    prep = json.loads((d / "preprocessing.json").read_text(encoding="utf-8"))

    train = np.load(d / "train.npz")
    test = np.load(d / "test.npz")

    x_tr = train["x"]
    y_tr = train["y"]
    x_te = test["x"]
    y_te = test["y"]

    feature_columns = features["feature_columns"]
    assert x_tr.shape[1] == len(feature_columns)
    assert x_te.shape[1] == x_tr.shape[1]
    assert x_tr.shape[0] == y_tr.shape[0]
    assert x_te.shape[0] == y_te.shape[0]

    assert set(np.unique(y_tr)).issubset({0, 1})
    assert set(np.unique(y_te)).issubset({0, 1})

    assert_finite("x_train", x_tr)
    assert_finite("x_test", x_te)

    split = prep["split"]
    assert split["train_samples"] == int(x_tr.shape[0])
    assert split["test_samples"] == int(x_te.shape[0])
    assert split["positive_train"] == int(y_tr.sum())
    assert split["positive_test"] == int(y_te.sum())

    one_hot = prep["categorical_one_hot"]
    assert one_hot["missing_bucket"] == "__missing__"
    assert one_hot["unknown_bucket"] == "__unknown__"
    for _, cats in one_hot["fitted_categories"].items():
        assert "__missing__" in cats
        assert "__unknown__" in cats


def main() -> None:
    for model in MODELS:
        for target in TARGETS:
            validate_one(model, target)
    print("ok")


if __name__ == "__main__":
    main()

