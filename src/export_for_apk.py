from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from feature_config import MODEL_CLINICAL, MODEL_LIFESTYLE, TARGETS

READY_ROOT = Path("data/processed/final/ml_ready")
MODEL_ROOT = Path("data/processed/final/ml_models")
OUT_ROOT = Path("data/processed/final/android_models")

MODELS = [MODEL_LIFESTYLE, MODEL_CLINICAL]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def export_one(feature_set: str, target: str) -> None:
    ready_dir = READY_ROOT / feature_set / target
    model_dir = MODEL_ROOT / feature_set / target

    features = load_json(ready_dir / "features.json")
    preprocessing = load_json(ready_dir / "preprocessing.json")
    meta = load_json(model_dir / "meta.json")

    weights_npz = np.load(model_dir / "weights.npz")
    w = weights_npz["w"].astype(float).tolist()
    b = float(weights_npz["b"][0])

    threshold = float(meta["threshold_selection"]["best_threshold"])

    feature_columns = features["feature_columns"]
    weights_len = len(w)

    if weights_len != len(feature_columns):
        raise ValueError(
            f"feature_columns len != weights len for {feature_set}/{target}: {len(feature_columns)} vs {weights_len}"
        )

    cat_encoded = {
        col: {} for col in preprocessing["categorical_features"]
    }
    for idx, name in enumerate(feature_columns):
        if "__" not in name:
            continue
        col, cat = name.split("__", 1)
        if col in cat_encoded:
            cat_encoded[col][cat] = idx

    payload = {
        "algorithm": "logistic_regression_gd",
        "feature_set": feature_set,
        "target": target,
        "threshold": threshold,
        "bias": b,
        "weights": w,
        "feature_columns": feature_columns,
        "preprocessing": {
            "numeric_features": preprocessing["numeric_features"],
            "numeric_imputer_medians": preprocessing["numeric_imputer"]["medians"],
            "numeric_scaler": preprocessing["numeric_scaler"],
            "categorical_features": preprocessing["categorical_features"],
            "categorical_one_hot": preprocessing["categorical_one_hot"],
            "categorical_feature_to_category_to_feature_index": cat_encoded,
            "buckets": {
                "missing_bucket": preprocessing["categorical_one_hot"]["missing_bucket"],
                "unknown_bucket": preprocessing["categorical_one_hot"]["unknown_bucket"],
            },
        },
    }

    out_dir = OUT_ROOT / feature_set / target
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    for feature_set in MODELS:
        for target in TARGETS:
            export_one(feature_set=feature_set, target=target)


if __name__ == "__main__":
    main()

