from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from feature_config import MODEL_CLINICAL, MODEL_LIFESTYLE, TARGETS
from metrics import average_precision_score, binary_classification_metrics, roc_auc_score, sigmoid

ROOT = Path("data/processed/final/ml_ready")
MODEL_ROOT = Path("data/processed/final/ml_models")
EVAL_ROOT = Path("data/processed/final/ml_eval")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_split(model: str, target: str) -> tuple[np.ndarray, np.ndarray]:
    d = ROOT / model / target
    test = np.load(d / "test.npz")
    return test["x"].astype(float), test["y"].astype(int)


def load_model(model: str, target: str) -> tuple[np.ndarray, float, dict]:
    d = MODEL_ROOT / model / target
    w = np.load(d / "weights.npz")
    weights = w["w"].astype(float)
    bias = float(w["b"][0])
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    return weights, bias, meta


def predict(weights: np.ndarray, bias: float, x: np.ndarray) -> np.ndarray:
    return sigmoid(x @ weights + bias)


def save_eval(model: str, target: str, payload: dict) -> None:
    out = EVAL_ROOT / model / target
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    results: dict[str, dict] = {}

    for model in [MODEL_LIFESTYLE, MODEL_CLINICAL]:
        for target in TARGETS:
            logger.info("eval model=%s target=%s", model, target)
            x_test, y_test = load_split(model, target)
            weights, bias, meta = load_model(model, target)
            y_score = predict(weights, bias, x_test)

            roc_auc = roc_auc_score(y_test, y_score)
            pr_auc = average_precision_score(y_test, y_score)
            threshold = float(meta.get("threshold_selection", {}).get("best_threshold", 0.5))
            cls = binary_classification_metrics(y_test, y_score, threshold=threshold)

            payload = {
                "model": model,
                "target": target,
                "threshold": float(threshold),
                "metrics": {
                    "roc_auc": roc_auc,
                    "pr_auc": pr_auc,
                    **cls,
                },
                "model_meta": meta,
            }

            save_eval(model, target, payload)
            results[f"{model}/{target}"] = payload

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    (EVAL_ROOT / "all_metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("done")


if __name__ == "__main__":
    main()

