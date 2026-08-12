from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from feature_config import MODEL_CLINICAL, MODEL_LIFESTYLE, TARGETS
from metrics import binary_classification_metrics, sigmoid

ROOT = Path("data/processed/final/ml_ready")
MODEL_ROOT = Path("data/processed/final/ml_models")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def stratified_split_indices(y: np.ndarray, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    y = y.astype(int)
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    rng = np.random.default_rng(seed)

    def sample(idx: np.ndarray) -> np.ndarray:
        n = int(round(len(idx) * frac))
        if n <= 0:
            return np.array([], dtype=int)
        if n >= len(idx):
            return idx
        return rng.choice(idx, size=n, replace=False)

    val_0 = sample(idx_0)
    val_1 = sample(idx_1)
    val_idx = np.concatenate([val_0, val_1])
    train_mask = np.ones(len(y), dtype=bool)
    train_mask[val_idx] = False
    train_idx = np.where(train_mask)[0]
    return train_idx, val_idx


def best_threshold(y_true: np.ndarray, y_score: np.ndarray, thresholds: list[float]) -> tuple[float, dict]:
    best_t = thresholds[0]
    best_payload: dict | None = None
    for t in thresholds:
        payload = binary_classification_metrics(y_true, y_score, threshold=t)
        if best_payload is None:
            best_payload = payload
            best_t = t
            continue
        if payload["f1"] != payload["f1"]:
            continue
        if best_payload["f1"] != best_payload["f1"]:
            best_payload = payload
            best_t = t
            continue
        if payload["f1"] > best_payload["f1"]:
            best_payload = payload
            best_t = t
            continue
        if payload["f1"] == best_payload["f1"] and payload["recall"] > best_payload["recall"]:
            best_payload = payload
            best_t = t
    if best_payload is None:
        best_payload = binary_classification_metrics(y_true, y_score, threshold=best_t)
    return float(best_t), best_payload


def train_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    reg_lambda: float,
    lr: float,
    max_steps: int,
    tol: float,
) -> tuple[np.ndarray, float, dict]:
    n = x.shape[0]
    x_aug = np.concatenate([x, np.ones((n, 1), dtype=float)], axis=1)
    d = x_aug.shape[1]

    w = np.zeros(d, dtype=float)
    pos = np.sum(y == 1)
    neg = n - pos
    pos_weight = (neg / pos) if pos > 0 else 1.0
    sw = np.where(y == 1, pos_weight, 1.0).astype(float)

    prev_loss = None
    for step in range(max_steps):
        logits = x_aug @ w
        p = sigmoid(logits)
        eps = 1e-12
        loss = -np.mean(sw * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))) + 0.5 * reg_lambda * float(w @ w)
        if prev_loss is not None and abs(prev_loss - loss) < tol:
            break
        prev_loss = loss
        grad = (x_aug.T @ (sw * (p - y))) / n + reg_lambda * w
        w = w - lr * grad

    weights = w[:-1]
    bias = float(w[-1])
    info = {"final_loss": float(prev_loss) if prev_loss is not None else float("nan")}
    return weights, bias, info


def load_split(model: str, target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = ROOT / model / target
    train = np.load(d / "train.npz")
    test = np.load(d / "test.npz")
    return train["x"].astype(float), train["y"].astype(int), test["x"].astype(float), test["y"].astype(int)


def save_model(model: str, target: str, weights: np.ndarray, bias: float, meta: dict) -> None:
    out = MODEL_ROOT / model / target
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "weights.npz", w=weights, b=np.array([bias], dtype=float))
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    reg_lambda = 1e-3
    lr = 0.05
    max_steps = 4000
    tol = 1e-8
    val_fraction = 0.2
    threshold_grid = [round(float(t), 2) for t in np.linspace(0.05, 0.95, 19)]

    for model in [MODEL_LIFESTYLE, MODEL_CLINICAL]:
        for target in TARGETS:
            logger.info("train model=%s target=%s", model, target)
            x_train_full, y_train_full, _, _ = load_split(model, target)
            seed = 42
            train_idx, val_idx = stratified_split_indices(y_train_full, frac=val_fraction, seed=seed)
            x_train = x_train_full[train_idx]
            y_train = y_train_full[train_idx]
            x_val = x_train_full[val_idx]
            y_val = y_train_full[val_idx]

            weights, bias, info = train_logistic_regression(
                x_train,
                y_train,
                reg_lambda=reg_lambda,
                lr=lr,
                max_steps=max_steps,
                tol=tol,
            )

            y_val_score = sigmoid(x_val @ weights + bias)
            best_t, val_metrics = best_threshold(y_val, y_val_score, thresholds=threshold_grid)

            meta = {
                "model": model,
                "target": target,
                "algorithm": "logistic_regression_gd",
                "reg_lambda": float(reg_lambda),
                "lr": float(lr),
                "max_steps": int(max_steps),
                "tol": float(tol),
                "training_info": info,
                "threshold_selection": {
                    "strategy": "max_f1_on_validation",
                    "val_fraction": float(val_fraction),
                    "threshold_grid": threshold_grid,
                    "best_threshold": float(best_t),
                    "val_metrics_at_best_threshold": val_metrics,
                },
            }
            save_model(model, target, weights, bias, meta)

    logger.info("done")


if __name__ == "__main__":
    main()

