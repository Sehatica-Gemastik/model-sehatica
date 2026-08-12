from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    scores = y_score.astype(float)
    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    if pos == 0 or neg == 0:
        return float("nan")

    n = len(scores)
    order = np.argsort(scores)
    ranks = np.empty(n, dtype=float)
    sorted_scores = scores[order]

    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        rank_start = i + 1
        rank_end = j
        avg_rank = (rank_start + rank_end) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    rank_sum_pos = float(np.sum(ranks[y_true == 1]))
    auc = (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    y_score = y_score.astype(float)
    n_pos = np.sum(y_true == 1)
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / (tp + fp)
    recall = tp / n_pos
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    delta = recall - recall_prev
    ap = float(np.sum(precision * delta))
    return ap


def binary_classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if tp + fp > 0 else float("nan")
    recall = tp / (tp + fn) if tp + fn > 0 else float("nan")
    specificity = tn / (tn + fp) if tn + fp > 0 else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision == precision
        and recall == recall
        and (precision + recall) > 0
        else float("nan")
    )
    balanced_accuracy = (
        (recall + specificity) / 2.0
        if recall == recall
        and specificity == specificity
        else float("nan")
    )
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "f1": float(f1),
    }

