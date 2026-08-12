from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from feature_config import (
    MODEL_CLINICAL,
    MODEL_LIFESTYLE,
    TARGETS,
    get_categorical,
    get_features,
    get_numeric,
    filter_known_labels,
)

INPUT_PATH = Path("data/processed/final/adult_clean.csv")
OUTPUT_ROOT = Path("data/processed/final/ml_ready")
DEFAULT_TEST_SIZE = 0.2
DEFAULT_SEED = 42

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_columns: list[str]


def stratified_split_indices(y: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = y.astype(int)
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]

    def sample_test(idx: np.ndarray) -> np.ndarray:
        n_test = int(round(len(idx) * test_size))
        if n_test <= 0:
            return np.array([], dtype=int)
        if n_test >= len(idx):
            return idx
        return rng.choice(idx, size=n_test, replace=False)

    test_0 = sample_test(idx_0)
    test_1 = sample_test(idx_1)
    test_idx = np.concatenate([test_0, test_1])
    train_mask = np.ones(len(y), dtype=bool)
    train_mask[test_idx] = False
    train_idx = np.where(train_mask)[0]
    return train_idx, test_idx


def fit_numeric(train_df: pd.DataFrame, numeric_features: list[str]) -> tuple[pd.Series, pd.Series, pd.Series]:
    train_part = train_df[numeric_features]
    medians = train_part.median(skipna=True)
    medians = medians.fillna(0)
    filled = train_part.fillna(medians)
    means = filled.mean(skipna=True)
    stds = filled.std(skipna=True, ddof=0).replace(0, 1)
    return medians, means, stds


def transform_numeric(
    df: pd.DataFrame,
    numeric_features: list[str],
    medians: pd.Series,
    means: pd.Series,
    stds: pd.Series,
) -> np.ndarray:
    x = df[numeric_features].copy()
    x = x.fillna(medians)
    for col in numeric_features:
        x[col] = (x[col] - means[col]) / stds[col]
    return x.to_numpy(dtype=float)


def fit_categorical(train_df: pd.DataFrame, categorical_features: list[str]) -> dict[str, list[str]]:
    fitted: dict[str, list[str]] = {}
    for col in categorical_features:
        values = train_df[col].dropna().unique().tolist()
        values = [str(v) for v in values]
        fitted[col] = sorted(values) + ["__missing__", "__unknown__"]
    return fitted


def transform_categorical(
    df: pd.DataFrame,
    categorical_features: list[str],
    fitted_categories: dict[str, list[str]],
) -> tuple[np.ndarray, list[str]]:
    feature_columns: list[str] = []
    blocks: list[np.ndarray] = []

    for col in categorical_features:
        cats = fitted_categories[col]
        col_values = df[col].astype("object").where(df[col].notna(), other="__missing__").astype(str)
        mat = np.zeros((len(df), len(cats)), dtype=float)
        cat_to_pos = {c: i for i, c in enumerate(cats)}
        for i, v in enumerate(col_values.to_numpy()):
            pos = cat_to_pos.get(v)
            if pos is None:
                pos = cat_to_pos["__unknown__"]
            mat[i, pos] = 1.0
        block_cols = [f"{col}__{c}" for c in cats]
        feature_columns.extend(block_cols)
        blocks.append(mat)

    if not blocks:
        return np.empty((len(df), 0), dtype=float), []
    return np.concatenate(blocks, axis=1), feature_columns


def build_features(
    df: pd.DataFrame,
    model: str,
    target: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    features = get_features(model, target)
    categorical = get_categorical(model, target)
    numeric = get_numeric(model, target)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"missing features for model={model} target={target}: {missing}")
    return df[features].copy(), categorical, numeric


def add_activity_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["work_total_minutes"] = (
        result[["vigorous_work_minutes", "moderate_work_minutes"]]
        .sum(axis=1, min_count=2)
    )
    result["recreation_total_minutes"] = (
        result[["vigorous_recreation_minutes", "moderate_recreation_minutes"]]
        .sum(axis=1, min_count=2)
    )
    result["vigorous_total_minutes"] = (
        result[["vigorous_work_minutes", "vigorous_recreation_minutes"]]
        .sum(axis=1, min_count=2)
    )
    result["moderate_total_minutes"] = (
        result[["moderate_work_minutes", "moderate_recreation_minutes"]]
        .sum(axis=1, min_count=2)
    )
    result["total_activity_minutes"] = (
        result[
            [
                "vigorous_work_minutes",
                "moderate_work_minutes",
                "transport_minutes",
                "vigorous_recreation_minutes",
                "moderate_recreation_minutes",
            ]
        ]
        .sum(axis=1, min_count=5)
    )

    result["total_activity_est_met"] = (
        result[
            [
                "vigorous_work_est_met",
                "moderate_work_est_met",
                "transport_walking_biking_est_met",
                "vigorous_recreation_est_met",
                "moderate_recreation_est_met",
            ]
        ]
        .sum(axis=1, min_count=5)
    )

    return result


def prepare_dataset(model: str, target: str, test_size: float, seed: int) -> tuple[DatasetSplit, dict]:
    df = pd.read_csv(INPUT_PATH)
    df = filter_known_labels(df, target)
    df = add_activity_aggregate_features(df)
    df = df.reset_index(drop=True)

    y = df[target].to_numpy()
    y = y.astype(float)
    y = y.astype(int)

    train_idx, test_idx = stratified_split_indices(y, test_size=test_size, seed=seed)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    x_train_raw, categorical, numeric = build_features(train_df, model, target)
    x_test_raw, _, _ = build_features(test_df, model, target)

    medians, means, stds = fit_numeric(x_train_raw, numeric)
    x_train_num = transform_numeric(x_train_raw, numeric, medians, means, stds)
    x_test_num = transform_numeric(x_test_raw, numeric, medians, means, stds)

    fitted_categories = fit_categorical(x_train_raw, categorical)
    x_train_cat, cat_feature_columns = transform_categorical(x_train_raw, categorical, fitted_categories)
    x_test_cat, _ = transform_categorical(x_test_raw, categorical, fitted_categories)

    x_train = np.concatenate([x_train_num, x_train_cat], axis=1)
    x_test = np.concatenate([x_test_num, x_test_cat], axis=1)

    num_feature_columns = numeric.copy()
    feature_columns = num_feature_columns + cat_feature_columns

    y_train = train_df[target].to_numpy().astype(int)
    y_test = test_df[target].to_numpy().astype(int)
    positive_train = int(y_train.sum())
    negative_train = int((y_train == 0).sum())
    positive_test = int(y_test.sum())
    negative_test = int((y_test == 0).sum())
    split = DatasetSplit(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        feature_columns=feature_columns,
    )

    preprocessing = {
        "split": {
            "test_size": float(test_size),
            "seed": int(seed),
            "train_samples": int(len(train_df)),
            "test_samples": int(len(test_df)),
            "positive_train": positive_train,
            "negative_train": negative_train,
            "positive_test": positive_test,
            "negative_test": negative_test,
        },
        "numeric_features": numeric,
        "categorical_features": categorical,
        "numeric_imputer": {
            "medians": {k: float(v) for k, v in medians.to_dict().items()},
        },
        "numeric_scaler": {
            "means": {k: float(v) for k, v in means.to_dict().items()},
            "stds": {k: float(v) for k, v in stds.to_dict().items()},
        },
        "categorical_one_hot": {
            "fitted_categories": {k: list(v) for k, v in fitted_categories.items()},
            "missing_bucket": "__missing__",
            "unknown_bucket": "__unknown__",
        },
    }

    return split, preprocessing


def save_split(
    model: str,
    target: str,
    split: DatasetSplit,
    preprocessing: dict,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "train.npz", x=split.x_train, y=split.y_train)
    np.savez(output_dir / "test.npz", x=split.x_test, y=split.y_test)
    (output_dir / "features.json").write_text(
        json.dumps({"feature_columns": split.feature_columns}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "preprocessing.json").write_text(
        json.dumps(preprocessing, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    test_size = float(DEFAULT_TEST_SIZE)
    seed = int(DEFAULT_SEED)

    for model in [MODEL_LIFESTYLE, MODEL_CLINICAL]:
        for target in TARGETS:
            logger.info("prepare model=%s target=%s", model, target)
            split, preprocessing = prepare_dataset(model=model, target=target, test_size=test_size, seed=seed)
            out_dir = OUTPUT_ROOT / model / target
            save_split(model, target, split, preprocessing, out_dir)


if __name__ == "__main__":
    main()

