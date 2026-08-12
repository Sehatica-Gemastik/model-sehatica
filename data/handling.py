from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path("data/processed/ptm_dataset.csv")
OUTPUT_DIR = Path("data/processed/final")

TARGETS = ["diabetes", "hypertension", "heart_disease", "stroke"]

ACTIVITY_PAIRS = {
    "vigorous_work": ("vigorous_work_days", "vigorous_work_minutes"),
    "moderate_work": ("moderate_work_days", "moderate_work_minutes"),
    "transport_walking_biking": ("transport_days", "transport_minutes"),
    "vigorous_recreation": ("vigorous_recreation_days", "vigorous_recreation_minutes"),
    "moderate_recreation": ("moderate_recreation_days", "moderate_recreation_minutes"),
}

MET_VALUES = {
    "vigorous_work": 8.0,
    "moderate_work": 4.0,
    "transport_walking_biking": 4.0,
    "vigorous_recreation": 8.0,
    "moderate_recreation": 4.0,
}

DEMOGRAPHIC = [
    "age", "sex", "race_ethnicity",
    "education", "income_poverty_ratio",
]

ANTHROPOMETRIC = [
    "weight_kg", "height_cm", "bmi",
    "waist_cm", "systolic_bp", "diastolic_bp",
]

DIETARY = [
    "calories_day1", "protein_g_day1", "carbohydrate_g_day1",
    "sugar_g_day1", "total_fat_g_day1", "saturated_fat_g_day1",
    "sodium_mg_day1", "fiber_g_day1", "cholesterol_mg_day1",
    "alcohol_g_day1",
]

ALCOHOL = [
    "alcohol_ever", "alcohol_frequency",
    "alcohol_drinks_per_day", "alcohol_binge_frequency",
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("loaded %d rows x %d columns from %s", len(df), len(df.columns), path)
    return df


def filter_adults(df: pd.DataFrame) -> pd.DataFrame:
    result = df[df["age"] >= 18].copy()
    logger.info("adults filter: %d -> %d rows", len(df), len(result))
    return result


def fix_sas_artifacts(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    artifact_threshold = 1e-50
    for col in result.select_dtypes(include=[np.number]).columns:
        if col == "SEQN":
            continue
        mask = (result[col].abs() < artifact_threshold) & result[col].ne(0)
        if mask.any():
            result.loc[mask, col] = 0.0
            logger.info("fixed %d near-zero SAS artifacts in '%s'", int(mask.sum()), col)
    return result


def fill_activity_zeros(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for flag, (days_col, mins_col) in ACTIVITY_PAIRS.items():
        if flag not in result.columns:
            continue
        no_activity = result[flag] == 0
        for col in [days_col, mins_col]:
            if col in result.columns:
                result.loc[no_activity, col] = 0
    return result


def prepare_medication(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "medication_any" in result.columns:
        result["medication_any"] = result["medication_any"].astype("Int64")
    return result


def engineer_met_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    for flag, (days_col, mins_col) in ACTIVITY_PAIRS.items():
        if days_col not in result.columns or mins_col not in result.columns:
            continue
        met_col = f"{flag}_est_met"
        result[met_col] = MET_VALUES[flag] * result[days_col] * result[mins_col]

    return result


def drop_all_unknown_targets(df: pd.DataFrame) -> pd.DataFrame:
    existing = [t for t in TARGETS if t in df.columns]
    result = df.dropna(subset=existing, how="all")
    logger.info("drop all-unknown targets: %d -> %d rows", len(df), len(result))
    return result


def generate_report(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "SEHATICA PTM HANDLING REPORT",
        "=" * 60,
        "",
        f"rows: {len(df):,}",
        f"columns: {len(df.columns):,}",
        "",
        "columns:",
    ]

    for col in df.columns:
        lines.append(f"  {col}")

    lines.append("")
    lines.append("missingness:")
    missing = df.isna().mean().sort_values(ascending=False) * 100
    for col, pct in missing.items():
        if pct > 0:
            lines.append(f"  {col}: {pct:.2f}%")

    lines.append("")
    lines.append("target distributions:")

    for target in TARGETS:
        if target not in df.columns:
            continue
        positive = int((df[target] == 1).sum())
        negative = int((df[target] == 0).sum())
        unknown = int(df[target].isna().sum())
        known = positive + negative
        lines.append("")
        lines.append(f"[{target}]")
        lines.append(f"  positive: {positive:,}")
        lines.append(f"  negative: {negative:,}")
        lines.append(f"  unknown: {unknown:,}")
        lines.append(f"  known: {known:,}")
        if known > 0:
            lines.append(f"  prevalence: {positive / known * 100:.2f}%")

    lines.append("")
    lines.append("feature groups:")
    lines.append(f"demographic: {', '.join(DEMOGRAPHIC)}")
    lines.append(f"anthropometric: {', '.join(ANTHROPOMETRIC)}")
    lines.append(f"dietary: {', '.join(DIETARY)}")
    lines.append(f"alcohol: {', '.join(ALCOHOL)}")
    lines.append("physical_activity: [flags + days/minutes + met features]")
    lines.append("medication: medication_any")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("report saved to %s", path)


def main() -> None:
    logger.info("starting handling pipeline")

    df = load_dataset(INPUT_PATH)
    df = filter_adults(df)
    df = fix_sas_artifacts(df)
    df = fill_activity_zeros(df)
    df = prepare_medication(df)
    df = engineer_met_features(df)
    df = drop_all_unknown_targets(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "adult_clean.csv"
    df.to_csv(output_path, index=False)
    logger.info("saved %s (%d rows x %d cols)", output_path, len(df), len(df.columns))

    generate_report(df, OUTPUT_DIR / "report.txt")
    logger.info("done")


if __name__ == "__main__":
    main()
