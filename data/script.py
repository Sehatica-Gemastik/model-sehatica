from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

CYCLE = "2017-2018"

BASE_URL = (
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/"
    "2017/DataFiles/"
)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = PROCESSED_DIR / "ptm_dataset.csv"
REPORT_FILE = PROCESSED_DIR / "dataset_report.txt"

FILES = {
    "demographics": "DEMO_J.XPT",
    "body_measures": "BMX_J.XPT",
    "blood_pressure": "BPX_J.XPT",
    "diabetes": "DIQ_J.XPT",
    "blood_pressure_questionnaire": "BPQ_J.XPT",
    "medical_conditions": "MCQ_J.XPT",
    "physical_activity": "PAQ_J.XPT",
    "dietary_total_day1": "DR1TOT_J.XPT",
    "alcohol": "ALQ_J.XPT",
    "prescription_medications": "RXQ_RX_J.XPT",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

logger = logging.getLogger(__name__)

def download_file(filename: str) -> Path:
    destination = RAW_DIR / filename

    if destination.exists() and destination.stat().st_size > 0:
        logger.info("Already exists: %s", destination)
        return destination

    url = BASE_URL + filename

    logger.info("Downloading: %s", url)

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    destination.write_bytes(response.content)

    logger.info("Saved %.2f MB -> %s", destination.stat().st_size / (1024 * 1024), destination)

    return destination


def load_xpt(filename: str) -> pd.DataFrame:
    path = RAW_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing NHANES file: {path}"
        )

    logger.info("Reading %s", path.name)

    df = pd.read_sas(path, format="xport", encoding="utf-8")

    df.columns = [str(column).strip().upper() for column in df.columns]

    return df

def require_seqn(df: pd.DataFrame, name: str) -> None:
    if "SEQN" not in df.columns:
        raise ValueError(f"{name} does not contain SEQN.")


def select_existing(df: pd.DataFrame, columns: Iterable[str], dataset_name: str) -> pd.DataFrame:
    columns = list(columns)

    existing = [column for column in columns if column in df.columns]

    missing = [column for column in columns if column not in df.columns]

    if missing:
        logger.warning("%s: missing variables: %s", dataset_name, ", ".join(missing))

    if "SEQN" not in existing:
        raise ValueError(f"{dataset_name}: SEQN is missing.")

    return df[existing].copy()


def clean_special_codes(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    for column in result.columns:
        if column == "SEQN":
            continue

        if not pd.api.types.is_numeric_dtype(result[column]):
            continue

        result[column] = result[column].replace(
            {
                7: np.nan,
                9: np.nan,
                77: np.nan,
                99: np.nan,
                777: np.nan,
                999: np.nan,
                7777: np.nan,
                9999: np.nan,
            }
        )

    return result


def binary_yes_no(series: pd.Series) -> pd.Series:
    return series.map(
        {
            1: 1,
            2: 0,
        }
    )


def build_demographics(df: pd.DataFrame) -> pd.DataFrame:
    selected = select_existing(
        df,
        [
            "SEQN",
            "RIDAGEYR",
            "RIAGENDR",
            "RIDRETH3",
            "DMDEDUC2",
            "INDFMPIR",
        ],
        "Demographics",
    )

    selected = selected.rename(
        columns={
            "RIDAGEYR": "age",
            "RIAGENDR": "sex",
            "RIDRETH3": "race_ethnicity",
            "DMDEDUC2": "education",
            "INDFMPIR": "income_poverty_ratio",
        }
    )

    return selected


def build_body_measures(df: pd.DataFrame) -> pd.DataFrame:
    selected = select_existing(
        df,
        [
            "SEQN",
            "BMXWT",
            "BMXHT",
            "BMXBMI",
            "BMXWAIST",
        ],
        "Body Measures",
    )

    selected = selected.rename(
        columns={
            "BMXWT": "weight_kg",
            "BMXHT": "height_cm",
            "BMXBMI": "bmi",
            "BMXWAIST": "waist_cm",
        }
    )

    return selected


def build_blood_pressure(df: pd.DataFrame) -> pd.DataFrame:
    reading_columns = [
        "BPXSY1",
        "BPXSY2",
        "BPXSY3",
        "BPXSY4",
        "BPXDI1",
        "BPXDI2",
        "BPXDI3",
        "BPXDI4",
    ]

    selected = select_existing(
        df,
        ["SEQN", *reading_columns],
        "Blood Pressure",
    )

    systolic = [
        column
        for column in [
            "BPXSY1",
            "BPXSY2",
            "BPXSY3",
            "BPXSY4",
        ]
        if column in selected.columns
    ]

    diastolic = [
        column
        for column in [
            "BPXDI1",
            "BPXDI2",
            "BPXDI3",
            "BPXDI4",
        ]
        if column in selected.columns
    ]

    selected["systolic_bp"] = selected[systolic].mean(axis=1, skipna=True)

    selected["diastolic_bp"] = selected[diastolic].mean(axis=1, skipna=True)

    return selected[
        [
            "SEQN",
            "systolic_bp",
            "diastolic_bp",
        ]
    ]


def build_diabetes(df: pd.DataFrame) -> pd.DataFrame:
    selected = select_existing(
        df,
        [
            "SEQN",
            "DIQ010",
        ],
        "Diabetes",
    )

    selected["diabetes"] = binary_yes_no(
        selected["DIQ010"]
    )

    return selected[
        [
            "SEQN",
            "diabetes",
        ]
    ]


def build_hypertension(
    df: pd.DataFrame,
) -> pd.DataFrame:
    selected = select_existing(
        df,
        [
            "SEQN",
            "BPQ020",
        ],
        "Hypertension",
    )

    selected["hypertension"] = binary_yes_no(
        selected["BPQ020"]
    )

    return selected[
        [
            "SEQN",
            "hypertension",
        ]
    ]


def build_cardiovascular_labels(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "SEQN",
        "MCQ160B",
        "MCQ160C",
        "MCQ160D",
        "MCQ160E",
        "MCQ160F",
    ]

    selected = select_existing(
        df,
        columns,
        "Medical Conditions",
    )

    heart_columns = [
        column
        for column in [
            "MCQ160B",
            "MCQ160C",
            "MCQ160D",
            "MCQ160E",
        ]
        if column in selected.columns
    ]

    for column in heart_columns + (
        ["MCQ160F"]
        if "MCQ160F" in selected.columns
        else []
    ):
        selected[column] = binary_yes_no(selected[column])

    if heart_columns:
        selected["heart_disease"] = (selected[heart_columns].max(axis=1, skipna=True))
    else:
        selected["heart_disease"] = np.nan

    if "MCQ160F" in selected.columns:
        selected["stroke"] = selected["MCQ160F"]
    else:
        selected["stroke"] = np.nan

    return selected[
        [
            "SEQN",
            "heart_disease",
            "stroke",
        ]
    ]


def build_physical_activity(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "SEQN",

        # Work - vigorous
        "PAQ605",
        "PAQ610",
        "PAD615",

        # Work - moderate
        "PAQ620",
        "PAQ625",
        "PAD630",

        # Transportation
        "PAQ635",
        "PAQ640",
        "PAD645",

        # Recreation - vigorous
        "PAQ650",
        "PAQ655",
        "PAD660",

        # Recreation - moderate
        "PAQ665",
        "PAQ670",
        "PAD675",

        # Sedentary
        "PAD680",
    ]

    selected = select_existing(df, columns, "Physical Activity")

    selected = selected.rename(
        columns={
            "PAQ605": "vigorous_work",
            "PAQ610": "vigorous_work_days",
            "PAD615": "vigorous_work_minutes",

            "PAQ620": "moderate_work",
            "PAQ625": "moderate_work_days",
            "PAD630": "moderate_work_minutes",

            "PAQ635": "transport_walking_biking",
            "PAQ640": "transport_days",
            "PAD645": "transport_minutes",

            "PAQ650": "vigorous_recreation",
            "PAQ655": "vigorous_recreation_days",
            "PAD660": "vigorous_recreation_minutes",

            "PAQ665": "moderate_recreation",
            "PAQ670": "moderate_recreation_days",
            "PAD675": "moderate_recreation_minutes",

            "PAD680": "sedentary_minutes",
        }
    )

    yes_no_columns = [
        "vigorous_work",
        "moderate_work",
        "transport_walking_biking",
        "vigorous_recreation",
        "moderate_recreation",
    ]

    for column in yes_no_columns:
        if column in selected.columns:
            selected[column] = binary_yes_no(selected[column])

    return selected


def build_dietary(df: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        "SEQN",
        "DR1TKCAL",
        "DR1TPROT",
        "DR1TCARB",
        "DR1TSUGR",
        "DR1TTFAT",
        "DR1TSFAT",
        "DR1TSODI",
        "DR1TFIBE",
        "DR1TCHOL",
        "DR1TALCO",
    ]

    selected = select_existing(df, candidates, "Dietary Day 1")

    selected = selected.rename(
        columns={
            "DR1TKCAL": "calories_day1",
            "DR1TPROT": "protein_g_day1",
            "DR1TCARB": "carbohydrate_g_day1",
            "DR1TSUGR": "sugar_g_day1",
            "DR1TTFAT": "total_fat_g_day1",
            "DR1TSFAT": "saturated_fat_g_day1",
            "DR1TSODI": "sodium_mg_day1",
            "DR1TFIBE": "fiber_g_day1",
            "DR1TCHOL": "cholesterol_mg_day1",
            "DR1TALCO": "alcohol_g_day1",
        }
    )

    return selected


def build_alcohol(df: pd.DataFrame) -> pd.DataFrame:
    selected = select_existing(
        df,
        [
            "SEQN",
            "ALQ111",
            "ALQ121",
            "ALQ130",
            "ALQ142",
        ],
        "Alcohol",
    )

    selected = selected.rename(
        columns={
            "ALQ111": "alcohol_ever",
            "ALQ121": "alcohol_frequency",
            "ALQ130": "alcohol_drinks_per_day",
            "ALQ142": "alcohol_binge_frequency",
        }
    )

    if "alcohol_ever" in selected.columns:
        selected["alcohol_ever"] = binary_yes_no(selected["alcohol_ever"])

    return selected


def build_medication(df: pd.DataFrame) -> pd.DataFrame:
    require_seqn(df, "Prescription Medications")

    medication = (df[["SEQN"]].drop_duplicates().copy())

    medication["medication_any"] = 1

    return medication

def merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    result = frames[0].copy()

    for frame in frames[1:]:
        result = result.merge(
            frame,
            on="SEQN",
            how="left",
        )

    return result

def clean_final_dataset(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result = result.drop_duplicates(subset=["SEQN"])

    result["SEQN"] = pd.to_numeric(result["SEQN"], errors="coerce")

    numeric_columns = [column for column in result.columns if column != "SEQN"]

    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result.loc[(result["age"] < 0) | (result["age"] > 80), "age"] = np.nan

    if "bmi" in result.columns:
        result.loc[(result["bmi"] < 10) | (result["bmi"] > 80), "bmi"] = np.nan

    return result

def create_report(df: pd.DataFrame) -> None:
    lines: list[str] = []

    lines.append(f"NHANES PTM DATASET REPORT - {CYCLE}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Rows: {len(df):,}")
    lines.append(f"Columns: {len(df.columns):,}")
    lines.append("")

    lines.append("Columns:")
    for column in df.columns:
        lines.append(f"-> {column}")

    lines.append("")
    lines.append("Missingness:")
    missing = (df.isna().mean().sort_values(ascending=False) * 100)

    for column, percentage in missing.items():
        lines.append(f"-> {column}: {percentage:.2f}%")

    lines.append("")
    lines.append("Target distributions:")

    targets = [
        "diabetes",
        "hypertension",
        "heart_disease",
        "stroke",
    ]

    for target in targets:
        if target not in df.columns:
            continue

        lines.append("")
        lines.append(f"[{target}]")

        counts = df[target].value_counts(dropna=False)

        for value, count in counts.items():
            lines.append(f"{value}: {count:,}")

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")

def main() -> None:
    logger.info("Starting NHANES %s dataset build", CYCLE)
    paths = {}

    for key, filename in FILES.items():
        try:
            paths[key] = download_file(filename)
        except requests.HTTPError as exc:
            logger.error("Failed to download %s: %s", filename, exc)

    raw = {}

    for key, path in paths.items():
        try:
            raw[key] = load_xpt(
                path.name
            )
            logger.info("%s: %s rows x %s columns", key, len(raw[key]), len(raw[key].columns))
        except Exception as exc:
            logger.error("Failed to read %s: %s", key, exc)

    required = [
        "demographics",
        "body_measures",
        "blood_pressure",
        "diabetes",
        "blood_pressure_questionnaire",
        "medical_conditions",
        "physical_activity",
        "dietary_total_day1",
        "alcohol",
    ]

    missing_required = [name for name in required if name not in raw]

    if missing_required:
        raise RuntimeError("Required NHANES files could not be loaded: " + ", ".join(missing_required))

    frames = [
        build_demographics(
            raw["demographics"]
        ),
        build_body_measures(
            raw["body_measures"]
        ),
        build_blood_pressure(
            raw["blood_pressure"]
        ),
        build_diabetes(
            raw["diabetes"]
        ),
        build_hypertension(
            raw["blood_pressure_questionnaire"]
        ),
        build_cardiovascular_labels(
            raw["medical_conditions"]
        ),
        build_physical_activity(
            raw["physical_activity"]
        ),
        build_dietary(
            raw["dietary_total_day1"]
        ),
        build_alcohol(
            raw["alcohol"]
        ),
    ]

    if "prescription_medications" in raw:
        frames.append(
            build_medication(raw["prescription_medications"])
        )

    logger.info("Merging participant-level datasets...")

    dataset = merge_frames(frames)

    dataset = clean_final_dataset(dataset)

    target_columns = [
        "diabetes",
        "hypertension",
        "heart_disease",
        "stroke",
    ]

    dataset = dataset.dropna(subset=target_columns, how="all")

    dataset.to_csv(OUTPUT_FILE, index=False)

    create_report(dataset)

    logger.info("")
    logger.info("Dataset successfully created.")
    logger.info("Output: %s", OUTPUT_FILE)
    logger.info("Report: %s", REPORT_FILE)
    logger.info("Shape: %s", dataset.shape)

    logger.info("")
    logger.info("Targets:")

    for target in target_columns:
        if target not in dataset.columns:
            continue

        distribution = (dataset[target].value_counts(dropna=False).to_dict())

        logger.info("  %s -> %s", target, distribution)


if __name__ == "__main__":
    main()