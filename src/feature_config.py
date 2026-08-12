from __future__ import annotations

import pandas as pd

ID_COL = "SEQN"

TARGETS = ["diabetes", "hypertension", "heart_disease", "stroke"]

MODEL_LIFESTYLE = "lifestyle"
MODEL_CLINICAL = "clinical"
MODEL_TYPES = [MODEL_LIFESTYLE, MODEL_CLINICAL]

LABEL_POLICY = "known_only"

DEMOGRAPHIC = [
    "age", "sex", "race_ethnicity",
    "education", "income_poverty_ratio",
]

DIETARY = [
    "calories_day1", "protein_g_day1", "carbohydrate_g_day1",
    "sugar_g_day1", "total_fat_g_day1", "saturated_fat_g_day1",
    "sodium_mg_day1", "fiber_g_day1", "cholesterol_mg_day1",
    "alcohol_g_day1",
]

PHYSICAL_ACTIVITY = [
    "vigorous_work", "vigorous_work_days", "vigorous_work_minutes",
    "moderate_work", "moderate_work_days", "moderate_work_minutes",
    "transport_walking_biking", "transport_days", "transport_minutes",
    "vigorous_recreation", "vigorous_recreation_days", "vigorous_recreation_minutes",
    "moderate_recreation", "moderate_recreation_days", "moderate_recreation_minutes",
    "sedentary_minutes",
    "vigorous_work_est_met", "moderate_work_est_met",
    "transport_walking_biking_est_met", "vigorous_recreation_est_met",
    "moderate_recreation_est_met",
    "work_total_minutes",
    "recreation_total_minutes",
    "vigorous_total_minutes",
    "moderate_total_minutes",
    "total_activity_minutes",
    "total_activity_est_met",
]

ALCOHOL = [
    "alcohol_ever", "alcohol_frequency",
    "alcohol_drinks_per_day", "alcohol_binge_frequency",
]

MEDICATION: list[str] = []

ANTHROPOMETRIC = [
    "weight_kg", "height_cm", "bmi",
    "waist_cm", "systolic_bp", "diastolic_bp",
]

LIFESTYLE_FEATURES = DEMOGRAPHIC + DIETARY + PHYSICAL_ACTIVITY + ALCOHOL
CLINICAL_FEATURES = LIFESTYLE_FEATURES + ANTHROPOMETRIC

CATEGORICAL_FEATURES = [
    "sex", "race_ethnicity", "education",
    "vigorous_work", "moderate_work", "transport_walking_biking",
    "vigorous_recreation", "moderate_recreation",
    "alcohol_ever",
]

NUMERIC_FEATURES = [
    col for col in LIFESTYLE_FEATURES + ANTHROPOMETRIC
    if col not in CATEGORICAL_FEATURES
]

TARGET_LEAKAGE_EXCLUSIONS = {
    "diabetes": [],
    "hypertension": ["systolic_bp", "diastolic_bp"],
    "heart_disease": [],
    "stroke": [],
}

TARGET_LEAKAGE_NOTES = {
    "diabetes": "",
    "hypertension": "blood pressure is direct measurement, excluded from clinical model",
    "heart_disease": "",
    "stroke": "",
}


def get_base_features(model: str) -> list[str]:
    if model == MODEL_LIFESTYLE:
        return LIFESTYLE_FEATURES.copy()
    if model == MODEL_CLINICAL:
        return CLINICAL_FEATURES.copy()
    raise ValueError(f"unknown model type: {model}")


def get_features(model: str, target: str) -> list[str]:
    if target not in TARGETS:
        raise ValueError(f"unknown target: {target}")
    excluded = set(TARGET_LEAKAGE_EXCLUSIONS.get(target, []))
    return [col for col in get_base_features(model) if col not in excluded]


def get_categorical(model: str, target: str) -> list[str]:
    allowed = set(get_features(model, target))
    return [col for col in CATEGORICAL_FEATURES if col in allowed]


def get_numeric(model: str, target: str) -> list[str]:
    allowed = set(get_features(model, target))
    return [col for col in NUMERIC_FEATURES if col in allowed]


def filter_known_labels(df: pd.DataFrame, target: str) -> pd.DataFrame:
    return df.dropna(subset=[target])


def describe_config() -> str:
    lines = [
        "SEHATICA FEATURE CONFIG",
        "=" * 60,
        "",
        f"label policy: {LABEL_POLICY}",
        "baseline uses known labels only, no recovery, no NaN -> 0",
        "",
        "models:",
        f"  {MODEL_LIFESTYLE}: demographic + diet + activity + alcohol",
        f"  {MODEL_CLINICAL}: lifestyle + anthropometric",
        "",
        f"lifestyle features: {len(LIFESTYLE_FEATURES)}",
        f"clinical features: {len(CLINICAL_FEATURES)}",
        "",
        "leakage exclusions per target:",
    ]

    for target in TARGETS:
        excluded = TARGET_LEAKAGE_EXCLUSIONS.get(target, [])
        note = TARGET_LEAKAGE_NOTES.get(target, "")
        lifestyle_n = len(get_features(MODEL_LIFESTYLE, target))
        clinical_n = len(get_features(MODEL_CLINICAL, target))
        lines.append(f"  {target}:")
        lines.append(f"    excluded: {excluded or 'none'}")
        lines.append(f"    note: {note}")
        lines.append(f"    lifestyle feature count: {lifestyle_n}")
        lines.append(f"    clinical feature count: {clinical_n}")

    lines.extend([
        "",
        "feature groups:",
        f"  demographic ({len(DEMOGRAPHIC)}): {', '.join(DEMOGRAPHIC)}",
        f"  dietary ({len(DIETARY)}): {', '.join(DIETARY)}",
        f"  physical_activity ({len(PHYSICAL_ACTIVITY)}): {len(PHYSICAL_ACTIVITY)} columns",
        f"  alcohol ({len(ALCOHOL)}): {', '.join(ALCOHOL)}",
        f"  anthropometric ({len(ANTHROPOMETRIC)}): {', '.join(ANTHROPOMETRIC)}",
    ])

    return "\n".join(lines) + "\n"
