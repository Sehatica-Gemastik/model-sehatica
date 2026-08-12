from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/processed/final/adult_clean.csv")
RAW_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/processed/final/target_audit.txt")

HEART_VARS = ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E"]
MCQ_AGE_THRESHOLD = 20

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TargetAuditResult:
    name: str
    known_positive: int = 0
    known_negative: int = 0
    unknown: int = 0
    recoverable: int = 0
    not_recoverable: int = 0
    source: str = ""
    recovery_rules: list[str] = field(default_factory=list)
    unknown_breakdown: list[str] = field(default_factory=list)
    training_recommendation: str = ""

    @property
    def known(self) -> int:
        return self.known_positive + self.known_negative


def load_adult_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("loaded %d rows from %s", len(df), path)
    return df


def load_raw(filename: str) -> pd.DataFrame:
    path = RAW_DIR / filename
    df = pd.read_sas(path, format="xport", encoding="utf-8")
    df.columns = [str(col).strip().upper() for col in df.columns]
    return df


def count_label(df: pd.DataFrame, col: str) -> tuple[int, int, int]:
    positive = int((df[col] == 1).sum())
    negative = int((df[col] == 0).sum())
    unknown = int(df[col].isna().sum())
    return positive, negative, unknown


def audit_diabetes(adult: pd.DataFrame, diq: pd.DataFrame) -> TargetAuditResult:
    merged = adult.merge(diq[["SEQN", "DIQ010", "DIQ050", "DIQ070"]], on="SEQN", how="left")
    unknown = merged[merged["diabetes"].isna()].copy()
    diq010 = unknown["DIQ010"]
    borderline = int((diq010 == 3).sum())
    dont_know = int((diq010 == 9).sum())
    recoverable = int((unknown["DIQ070"] == 1).sum())
    not_recoverable = len(unknown) - recoverable

    pos, neg, unk = count_label(adult, "diabetes")
    return TargetAuditResult(
        name="diabetes",
        known_positive=pos,
        known_negative=neg,
        unknown=unk,
        recoverable=recoverable,
        not_recoverable=not_recoverable,
        source="DIQ_J (DIQ010 primary, DIQ070 supplementary)",
        recovery_rules=[
            "DIQ010=1 -> positive, DIQ010=2 -> negative",
            "DIQ010=3 (borderline) -> unknown unless DIQ070=1 -> positive",
            "DIQ010=9 (don't know) -> not recoverable",
        ],
        unknown_breakdown=[
            f"borderline (DIQ010=3): {borderline}",
            f"don't know (DIQ010=9): {dont_know}",
            f"recoverable via DIQ070=1 (taking diabetes pills): {recoverable}",
            f"not recoverable: {not_recoverable}",
        ],
        training_recommendation=(
            "train on known labels only; optional deterministic relabel "
            "for borderline cases with DIQ070=1 (+33 samples)"
        ),
    )


def audit_hypertension(adult: pd.DataFrame, bpq: pd.DataFrame) -> TargetAuditResult:
    merged = adult.merge(bpq[["SEQN", "BPQ020", "BPQ030", "BPQ040A"]], on="SEQN", how="left")
    unknown = merged[merged["hypertension"].isna()].copy()

    dont_know = int((unknown["BPQ020"] == 9).sum())
    has_bp = unknown["systolic_bp"].notna() & unknown["diastolic_bp"].notna()
    elevated = has_bp & (
        (unknown["systolic_bp"] >= 140) | (unknown["diastolic_bp"] >= 90)
    )
    recoverable_bp = int(elevated.sum())
    recoverable_meds = int((unknown["BPQ030"] == 1).sum())
    recoverable = recoverable_bp + recoverable_meds
    not_recoverable = len(unknown) - recoverable

    pos, neg, unk = count_label(adult, "hypertension")
    return TargetAuditResult(
        name="hypertension",
        known_positive=pos,
        known_negative=neg,
        unknown=unk,
        recoverable=recoverable,
        not_recoverable=not_recoverable,
        source="BPQ_J (BPQ020 primary) + BPX_J measured BP (supplementary)",
        recovery_rules=[
            "BPQ020=1 -> positive, BPQ020=2 -> negative",
            "BPQ020=9 (don't know) -> unknown",
            "BPQ030=1 (taking HBP meds) -> positive if BPQ020 missing",
            "measured BP: systolic>=140 OR diastolic>=90 -> positive (clinical rule)",
        ],
        unknown_breakdown=[
            f"don't know (BPQ020=9): {dont_know}",
            f"recoverable via measured BP elevation: {recoverable_bp}",
            f"recoverable via BPQ030=1 (HBP medication): {recoverable_meds}",
            f"not recoverable: {not_recoverable}",
        ],
        training_recommendation=(
            "train on known labels; measured BP recovery uses different "
            "definition than self-reported, treat as optional experiment"
        ),
    )


def audit_heart_disease(adult: pd.DataFrame, mcq: pd.DataFrame) -> TargetAuditResult:
    cols = ["SEQN", *HEART_VARS]
    merged = adult.merge(mcq[cols], on="SEQN", how="left")
    unknown = merged[merged["heart_disease"].isna()].copy()

    age_skip = int((unknown["age"] < MCQ_AGE_THRESHOLD).sum())
    age_eligible = unknown[unknown["age"] >= MCQ_AGE_THRESHOLD]
    dont_know = int((age_eligible[HEART_VARS] == 9).any(axis=1).sum()) if len(age_eligible) else 0

    recoverable = 0
    not_recoverable = len(unknown)

    pos, neg, unk = count_label(adult, "heart_disease")
    return TargetAuditResult(
        name="heart_disease",
        known_positive=pos,
        known_negative=neg,
        unknown=unk,
        recoverable=recoverable,
        not_recoverable=not_recoverable,
        source="MCQ_J (MCQ160B-E, age >= 20 skip pattern)",
        recovery_rules=[
            "any MCQ160B-E=1 -> positive, all MCQ160B-E=2 -> negative",
            "age < 20: questions not asked (NHANES skip pattern)",
            "MCQ160*=9 (don't know) -> not recoverable",
        ],
        unknown_breakdown=[
            f"age < 20 (question not asked): {age_skip}",
            f"age >= 20 with don't know responses: {dont_know}",
            f"recoverable: {recoverable}",
            f"not recoverable: {not_recoverable}",
        ],
        training_recommendation=(
            "train on known labels only; unknown is mostly age 18-19 skip, "
            "do not impute as negative"
        ),
    )


def audit_stroke(adult: pd.DataFrame, mcq: pd.DataFrame) -> TargetAuditResult:
    merged = adult.merge(mcq[["SEQN", "MCQ160F"]], on="SEQN", how="left")
    unknown = merged[merged["stroke"].isna()].copy()

    age_skip = int((unknown["age"] < MCQ_AGE_THRESHOLD).sum())
    age_eligible = unknown[unknown["age"] >= MCQ_AGE_THRESHOLD]
    dont_know = int((age_eligible["MCQ160F"] == 9).sum())

    recoverable = 0
    not_recoverable = len(unknown)

    pos, neg, unk = count_label(adult, "stroke")
    return TargetAuditResult(
        name="stroke",
        known_positive=pos,
        known_negative=neg,
        unknown=unk,
        recoverable=recoverable,
        not_recoverable=not_recoverable,
        source="MCQ_J (MCQ160F, age >= 20 skip pattern)",
        recovery_rules=[
            "MCQ160F=1 -> positive, MCQ160F=2 -> negative",
            "age < 20: question not asked (NHANES skip pattern)",
            "MCQ160F=9 (don't know) -> not recoverable",
        ],
        unknown_breakdown=[
            f"age < 20 (question not asked): {age_skip}",
            f"age >= 20 with don't know (MCQ160F=9): {dont_know}",
            f"recoverable: {recoverable}",
            f"not recoverable: {not_recoverable}",
        ],
        training_recommendation=(
            "train on known labels only; unknown is mostly age 18-19 skip, "
            "do not impute as negative"
        ),
    )


def format_result(result: TargetAuditResult) -> list[str]:
    lines = [
        result.name,
        "-" * len(result.name),
        f"source: {result.source}",
        f"known: {result.known:,} (positive: {result.known_positive:,}, negative: {result.known_negative:,})",
        f"unknown: {result.unknown:,}",
        f"potentially recoverable: {result.recoverable:,}",
        f"not recoverable: {result.not_recoverable:,}",
        "",
        "recovery rules:",
    ]
    lines.extend(f"  {rule}" for rule in result.recovery_rules)
    lines.append("")
    lines.append("unknown breakdown:")
    lines.extend(f"  {item}" for item in result.unknown_breakdown)
    lines.append("")
    lines.append(f"training recommendation: {result.training_recommendation}")
    return lines


def generate_report(results: list[TargetAuditResult], path: Path, n_adults: int) -> None:
    lines = [
        "SEHATICA PTM TARGET AUDIT",
        "=" * 60,
        "",
        "principle: missing label != negative label",
        "NaN means unknown, not healthy",
        "",
    ]

    for result in results:
        lines.extend(format_result(result))
        lines.append("")

    lines.extend([
        "summary",
        "-" * 60,
        f"total adults: {n_adults:,}",
        "",
        "per-target training populations (known labels only):",
    ])

    for result in results:
        lines.append(f"  {result.name}: {result.known:,} samples")

    lines.extend([
        "",
        "next step: train/test split -> feature preprocessing -> model training",
        "each target may use a different training population",
    ])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("report saved to %s", path)


def main() -> None:
    logger.info("starting target audit")

    adult = load_adult_clean(INPUT_PATH)
    diq = load_raw("DIQ_J.XPT")
    bpq = load_raw("BPQ_J.XPT")
    mcq = load_raw("MCQ_J.XPT")

    results = [
        audit_diabetes(adult, diq),
        audit_hypertension(adult, bpq),
        audit_heart_disease(adult, mcq),
        audit_stroke(adult, mcq),
    ]

    generate_report(results, OUTPUT_PATH, len(adult))

    for result in results:
        logger.info(
            "%s: known=%d unknown=%d recoverable=%d not_recoverable=%d",
            result.name,
            result.known,
            result.unknown,
            result.recoverable,
            result.not_recoverable,
        )

    logger.info("done")


if __name__ == "__main__":
    main()