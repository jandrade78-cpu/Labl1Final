import pandas as pd
import streamlit as st
import config as C
from features import prepare


@st.cache_data(show_spinner=False)
def load_csv(file_or_path) -> pd.DataFrame:
    return pd.read_csv(file_or_path)


def validate_schema(df: pd.DataFrame) -> dict:
    errors, warnings = [], []
    if df.empty:
        errors.append("The file has no rows.")
    missing = [c for c in C.RAW_REQUIRED if c not in df.columns]
    if missing:
        errors.append("Missing required columns: " + ", ".join(missing))
    for c in ["Queue_Length", "SLA_Days"]:
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            errors.append(f"Column '{c}' must be numeric.")
    leak = [c for c in C.LEAKAGE_COLUMNS if c in df.columns]
    if leak:
        warnings.append("Post-outcome columns present (" + ", ".join(leak) +
                        "). They are never used as model inputs (leakage prevention).")
    return {"ok": not errors, "errors": errors, "warnings": warnings}


def quality_summary(df: pd.DataFrame) -> dict:
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0].rename("Null count").to_frame()
    nulls["Null %"] = (nulls["Null count"] / max(len(df), 1) * 100).round(2)
    return {"rows": len(df), "cols": df.shape[1], "duplicates": int(df.duplicated().sum()), "nulls": nulls}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, add engineered features (+ SLA_Breach label where Processing_Days exists).
    Missing feature values are imputed inside the saved pipeline, so rows are not discarded."""
    return prepare(df.drop_duplicates()).reset_index(drop=True)
