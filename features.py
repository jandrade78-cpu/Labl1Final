"""Feature engineering shared by train_model.py and the Streamlit app (no streamlit import)."""
import numpy as np
import pandas as pd
import config as C


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    d = pd.to_datetime(out["Submission_Date"], errors="coerce")
    out["Submit_Month"] = d.dt.month.astype("float")
    out["Submit_Weekday"] = d.dt.day_name().str[:3]
    for c in C.CATEGORICAL:
        out[c] = out[c].astype(object).where(out[c].notna(), np.nan)
    if "Processing_Days" in out.columns:
        known = out["Processing_Days"].notna()
        out[C.TARGET] = np.where(known, (out["Processing_Days"] > out["SLA_Days"]).astype(float), np.nan)
    return out
