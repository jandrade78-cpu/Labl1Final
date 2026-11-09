"""python train_model.py [path/to/student_service_requests.csv]
Rebuilds the midterm model: target SLA_Breach = Processing_Days > SLA_Days (labeled rows only),
leakage columns excluded, preprocessing + model saved as ONE joblib pipeline."""
import sys, os, datetime as dt
import pandas as pd, joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
import config as C
from features import prepare

VERSION = "1.0.0"


def build(model):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), C.NUMERIC),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), C.CATEGORICAL)])
    return Pipeline([("pre", pre), ("model", model)])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else C.SAMPLE_DATA_PATH
    df = prepare(pd.read_csv(path))
    lab = df[df[C.TARGET].notna()].copy()
    X, y = lab[C.FEATURES], lab[C.TARGET].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    cands = {"Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, class_weight="balanced", random_state=42),
             "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=3,
                                                     class_weight="balanced", random_state=42)}
    comp = {}
    for n, m in cands.items():
        comp[n] = float(cross_val_score(build(m), Xtr, ytr, cv=5, scoring="f1").mean())
    best = max(comp, key=comp.get)
    pipe = build(cands[best]).fit(Xtr, ytr)
    pred = pipe.predict(Xte)

    names = pipe.named_steps["pre"].get_feature_names_out()
    imp = dict(zip(names, pipe.named_steps["model"].feature_importances_.round(4).tolist()))
    sample = Xte.head(5)
    meta = {
        "model_name": f"{best} (preprocessing + model pipeline)", "version": VERSION,
        "trained_at": f"{dt.datetime.now():%Y-%m-%d %H:%M}", "target": "SLA_Breach = Processing_Days > SLA_Days",
        "features": C.FEATURES, "n_labeled": len(lab), "n_train": len(Xtr), "n_test": len(Xte),
        "breach_rate": float(y.mean()), "cv_f1": comp,
        "categories": {c: sorted(lab[c].dropna().unique().tolist()) for c in C.CATEGORICAL if c != "Submit_Weekday"},
        "metrics": {"accuracy": accuracy_score(yte, pred), "precision": precision_score(yte, pred, zero_division=0),
                    "recall": recall_score(yte, pred), "f1": f1_score(yte, pred),
                    "confusion_matrix": confusion_matrix(yte, pred).tolist()},
        "feature_importance": imp,
        "consistency_cases": {"X": sample.to_dict("records"), "proba": pipe.predict_proba(sample)[:, 1].round(6).tolist()},
    }
    joblib.dump({"pipeline": pipe, "metadata": meta}, C.MODEL_PATH)
    print("CV F1:", {k: round(v, 3) for k, v in comp.items()}, "-> chose", best)
    print("Test:", {k: round(v, 3) for k, v in meta["metrics"].items() if k != "confusion_matrix"}, "n_labeled", len(lab))


if __name__ == "__main__":
    main()
