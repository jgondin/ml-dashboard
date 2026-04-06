"""
Train an XGBoost classifier for sensitive device prediction.

Outputs:
  models/xgboost_model.json   — trained XGBoost model
  models/shap_explainer.pkl   — SHAP TreeExplainer (wraps the model)

The inference parquet in data/predictions/ is expected to contain only
metadata + feature columns (no prediction_score, no shap_* columns).
The dashboard loads the explainer and computes predictions + SHAP at runtime.

Usage:
    uv run python train.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_PATH  = "data/predictions"
MODEL_DIR   = Path("models")
RANDOM_SEED = 42

FEATURE_COLS = [
    "login_failures",
    "geo_anomaly_score",
    "patch_lag_days",
    "failed_auths",
    "unusual_process_count",
    "data_exfil_bytes",
]

# Known sensitive devices used as ground-truth labels for training
SENSITIVE_DEVICES = {
    f"TenantA_ap_{i:03d}" for i in range(1, 5)
} | {
    f"TenantB_ap_{i:03d}" for i in range(1, 4)
} | {
    f"TenantC_ap_{i:03d}" for i in range(1, 4)
}

# ── Load inference data ───────────────────────────────────────────────────────
print("Loading inference data...")
df = pd.read_parquet(INPUT_PATH)
if not pd.api.types.is_datetime64_any_dtype(df["date"]):
    df["date"] = pd.to_datetime(df["date"])

# Derive binary label from known ground-truth sensitive devices
df["label"] = df["ap_serial"].isin(SENSITIVE_DEVICES).astype(int)

print(f"  Records       : {len(df):,}")
print(f"  Sensitive (1) : {df['label'].sum():,}  ({df['label'].mean()*100:.1f}%)")
print(f"  Non-sensitive : {(df['label']==0).sum():,}")

# ── Train / test split ────────────────────────────────────────────────────────
X = df[FEATURE_COLS]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
)

# ── Train XGBoost ─────────────────────────────────────────────────────────────
print("\nTraining XGBoost...")
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=RANDOM_SEED,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)
auc    = roc_auc_score(y_test, y_prob)

print(f"\nTest AUC : {auc:.4f}")
print(classification_report(y_test, y_pred, target_names=["Non-Sensitive", "Sensitive"]))

# ── Build and save SHAP explainer ─────────────────────────────────────────────
print("Building SHAP explainer...")
explainer = shap.TreeExplainer(model)

MODEL_DIR.mkdir(exist_ok=True)

model_path     = MODEL_DIR / "xgboost_model.json"
explainer_path = MODEL_DIR / "shap_explainer.pkl"

model.save_model(model_path)
with open(explainer_path, "wb") as f:
    pickle.dump(explainer, f)

print(f"\nSaved model    → {model_path}")
print(f"Saved explainer → {explainer_path}")
print("\nDone. Run the dashboard with: streamlit run threat_prediction_dashboard.py")
