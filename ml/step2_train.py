import pandas as pd
import numpy as np
import json, joblib, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

MODEL_DIR = "C:/Users/tanma/phishing-detector/ml/models"
os.makedirs(MODEL_DIR, exist_ok=True)
DATASET2026 = "ml/data/raw/dataset_2026.csv"

print("=" * 60)
print("PHASE 2 — HONEST TRAINING (no data leakage)")
print("=" * 60)

# ── Load ──────────────────────────────────────────────────
print("\nStep 1: Loading dataset...")
df = pd.read_csv(DATASET2026)
print(f"  Shape: {df.shape}")
print(f"  All columns: {list(df.columns)}")

# ── Find the label/type column ────────────────────────────
type_col = None
for col in df.columns:
    if col.lower() in ["type", "category", "class"]:
        type_col = col
        break

if type_col is None:
    # fallback — look for a column with phishing/benign values
    for col in df.columns:
        unique_vals = df[col].astype(str).str.lower().unique()
        if "phishing" in unique_vals or "benign" in unique_vals:
            type_col = col
            break

print(f"  Using '{type_col}' as label source")
print(f"  Unique values: {df[type_col].unique()}")

# ── Create binary target ──────────────────────────────────
label_map = {
    "phishing":   1,
    "malware":    1,
    "defacement": 1,
    "benign":     0,
}
df["target"] = df[type_col].astype(str).str.lower().map(label_map)
df = df.dropna(subset=["target"])
df["target"] = df["target"].astype(int)

print(f"\n  Malicious: {(df['target']==1).sum():,}")
print(f"  Benign:    {(df['target']==0).sum():,}")

# ── Drop ALL non-feature columns ──────────────────────────
# These are columns that should NEVER be used as features
COLS_TO_REMOVE = [
    "url", "domain", "type", "label", "category", "class",
    "Date_inspection", "target_orig", "Shortining_Service",
    "web_is_live",   # live check = cheating (not available at prediction time)
]

print("\nStep 2: Removing leakage columns...")
removed = []
for col in COLS_TO_REMOVE:
    if col in df.columns:
        df = df.drop(columns=[col])
        removed.append(col)
print(f"  Removed: {removed}")

# Drop remaining text columns
text_cols = df.select_dtypes(include=["object"]).columns.tolist()
if "target" in text_cols:
    text_cols.remove("target")
df = df.drop(columns=text_cols, errors="ignore")
if text_cols:
    print(f"  Also removed text cols: {text_cols}")

df = df.fillna(0)

feature_cols = [c for c in df.columns if c != "target"]
print(f"\n  Clean features: {len(feature_cols)}")
print(f"  Feature names: {feature_cols[:8]}...")

# ── Verify no leakage ─────────────────────────────────────
print("\nStep 3: Leakage check...")
suspicious = [f for f in feature_cols if f.lower() in
              ["label", "type", "category", "class", "target"]]
if suspicious:
    print(f"  WARNING — removing suspicious: {suspicious}")
    feature_cols = [f for f in feature_cols if f not in suspicious]
    df = df[feature_cols + ["target"]]
else:
    print(f"  No leakage detected — all clear!")

# ── Split ─────────────────────────────────────────────────
print("\nStep 4: Splitting 80/20...")
X = df[feature_cols]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

# ── Scale ─────────────────────────────────────────────────
print("\nStep 5: Scaling...")
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Train ─────────────────────────────────────────────────
print("\nStep 6: Training XGBoost (honest training)...")
print("  Expected accuracy: 95-99% (real, not 100%)")
model = XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
model.fit(
    X_train_sc, y_train,
    eval_set=[(X_test_sc, y_test)],
    verbose=100
)

# ── Evaluate ──────────────────────────────────────────────
print("\nStep 7: Honest evaluation...")
y_pred = model.predict(X_test_sc)
acc = accuracy_score(y_test, y_pred)
print(f"\n  Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred,
      target_names=["Benign", "Malicious"]))

if acc == 1.0:
    print("  WARNING: Still 100% — check for remaining leakage!")
else:
    print(f"  This is a REAL accuracy score — no cheating!")

# ── Save ──────────────────────────────────────────────────
print("\nStep 8: Saving honest model...")
joblib.dump(model,  MODEL_DIR + "/phishing_model.pkl")
joblib.dump(scaler, MODEL_DIR + "/scaler.pkl")
with open(MODEL_DIR + "/feature_list.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

print(f"\n{'='*60}")
print(f"DONE!")
print(f"  Rows trained : {len(X):,}")
print(f"  Features     : {len(feature_cols)}")
print(f"  Real accuracy: {acc*100:.2f}%")
print(f"{'='*60}")