"""
MEGA TRAINING SCRIPT — Phase 3
================================
Combines ALL datasets + URLhaus live feed
Extracts 40 URL-only features from every URL
Trains XGBoost on everything combined
Target: 5M+ URLs, 95-97% real accuracy

Run: python ml/step3_mega_train.py
"""

import os
import re
import math
import json
import joblib
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

RAW_DIR   = Path("ml/data/raw")
MODEL_DIR = Path("ml/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ─── FEATURE EXTRACTION ───────────────────────────────────────────────────────
# 40 URL-only features — no webpage visit needed — works in 0ms

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "secure", "account", "update",
    "banking", "paypal", "confirm", "password", "wallet", "bitcoin",
    "apple", "microsoft", "amazon", "google", "facebook", "netflix",
    "support", "ebay", "alert", "suspend", "urgent", "free", "lucky",
    "winner", "prize", "click", "verify", "credential", "recover",
]

BAD_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top",
    ".club", ".online", ".site", ".buzz", ".icu", ".fun",
    ".rest", ".click", ".monster", ".cyou", ".cfd",
}

TRUSTED_DOMAINS = {
    "google.com", "youtube.com", "facebook.com", "amazon.com",
    "wikipedia.org", "twitter.com", "instagram.com", "linkedin.com",
    "github.com", "microsoft.com", "apple.com", "netflix.com",
    "reddit.com", "yahoo.com", "whatsapp.com", "zoom.us",
}

def shannon_entropy(s: str) -> float:
    if not s: return 0.0
    freq = [s.count(c) / len(s) for c in set(s)]
    return round(-sum(p * math.log2(p) for p in freq), 4)

def extract_features(url: str) -> dict:
    """Extract 40 URL-only features instantly — no network call."""
    try:
        url = str(url).strip()
        if not url.startswith("http"):
            url = "http://" + url

        parsed  = urlparse(url)
        domain  = parsed.netloc.lower()
        path    = parsed.path.lower()
        query   = parsed.query.lower()
        url_low = url.lower()

        # Clean domain
        domain_clean = domain.replace("www.", "")
        parts = domain_clean.split(".")
        tld   = "." + parts[-1] if len(parts) > 1 else ""
        sld   = parts[-2] if len(parts) >= 2 else domain_clean

        # Keyword counts
        kw_hits = sum(1 for k in SUSPICIOUS_KEYWORDS if k in url_low)
        brand_in_domain = any(b in domain_clean for b in [
            "paypal", "apple", "amazon", "microsoft", "google",
            "facebook", "netflix", "instagram", "bank"
        ])

        return {
            # Length features
            "url_length":        len(url),
            "domain_length":     len(domain),
            "path_length":       len(path),
            "query_length":      len(query),
            "tld_length":        len(tld),

            # Count features
            "num_dots":          url.count("."),
            "num_hyphens":       url.count("-"),
            "num_underscores":   url.count("_"),
            "num_slashes":       url.count("/"),
            "num_question":      url.count("?"),
            "num_equals":        url.count("="),
            "num_ampersand":     url.count("&"),
            "num_at":            url.count("@"),
            "num_percent":       url.count("%"),
            "num_hash":          url.count("#"),
            "num_digits":        sum(c.isdigit() for c in url),
            "num_letters":       sum(c.isalpha() for c in url),

            # Ratio features
            "digit_ratio":       round(sum(c.isdigit() for c in url) / max(len(url), 1), 4),
            "letter_ratio":      round(sum(c.isalpha() for c in url) / max(len(url), 1), 4),
            "special_ratio":     round(sum(not c.isalnum() for c in url) / max(len(url), 1), 4),
            "domain_digit_ratio":round(sum(c.isdigit() for c in domain) / max(len(domain), 1), 4),

            # Entropy
            "url_entropy":       shannon_entropy(url),
            "domain_entropy":    shannon_entropy(sld),
            "path_entropy":      shannon_entropy(path),

            # Boolean features
            "is_https":          int(parsed.scheme == "https"),
            "has_ip":            int(bool(re.match(r"https?://\d+\.\d+\.\d+\.\d+", url))),
            "has_port":          int(bool(re.search(r":\d{2,5}(/|$)", domain))),
            "has_double_slash":  int("//" in path),
            "has_at_symbol":     int("@" in url),
            "bad_tld":           int(tld in BAD_TLDS),
            "is_shortened":      int(any(s in domain for s in ["bit.ly","tinyurl","goo.gl","t.co","ow.ly","short","tiny"])),

            # Phishing signals
            "keyword_count":     kw_hits,
            "has_keywords":      int(kw_hits > 0),
            "brand_in_domain":   int(brand_in_domain),
            "brand_in_path":     int(any(b in path for b in ["paypal","apple","amazon","microsoft","google"])),
            "subdomain_count":   max(0, domain.count(".") - 1),
            "path_depth":        len([p for p in path.split("/") if p]),
            "param_count":       len(query.split("&")) if query else 0,
            "has_executable":    int(bool(re.search(r"\.(exe|bat|cmd|sh|php|vbs|jar)$", path))),
            "trusted_domain":    int(domain_clean in TRUSTED_DOMAINS or
                                    any(domain_clean.endswith("."+t) for t in TRUSTED_DOMAINS)),
        }
    except Exception:
        return {k: 0 for k in [
            "url_length","domain_length","path_length","query_length","tld_length",
            "num_dots","num_hyphens","num_underscores","num_slashes","num_question",
            "num_equals","num_ampersand","num_at","num_percent","num_hash",
            "num_digits","num_letters","digit_ratio","letter_ratio","special_ratio",
            "domain_digit_ratio","url_entropy","domain_entropy","path_entropy",
            "is_https","has_ip","has_port","has_double_slash","has_at_symbol",
            "bad_tld","is_shortened","keyword_count","has_keywords","brand_in_domain",
            "brand_in_path","subdomain_count","path_depth","param_count",
            "has_executable","trusted_domain",
        ]}

FEATURE_NAMES = list(extract_features("https://example.com").keys())
print(f"Features defined: {len(FEATURE_NAMES)}")

# ─── DATASET LOADERS ─────────────────────────────────────────────────────────

def load_urlhaus() -> pd.DataFrame:
    """Download URLhaus live malware feed — 3.6M URLs updated every 5 min."""
    print("\n[URLhaus] Downloading live malware feed...")
    try:
        r = requests.get(
            "https://urlhaus.abuse.ch/downloads/csv_online/",
            timeout=60,
            headers={"User-Agent": "PhishingDetector/1.0 research"}
        )
        from io import StringIO
        lines = [l for l in r.text.splitlines() if not l.startswith("#")]
        df = pd.read_csv(StringIO("\n".join(lines)), on_bad_lines="skip")
        df = df[["url"]].dropna()
        df["label"] = 1  # all malicious
        df["source"] = "urlhaus"
        print(f"  URLhaus: {len(df):,} malware URLs")
        return df[["url","label","source"]]
    except Exception as e:
        print(f"  URLhaus failed: {e}")
        return pd.DataFrame(columns=["url","label","source"])


def load_openphish() -> pd.DataFrame:
    """Download OpenPhish live phishing feed."""
    print("[OpenPhish] Downloading live phishing feed...")
    try:
        r = requests.get("https://openphish.com/feed.txt", timeout=30,
                        headers={"User-Agent": "PhishingDetector/1.0"})
        urls = [l.strip() for l in r.text.splitlines() if l.strip()]
        df = pd.DataFrame({"url": urls, "label": 1, "source": "openphish"})
        print(f"  OpenPhish: {len(df):,} phishing URLs")
        return df
    except Exception as e:
        print(f"  OpenPhish failed: {e}")
        return pd.DataFrame(columns=["url","label","source"])


def load_csv_dataset(path: Path, url_col: str, label_col: str,
                     label_map: dict, source: str) -> pd.DataFrame:
    """Generic CSV loader with flexible column mapping."""
    try:
        df = pd.read_csv(path, on_bad_lines="skip", low_memory=False)
        print(f"  Columns: {list(df.columns[:6])}")

        # Find URL column
        url_c = next((c for c in df.columns if c.lower() == url_col.lower()), None)
        if url_c is None:
            url_c = next((c for c in df.columns
                         if any(k in c.lower() for k in ["url","link","website"])), None)
        if url_c is None:
            print(f"  ERROR: no URL column found in {path.name}")
            return pd.DataFrame(columns=["url","label","source"])

        # Find label column
        lbl_c = next((c for c in df.columns if c.lower() == label_col.lower()), None)
        if lbl_c is None:
            lbl_c = next((c for c in df.columns
                         if any(k in c.lower() for k in ["label","type","status","class"])), None)

        if lbl_c is None:
            print(f"  ERROR: no label column in {path.name}")
            return pd.DataFrame(columns=["url","label","source"])

        df["url"]    = df[url_c].astype(str)
        df["label"]  = df[lbl_c].astype(str).str.lower().str.strip().map(label_map)
        df = df.dropna(subset=["label"])
        df["label"]  = df["label"].astype(int)
        df["source"] = source

        print(f"  {source}: {len(df):,} rows | "
              f"phishing={int((df['label']==1).sum()):,} "
              f"legit={int((df['label']==0).sum()):,}")
        return df[["url","label","source"]]

    except Exception as e:
        print(f"  ERROR loading {path.name}: {e}")
        return pd.DataFrame(columns=["url","label","source"])


# ─── LOAD ALL DATASETS ────────────────────────────────────────────────────────

def load_all_datasets() -> pd.DataFrame:
    print("\n" + "="*60)
    print("LOADING ALL DATASETS")
    print("="*60)

    frames = []

    # 1. URLhaus live
    frames.append(load_urlhaus())

    # 2. OpenPhish live
    frames.append(load_openphish())

    # 3. malicious_phish.csv / url_dataset.csv (sid321axn style)
    for fname in ["malicious_phish.csv", "url_dataset.csv"]:
        path = RAW_DIR / fname
        if path.exists():
            print(f"\n[{fname}] Loading...")
            frames.append(load_csv_dataset(
                path, "url", "type",
                {"benign":0,"legitimate":0,"phishing":1,"malware":1,"defacement":1},
                fname
            ))

    # 4. phishing_site_urls.csv (taruntiwarihp)
    path = RAW_DIR / "phishing_site_urls.csv"
    if path.exists():
        print(f"\n[phishing_site_urls.csv] Loading...")
        frames.append(load_csv_dataset(
            path, "URL", "Label",
            {"good":0,"legitimate":0,"bad":1,"phishing":1,"0":0,"1":1},
            "phishing_site_urls"
        ))

    # 5. urlset.csv (joebeachcapital)
    path = RAW_DIR / "urlset.csv"
    if path.exists():
        print(f"\n[urlset.csv] Loading...")
        frames.append(load_csv_dataset(
            path, "url", "label",
            {"good":0,"legitimate":0,"bad":1,"phishing":1,"0":0,"1":1},
            "urlset"
        ))

    # 6. Phishing_Legitimate_full.csv (shashwatwork)
    path = RAW_DIR / "Phishing_Legitimate_full.csv"
    if path.exists():
        print(f"\n[Phishing_Legitimate_full.csv] Loading...")
        frames.append(load_csv_dataset(
            path, "url", "label",
            {"legitimate":0,"phishing":1,"0":0,"1":1,"good":0,"bad":1},
            "phishing_legitimate"
        ))

    # 7. PhiUSIIL — URL + label columns only
    phiusiil_dir = RAW_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv"
    phiusiil_csv = None
    if phiusiil_dir.is_dir():
        csvs = list(phiusiil_dir.glob("*.csv"))
        if csvs: phiusiil_csv = csvs[0]
    elif phiusiil_dir.is_file():
        phiusiil_csv = phiusiil_dir

    if phiusiil_csv:
        print(f"\n[PhiUSIIL] Loading URL+label only...")
        try:
            df = pd.read_csv(phiusiil_csv, usecols=lambda c: c.lower() in ["url","label"],
                            on_bad_lines="skip", low_memory=False)
            url_c = next((c for c in df.columns if c.lower()=="url"), None)
            lbl_c = next((c for c in df.columns if c.lower()=="label"), None)
            if url_c and lbl_c:
                df["url"]    = df[url_c]
                df["label"]  = pd.to_numeric(df[lbl_c], errors="coerce")
                df = df.dropna(subset=["label"])
                df["label"]  = df["label"].astype(int)
                df["source"] = "phiusiil"
                frames.append(df[["url","label","source"]])
                print(f"  PhiUSIIL: {len(df):,} rows")
        except Exception as e:
            print(f"  PhiUSIIL error: {e}")

    # Combine all
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["url","label"])
    combined["url"] = combined["url"].astype(str).str.strip()
    combined = combined[combined["url"].str.len() > 5]
    combined = combined.drop_duplicates(subset=["url"])
    combined["label"] = combined["label"].astype(int)

    print(f"\n{'='*60}")
    print(f"COMBINED: {len(combined):,} unique URLs")
    print(f"  Phishing/Malicious: {(combined['label']==1).sum():,}")
    print(f"  Legitimate:         {(combined['label']==0).sum():,}")
    print(f"\nSources:")
    for src, cnt in combined["source"].value_counts().items():
        print(f"  {src:<30} {cnt:>10,}")
    print("="*60)

    return combined


# ─── FEATURE EXTRACTION PIPELINE ─────────────────────────────────────────────

def extract_all_features(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\nExtracting {len(FEATURE_NAMES)} features from {len(df):,} URLs...")
    print("This may take 10-30 minutes depending on dataset size...")

    batch_size = 50_000
    all_features = []

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        feats = [extract_features(url) for url in batch["url"]]
        all_features.extend(feats)

        done = min(i + batch_size, len(df))
        pct  = done / len(df) * 100
        print(f"  {done:>8,} / {len(df):,} ({pct:.1f}%)")

    feat_df = pd.DataFrame(all_features, columns=FEATURE_NAMES)
    feat_df["label"] = df["label"].values
    return feat_df


# ─── TRAINING ────────────────────────────────────────────────────────────────

def train(feat_df: pd.DataFrame):
    print(f"\n{'='*60}")
    print("TRAINING XGBOOST MODEL")
    print(f"{'='*60}")

    X = feat_df[FEATURE_NAMES]
    y = feat_df["label"]

    # Balance dataset — cap majority class at 3x minority
    n_phish = int((y==1).sum())
    n_legit = int((y==0).sum())
    minority = min(n_phish, n_legit)
    cap      = min(minority * 3, 500_000)

    if n_phish > cap:
        idx_phish = feat_df[feat_df["label"]==1].sample(cap, random_state=42).index
        idx_legit = feat_df[feat_df["label"]==0].index
        feat_df   = feat_df.loc[idx_phish.union(idx_legit)]
    elif n_legit > cap:
        idx_legit = feat_df[feat_df["label"]==0].sample(cap, random_state=42).index
        idx_phish = feat_df[feat_df["label"]==1].index
        feat_df   = feat_df.loc[idx_phish.union(idx_legit)]

    X = feat_df[FEATURE_NAMES].fillna(0)
    y = feat_df["label"]
    print(f"Balanced dataset: {len(X):,} rows")
    print(f"  Phishing: {(y==1).sum():,} | Legit: {(y==0).sum():,}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

    # Scale
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Train
    print("\nTraining XGBoost (this takes 5-20 minutes)...")
    model = XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train_sc, y_train,
        eval_set=[(X_test_sc, y_test)],
        verbose=100
    )

    # Evaluate
    y_pred = model.predict(X_test_sc)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["Legitimate", "Phishing"],
        output_dict=True
    )
    print(f"\nAccuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred,
          target_names=["Legitimate","Phishing"]))

    # Save
    joblib.dump(model,  MODEL_DIR / "phishing_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    with open(MODEL_DIR / "feature_list.json", "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)
    with open(MODEL_DIR / "model_metrics.json", "w") as f:
        json.dump({
            "accuracy": float(acc),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "features": len(FEATURE_NAMES),
            "evaluation": "Stratified random 80/20 split (random_state=42)",
            "classification_report": report,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"MODEL SAVED!")
    print(f"  Features  : {len(FEATURE_NAMES)}")
    print(f"  Train rows: {len(X_train):,}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Saved to  : ml/models/")
    print(f"{'='*60}")
    print("\nNow restart uvicorn to load the new model!")

    return model, scaler, acc


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*60)
    print("PHISHING DETECTOR — MEGA TRAINING PIPELINE")
    print("="*60)

    # Step 1: Load all datasets
    df = load_all_datasets()

    # Step 2: Extract URL-only features
    feat_df = extract_all_features(df)

    # Step 3: Train model
    model, scaler, acc = train(feat_df)

    print("\nDone! Your model is now trained on real URL-only features.")
    print("This model works on ANY raw URL instantly — no webpage visit needed.")
