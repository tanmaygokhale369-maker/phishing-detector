from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, json, numpy as np
import re, math, requests, os
from urllib.parse import urlparse

app = FastAPI(title="Phishing Detector API v2 — URL-only ML Model")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Load ML Model ─────────────────────────────────────────
MODEL_DIR = "ml/models"
model  = joblib.load(f"{MODEL_DIR}/phishing_model.pkl")
scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
with open(f"{MODEL_DIR}/feature_list.json") as f:
    FEATURES = json.load(f)
print(f"✓ Model loaded — {len(FEATURES)} URL-only features")

# ── Load live cache ───────────────────────────────────────
LIVE_CACHE = "ml/models/live_phishing_urls.json"
live_domains = set()
live_urls    = set()

def load_cache():
    global live_domains, live_urls
    try:
        if os.path.exists(LIVE_CACHE):
            with open(LIVE_CACHE) as f:
                data = json.load(f)
            live_domains = set(data.get("domains", []))
            live_urls    = set(data.get("urls", []))
            print(f"✓ Live cache: {len(live_urls):,} URLs, {len(live_domains):,} domains")
    except Exception as e:
        print(f"Cache load failed: {e}")

load_cache()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

TRUSTED = {
    "google.com","youtube.com","facebook.com","amazon.com","amazon.in",
    "wikipedia.org","twitter.com","instagram.com","linkedin.com","github.com",
    "microsoft.com","apple.com","netflix.com","reddit.com","yahoo.com",
    "hotstar.com","flipkart.com","jiohotstar.com","stackoverflow.com",
    "claude.ai","anthropic.com","kaggle.com","sonyliv.com","fundedsquad.com",
    "zerodha.com","groww.in","paytm.com","phonepe.com","swiggy.com",
    "zomato.com","instantfunding.com","tradingview.com","binance.com",
    "coinbase.com","upstox.com","angelone.in","nseindia.com","bseindia.com",
}

BAD_TLDS = {
    ".tk",".ml",".ga",".cf",".gq",".xyz",".top",
    ".club",".online",".site",".buzz",".icu",".fun",
    ".rest",".click",".monster",".cyou",".cfd",
}

SUSPICIOUS_KEYWORDS = [
    "login","signin","verify","secure","account","update","banking",
    "paypal","confirm","password","wallet","bitcoin","apple","microsoft",
    "amazon","google","facebook","netflix","support","ebay","alert",
    "suspend","urgent","free","lucky","winner","prize","credential","recover",
]

def entropy(s):
    if not s: return 0
    freq = [s.count(c)/len(s) for c in set(s)]
    return round(-sum(p*math.log2(p) for p in freq), 4)

def extract_features(url: str) -> np.ndarray:
    try:
        if not url.startswith("http"):
            url = "http://" + url
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path   = parsed.path.lower()
        query  = parsed.query.lower()
        url_low = url.lower()
        domain_clean = domain.replace("www.", "")
        parts = domain_clean.split(".")
        tld   = "." + parts[-1] if len(parts) > 1 else ""
        sld   = parts[-2] if len(parts) >= 2 else domain_clean
        kw_hits = sum(1 for k in SUSPICIOUS_KEYWORDS if k in url_low)
        brand_in_domain = any(b in domain_clean for b in [
            "paypal","apple","amazon","microsoft","google","facebook","netflix","instagram","bank"
        ])
        feat = {
            "url_length":        len(url),
            "domain_length":     len(domain),
            "path_length":       len(path),
            "query_length":      len(query),
            "tld_length":        len(tld),
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
            "digit_ratio":       round(sum(c.isdigit() for c in url)/max(len(url),1),4),
            "letter_ratio":      round(sum(c.isalpha() for c in url)/max(len(url),1),4),
            "special_ratio":     round(sum(not c.isalnum() for c in url)/max(len(url),1),4),
            "domain_digit_ratio":round(sum(c.isdigit() for c in domain)/max(len(domain),1),4),
            "url_entropy":       entropy(url),
            "domain_entropy":    entropy(sld),
            "path_entropy":      entropy(path),
            "is_https":          int(parsed.scheme=="https"),
            "has_ip":            int(bool(re.match(r"https?://\d+\.\d+\.\d+\.\d+",url))),
            "has_port":          int(bool(re.search(r":\d{2,5}(/|$)",domain))),
            "has_double_slash":  int("//" in path),
            "has_at_symbol":     int("@" in url),
            "bad_tld":           int(tld in BAD_TLDS),
            "is_shortened":      int(any(s in domain for s in ["bit.ly","tinyurl","goo.gl","t.co","ow.ly"])),
            "keyword_count":     kw_hits,
            "has_keywords":      int(kw_hits > 0),
            "brand_in_domain":   int(brand_in_domain),
            "brand_in_path":     int(any(b in path for b in ["paypal","apple","amazon","microsoft","google"])),
            "subdomain_count":   max(0, domain.count(".")-1),
            "path_depth":        len([p for p in path.split("/") if p]),
            "param_count":       len(query.split("&")) if query else 0,
            "has_executable":    int(bool(re.search(r"\.(exe|bat|cmd|sh|php|vbs|jar)$",path))),
            "trusted_domain":    int(domain_clean in TRUSTED or any(domain_clean.endswith("."+t) for t in TRUSTED)),
        }
        return np.array([[feat.get(f,0) for f in FEATURES]])
    except:
        return np.zeros((1, len(FEATURES)))

def check_gsb(url):
    try:
        if not GOOGLE_API_KEY:
            return False, None

        payload = {
            "client": {"clientId":"phishing-detector","clientVersion":"2.0"},
            "threatInfo": {
                "threatTypes":["MALWARE","SOCIAL_ENGINEERING",
                               "UNWANTED_SOFTWARE","POTENTIALLY_HARMFUL_APPLICATION"],
                "platformTypes":["ANY_PLATFORM"],
                "threatEntryTypes":["URL"],
                "threatEntries":[{"url":url}]
            }
        }
        gsb_url = (
            "https://safebrowsing.googleapis.com/v4/threatMatches:find"
            f"?key={GOOGLE_API_KEY}"
        )
        r = requests.post(gsb_url, json=payload, timeout=5)
        data = r.json()
        if "matches" in data and data["matches"]:
            return True, data["matches"][0]["threatType"]
        return False, None
    except:
        return False, None

class URLRequest(BaseModel):
    url: str

@app.get("/")
def home():
    return {
        "status": "Phishing Detector API v2",
        "model": f"XGBoost — {len(FEATURES)} URL-only features",
        "accuracy": "98.03%",
        "live_cache": len(live_urls),
    }

@app.get("/health")
def health():
    return {"status":"ok","features":len(FEATURES),"live_urls":len(live_urls)}

@app.post("/reload-cache")
def reload():
    load_cache()
    return {"status":"reloaded","urls":len(live_urls)}

@app.post("/predict")
def predict(req: URLRequest):
    url = req.url.strip()
    if not url.startswith("http"):
        url = "https://" + url

    parsed = urlparse(url)
    domain = parsed.netloc.lower().replace("www.","")

    # ── Layer 1: Trusted whitelist ────────────────────────
    for t in TRUSTED:
        if domain == t or domain.endswith("."+t):
            return {
                "url":url,"is_phishing":False,
                "confidence":0.01,"risk_level":"safe",
                "signals":["Verified trusted domain"],
                "source":"trusted_list"
            }

    # ── Layer 2: Live phishing cache ──────────────────────
    if domain in live_domains or url in live_urls:
        return {
            "url":url,"is_phishing":True,
            "confidence":0.97,"risk_level":"dangerous",
            "signals":["Found in live phishing database",
                      "Confirmed active phishing site"],
            "source":"OpenPhish live feed"
        }

    # ── Layer 3: Google Safe Browsing ─────────────────────
    gsb_hit, threat = check_gsb(url)
    if gsb_hit:
        return {
            "url":url,"is_phishing":True,
            "confidence":0.99,"risk_level":"dangerous",
            "signals":[f"Google confirmed: {threat}",
                      "In Google live threat database"],
            "source":"Google Safe Browsing"
        }

    # ── Layer 4: ML Model (URL-only, 98% accuracy) ────────
    try:
        X     = extract_features(url)
        X_sc  = scaler.transform(X)
        prob  = float(model.predict_proba(X_sc)[0][1])
    except Exception as e:
        prob = 0.1

    # Build signals from features
    signals = []
    parsed2 = urlparse(url)
    dom2    = parsed2.netloc.lower()
    parts2  = dom2.replace("www.","").split(".")
    tld2    = "."+parts2[-1] if len(parts2)>1 else ""

    if re.match(r"https?://\d+\.\d+\.\d+\.\d+", url):
        signals.append("IP address used as domain")
    if tld2 in BAD_TLDS:
        signals.append(f"High-risk TLD: {tld2}")
    if "@" in url:
        signals.append("@ symbol in URL")
    kws = [k for k in SUSPICIOUS_KEYWORDS if k in url.lower()]
    if kws:
        signals.append(f"Phishing keywords: {', '.join(kws[:3])}")
    if max(0, dom2.count(".")-1) >= 3:
        signals.append("Deep subdomain nesting")
    if not url.startswith("https") and prob > 0.3:
        signals.append("No HTTPS")

    if prob >= 0.70:
        risk = "dangerous"
    elif prob >= 0.40:
        risk = "suspicious"
    else:
        risk = "safe"
        signals = signals or ["No threats detected"]

    return {
        "url":url,
        "is_phishing": risk != "safe",
        "confidence":  round(prob, 4),
        "risk_level":  risk,
        "signals":     signals or ["URL pattern analysis complete"],
        "source":      f"ML Model (98% accuracy, {len(FEATURES)} features)"
    }