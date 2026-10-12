import requests
import json
import time
import os
from datetime import datetime
from urllib.parse import urlparse

CACHE_FILE = "ml/models/live_phishing_urls.json"
UPDATE_INTERVAL = 1800  # 30 minutes now

EXTRA_BAD_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top",
    ".club", ".online", ".site", ".buzz", ".icu", ".fun",
    ".rest", ".click", ".monster", ".cyou", ".cfd",
    ".pp.ua", ".000webhostapp.com", ".netlify.app",
    ".vercel.app", ".glitch.me", ".repl.co",
}

def fetch_openphish():
    try:
        r = requests.get(
            "https://openphish.com/feed.txt",
            timeout=30,
            headers={"User-Agent": "PhishingDetector/2.0"}
        )
        urls = set(l.strip() for l in r.text.splitlines() if l.strip())
        print(f"  OpenPhish: {len(urls):,} URLs")
        return urls
    except Exception as e:
        print(f"  OpenPhish failed: {e}")
        return set()

def fetch_urlhaus():
    try:
        r = requests.get(
            "https://urlhaus.abuse.ch/downloads/text/",
            timeout=60,
            headers={"User-Agent": "PhishingDetector/2.0"}
        )
        urls = set(
            l.strip() for l in r.text.splitlines()
            if l.strip() and not l.startswith("#")
        )
        print(f"  URLhaus: {len(urls):,} URLs")
        return urls
    except Exception as e:
        print(f"  URLhaus failed: {e}")
        return set()

def fetch_phishtank():
    try:
        r = requests.get(
            "http://data.phishtank.com/data/online-valid.csv",
            timeout=60,
            headers={"User-Agent": "phishing-detector-research"}
        )
        urls = set()
        for line in r.text.splitlines()[1:]:
            parts = line.split(",")
            if len(parts) >= 2:
                urls.add(parts[1].strip('"'))
        print(f"  PhishTank: {len(urls):,} URLs")
        return urls
    except Exception as e:
        print(f"  PhishTank failed: {e}")
        return set()

def extract_domains(urls):
    domains = set()
    for url in urls:
        try:
            d = urlparse(url).netloc.lower().replace("www.", "")
            if d:
                domains.add(d)
        except:
            pass
    return domains

def update():
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Updating live phishing feeds...")

    all_urls = set()
    all_urls |= fetch_openphish()
    all_urls |= fetch_urlhaus()
    all_urls |= fetch_phishtank()

    domains = extract_domains(all_urls)

    os.makedirs("ml/models", exist_ok=True)
    data = {
        "updated_at": datetime.now().isoformat(),
        "url_count": len(all_urls),
        "domain_count": len(domains),
        "urls": list(all_urls),
        "domains": list(domains),
    }
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f)

    print(f"  Saved {len(all_urls):,} URLs | {len(domains):,} domains")
    print(f"  Next update in 30 minutes...")

if __name__ == "__main__":
    print("Live phishing updater started — pulling from 3 sources")
    while True:
        update()
        time.sleep(UPDATE_INTERVAL)