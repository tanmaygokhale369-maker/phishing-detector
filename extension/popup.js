async function main() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) return;

  const urlBox     = document.getElementById("url-box");
  const card       = document.getElementById("card");
  const icon       = document.getElementById("icon");
  const label      = document.getElementById("label");
  const confidence = document.getElementById("confidence");

  urlBox.textContent = tab.url.length > 55
    ? tab.url.substring(0, 55) + "..."
    : tab.url;

  let result = null;

  // Try stored result first
  const key = `result_${tab.id}`;
  const stored = await chrome.storage.local.get(key);
  if (stored[key]) {
    result = stored[key];
  } else {
    // Call API directly
    try {
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: tab.url })
      });
      result = await res.json();
    } catch(e) {
      icon.textContent       = "⚠️";
      label.textContent      = "API Offline";
      label.className        = "status-label suspicious";
      confidence.textContent = "Start uvicorn in VS Code terminal";
      return;
    }
  }

  const pct     = Math.round((result.confidence || 0) * 100);
  const source  = result.source || "analysis";
  const signals = result.signals || [];
  const sigText = signals.slice(0, 2).join(" • ");

  // ── Render based on risk level ────────────────────────
  if (result.risk_level === "dangerous") {
    card.className         = "card dangerous";
    icon.textContent       = "🚫";
    label.textContent      = "Phishing Detected!";
    label.className        = "status-label dangerous";
    confidence.textContent = `${pct}% confidence — ${source}`;
    if (sigText) confidence.textContent += `\n${sigText}`;

  } else if (result.risk_level === "suspicious") {
    card.className         = "card suspicious";
    icon.textContent       = "⚠️";
    label.textContent      = "Suspicious Site";
    label.className        = "status-label suspicious";
    confidence.textContent = `${pct}% confidence — proceed carefully`;
    if (sigText) confidence.textContent += `\n${sigText}`;

  } else {
    card.className  = "card safe";
    icon.textContent = "✅";
    label.textContent = "Safe";
    label.className  = "status-label safe";

    // Show friendly message based on source
    if (source === "trusted_list") {
      confidence.textContent = "✓ Verified trusted domain";
    } else if (source === "Google Safe Browsing") {
      confidence.textContent = "✓ Cleared by Google Safe Browsing";
    } else {
      confidence.textContent = `${pct}% — No threats detected`;
    }
  }
}

main();