const API = "http://127.0.0.1:8000/predict";

async function checkURL(url, tabId) {
  if (!url || !url.startsWith("http")) return;

  try {
    const res = await fetch(API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: url })
    });

    if (!res.ok) throw new Error("API error");
    const data = await res.json();

    await chrome.storage.local.set({ [`result_${tabId}`]: data });

    if (data.risk_level === "dangerous") {
      chrome.action.setBadgeText({ text: "!", tabId });
      chrome.action.setBadgeBackgroundColor({ color: "#E24B4A", tabId });
    } else if (data.risk_level === "suspicious") {
      chrome.action.setBadgeText({ text: "?", tabId });
      chrome.action.setBadgeBackgroundColor({ color: "#BA7517", tabId });
    } else {
      chrome.action.setBadgeText({ text: "OK", tabId });
      chrome.action.setBadgeBackgroundColor({ color: "#1D9E75", tabId });
    }

  } catch (e) {
    console.log("API not reachable:", e.message);
    chrome.action.setBadgeText({ text: "...", tabId });
    chrome.action.setBadgeBackgroundColor({ color: "#888", tabId });
  }
}

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete" && tab.url) {
    checkURL(tab.url, tabId);
  }
});

chrome.tabs.onActivated.addListener(async ({ tabId }) => {
  try {
    const tab = await chrome.tabs.get(tabId);
    if (tab.url) checkURL(tab.url, tabId);
  } catch (e) {}
});