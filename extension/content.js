chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type !== "BLOCK") return;

  document.body.innerHTML = `
    <div style="display:flex;flex-direction:column;align-items:center;
      justify-content:center;min-height:100vh;background:#FCEBEB;
      font-family:system-ui;text-align:center;padding:2rem">
      <div style="font-size:80px;margin-bottom:1rem">🚫</div>
      <h1 style="color:#A32D2D;font-size:28px;margin-bottom:.5rem">
        Phishing Site Blocked
      </h1>
      <p style="color:#791F1F;max-width:400px;line-height:1.6">
        Our ML model detected this as a phishing site with
        <strong>${msg.confidence}%</strong> confidence.
      </p>
      <button onclick="history.back()"
        style="margin-top:1.5rem;padding:.6rem 1.5rem;
        background:#A32D2D;color:#fff;border:none;
        border-radius:8px;cursor:pointer;font-size:15px">
        ← Go back to safety
      </button>
    </div>`;
});