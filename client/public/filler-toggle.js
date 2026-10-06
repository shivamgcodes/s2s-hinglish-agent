// Turn-filler toggle (user 2026-10-06): top-right; "Ticker" = 1 s soft clicks fed to the MODEL input when the agent
// pauses (not after a check-line), so it yields the turn; "Off" = nothing. Applies immediately and persists.
(function () {
  const box = document.createElement("div");
  box.style.cssText = "position:fixed;top:10px;right:14px;z-index:9999;font:13px system-ui,sans-serif;" +
    "background:#fff;border:1px solid #ccc;border-radius:8px;padding:6px 10px;box-shadow:0 1px 4px rgba(0,0,0,.15);" +
    "display:flex;gap:8px;align-items:center;color:#222";
  box.innerHTML = '<span title="When the agent pauses, feed the model a 1 s soft ticker so it yields the turn (you never hear it).">Turn filler:</span>' +
    '<button data-m="ticker">Ticker</button><button data-m="off">Off</button>';
  const btns = box.querySelectorAll("button");
  btns.forEach(b => b.style.cssText = "border:1px solid #999;border-radius:6px;padding:2px 10px;cursor:pointer;background:#f4f4f4");
  function show(m) {
    btns.forEach(b => { const on = b.dataset.m === m; b.style.background = on ? "#2e7d32" : "#f4f4f4"; b.style.color = on ? "#fff" : "#222"; });
  }
  async function get() { try { const r = await fetch("/api/filler"); show((await r.json()).mode); } catch (e) {} }
  btns.forEach(b => b.onclick = async () => {
    try { const r = await fetch("/api/filler?mode=" + b.dataset.m, { method: "POST" }); show((await r.json()).mode); } catch (e) {}
  });
  document.addEventListener("DOMContentLoaded", () => { document.body.appendChild(box); get(); });
  if (document.readyState !== "loading") { document.body.appendChild(box); get(); }
})();
