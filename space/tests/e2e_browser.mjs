// Headless browser pass through the Space backend against fake_runpod_lite (Playwright Chromium, fake mic).
// Proves the client fork: setup -> Connect -> warm-up panel (waking) -> ready -> <Conversation> -> handshake ->
// live -> session_end reason shown. run_e2e_browser.sh starts the fake LB + Space and runs this.
// Usage: cd /root/e2e && node /root/deploy_serverless/space/tests/e2e_browser.mjs [baseUrl] [outDir] [maxWaitS]
import { chromium } from "/root/e2e/node_modules/playwright/index.mjs";
import fs from "node:fs";

const base = process.argv[2] ?? "http://localhost:27860";
const out = process.argv[3] ?? "/root/deploy_serverless/space/tests/out_e2e";
const maxWait = Number(process.argv[4] ?? 90);
const wav = "/root/deploy/assets/ws/hinglish/tests/inputs/V3/cab_07_g3.wav";
fs.mkdirSync(out, { recursive: true });

const res = { base, console_errors: [], ws_urls: [], warmup_states: [], checks: {} };
const browser = await chromium.launch({
  args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
    `--use-file-for-fake-audio-capture=${wav}`, "--autoplay-policy=no-user-gesture-required"],
});
const ctx = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
await ctx.grantPermissions(["microphone"], { origin: base });
const page = await ctx.newPage();
page.on("console", m => { if (m.type() === "error") res.console_errors.push(m.text().slice(0, 300)); });
page.on("pageerror", e => res.console_errors.push("pageerror: " + String(e).slice(0, 300)));
const reqs = [];
page.on("request", r => { if (r.url().includes("/api/session")) reqs.push(`${r.method()} ${new URL(r.url()).pathname}`); });
let wsFrames = { recv: 0, sent: 0, k7: 0, k1: 0 };
page.on("websocket", ws => {
  res.ws_urls.push(ws.url());
  ws.on("framereceived", f => {
    wsFrames.recv++;
    if (typeof f.payload !== "string") { if (f.payload[0] === 7) wsFrames.k7++; if (f.payload[0] === 1) wsFrames.k1++; }
  });
  ws.on("framesent", () => wsFrames.sent++);
});

const t0 = Date.now();
await page.goto(base, { waitUntil: "networkidle" });
await page.getByTestId("sel-agent-type").waitFor();
res.checks.n_agent_types = await page.locator('[data-testid="sel-agent-type"] option').count();
res.checks.stock_panel_hidden = (await page.getByTestId("stock-panel").count()) === 0;
res.checks.disclaimer_setup = await page.getByTestId("disclaimer").count();
await page.getByTestId("role-prompt-preview").waitFor();
await page.screenshot({ path: `${out}/01_setup.png`, fullPage: true });

await page.getByTestId("btn-connect").click();
await page.getByTestId("warmup-panel").waitFor({ timeout: 15000 });
res.t_warmup_panel_s = (Date.now() - t0) / 1000;
await page.screenshot({ path: `${out}/02_warmup.png`, fullPage: true });
while ((Date.now() - t0) / 1000 < maxWait) {
  if (await page.getByTestId("warmup-panel").count() === 0) break;
  const st = await page.getByTestId("warmup-state").getAttribute("data-state").catch(() => null);
  if (st && res.warmup_states[res.warmup_states.length - 1] !== st) res.warmup_states.push(st);
  await page.waitForTimeout(300);
}
res.checks.warmup_detail_seen = res.warmup_states.length > 0;
await page.getByTestId("phase").waitFor({ timeout: 15000 });
await page.getByText("Live: speak as the customer.").waitFor({ timeout: 30000 });
res.t_live_s = (Date.now() - t0) / 1000;
await page.waitForTimeout(3000);
await page.screenshot({ path: `${out}/03_live.png`, fullPage: true });
while ((Date.now() - t0) / 1000 < maxWait) {
  await page.waitForTimeout(1000);
  const phase = await page.getByTestId("phase").innerText();
  if (/Session ended|Disconnected/.test(phase)) break;
}
res.checks.phase_final = await page.getByTestId("phase").innerText();
res.checks.metrics_text = (await page.getByTestId("metrics-panel").innerText()).replace(/\s+/g, " ").slice(0, 300);
res.checks.text_segments = await page.getByTestId("text-seg").count();
res.checks.disclaimer_conv = await page.getByTestId("disclaimer").count();
await page.screenshot({ path: `${out}/04_end.png`, fullPage: true });
res.ws_frames = wsFrames;
res.session_requests = reqs.filter((v, i, a) => a.indexOf(v) === i);
res.wall_s = (Date.now() - t0) / 1000;
await browser.close();
fs.writeFileSync(`${out}/e2e_result.json`, JSON.stringify(res, null, 1));
console.log(JSON.stringify(res, null, 1));
