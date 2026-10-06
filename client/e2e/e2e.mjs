// DEP1 Track 2 headless browser test (Playwright Chromium on pod2, fake mic fed from a customer wav).
// Usage: cd /root/e2e && node /root/deploy/client/e2e/e2e.mjs [baseUrl] [outDir] [agentType] [pairing] [maxWaitS]
import { chromium } from "/root/e2e/node_modules/playwright/index.mjs";
import fs from "node:fs";

const base = process.argv[2] ?? "https://localhost:8998";
const out = process.argv[3] ?? "/root/deploy/client/e2e/out";
const agentType = process.argv[4] ?? "cab_ride_support";
const pairing = process.argv[5] ?? "g2";
const maxWait = Number(process.argv[6] ?? 70);
const wav = "/root/deploy/assets/ws/hinglish/tests/inputs/V3/cab_07_g3.wav";
fs.mkdirSync(out, { recursive: true });

const res = { base, agentType, pairing, console_errors: [], console_unknown: 0, ws_urls: [], checks: {} };
const browser = await chromium.launch({
  args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
    `--use-file-for-fake-audio-capture=${wav}`, "--autoplay-policy=no-user-gesture-required",
    "--ignore-certificate-errors"],
});
const ctx = await browser.newContext({ ignoreHTTPSErrors: true, viewport: { width: 1440, height: 1000 } });
await ctx.grantPermissions(["microphone"], { origin: base });
const page = await ctx.newPage();
page.on("console", m => {
  const t = m.text();
  if (m.type() === "error") res.console_errors.push(t.slice(0, 300));
  if (/Unknown message/i.test(t)) res.console_unknown++;
});
page.on("pageerror", e => res.console_errors.push("pageerror: " + String(e).slice(0, 300)));
let wsFrames = { recv: 0, sent: 0, k7: 0 };
page.on("websocket", ws => {
  res.ws_urls.push(ws.url());
  ws.on("framereceived", f => { wsFrames.recv++; if (typeof f.payload !== "string" && f.payload[0] === 7) wsFrames.k7++; });
  ws.on("framesent", () => wsFrames.sent++);
});

const t0 = Date.now();
await page.goto(base, { waitUntil: "networkidle" });
await page.getByTestId("sel-agent-type").waitFor();
res.checks.n_agent_types = await page.locator('[data-testid="sel-agent-type"] option').count();
await page.getByTestId("sel-agent-type").selectOption(agentType);
const recOpts = page.locator('[data-testid="sel-record"] option');
res.checks.n_records_for_type = await recOpts.count();
const second = await recOpts.nth(1).getAttribute("value");
await page.getByTestId("sel-record").selectOption(second);
res.record_id = second;
await page.getByTestId("sel-pairing").getByText(new RegExp(`^${pairing} `)).click();
await page.getByTestId("role-prompt-preview").waitFor();
await page.waitForTimeout(500);
res.checks.role_prompt_preview = (await page.getByTestId("role-prompt-preview").innerText()).slice(0, 160);
res.checks.voice_line = await page.getByText(/^Voice:/).innerText();
res.checks.samples_placeholders_setup = await page.getByTestId("sample-placeholder").count();
res.checks.samples_items_setup = await page.getByTestId("sample-item").count();
res.checks.disclaimer_setup = await page.getByTestId("disclaimer").count();
await page.screenshot({ path: `${out}/01_setup.png`, fullPage: true });

await page.getByTestId("btn-connect").click();
await page.getByTestId("phase").waitFor({ timeout: 15000 });
res.checks.phase_initial = await page.getByTestId("phase").innerText();
await page.screenshot({ path: `${out}/02_prompt_phase.png`, fullPage: true });
await page.getByText("Live: speak as the customer.").waitFor({ timeout: 30000 });
res.t_handshake_s = (Date.now() - t0) / 1000;

// wait for >= 2 action groups or the end of the session
let shot3 = false;
while ((Date.now() - t0) / 1000 < maxWait) {
  await page.waitForTimeout(2000);
  const groups = await page.getByTestId("action-group").count();
  if (!shot3 && groups >= 1) { await page.screenshot({ path: `${out}/03_live_action.png`, fullPage: true }); shot3 = true; }
  const phase = await page.getByTestId("phase").innerText();
  if (/Session ended|Disconnected/.test(phase)) break;
}
res.checks.phase_final = await page.getByTestId("phase").innerText();
res.checks.text_segments = await page.getByTestId("text-seg").count();
res.checks.text_panel_header = (await page.getByTestId("text-panel").locator("h2 + span").innerText()).slice(0, 120);
res.checks.action_groups = await page.getByTestId("action-group").count();
res.checks.actions_header = (await page.getByTestId("actions-panel").locator("h2 + span").innerText());
res.checks.metrics_text = (await page.getByTestId("metrics-panel").innerText()).replace(/\s+/g, " ").slice(0, 600);
res.checks.session_text = (await page.getByTestId("session-panel").innerText()).replace(/\s+/g, " ").slice(0, 400);
res.checks.disclaimer_conv = await page.getByTestId("disclaimer").count();
res.checks.download_link = await page.getByText("Download stereo recording").count();
await page.screenshot({ path: `${out}/04_end.png`, fullPage: true });
res.ws_frames = wsFrames;
res.wall_s = (Date.now() - t0) / 1000;
await browser.close();
fs.writeFileSync(`${out}/e2e_result.json`, JSON.stringify(res, null, 1));
console.log(JSON.stringify(res, null, 1));
