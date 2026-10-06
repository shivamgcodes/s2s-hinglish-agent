// Headless check of the ticker indicator (2026-10-06): Space backend + real worker_server --mock (MockEngine replay,
// real TurnFiller -> 0x07 filler events) + tests/fake_runpod.py. run_e2e_ticker.sh starts them and runs this.
// Samples the indicator's data-state every 100 ms, switches the filler Off mid-call via POST /api/filler and checks
// the indicator goes "off", then back to ticker. Screenshots: ticker_filling.png, ticker_off.png.
// Usage: cd /root/e2e && node <this> [baseUrl] [outDir] [callS]
import { chromium } from "/root/e2e/node_modules/playwright/index.mjs";
import fs from "node:fs";

const base = process.argv[2] ?? "http://localhost:18860";
const out = process.argv[3] ?? "/tmp/out_ticker";
const callS = Number(process.argv[4] ?? 40);
fs.mkdirSync(out, { recursive: true });
const res = { base, states: [], console_errors: [], checks: {} };
const browser = await chromium.launch({ args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
  "--autoplay-policy=no-user-gesture-required"] });
const ctx = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
await ctx.grantPermissions(["microphone"], { origin: base });
const page = await ctx.newPage();
page.on("pageerror", e => res.console_errors.push("pageerror: " + String(e).slice(0, 300)));
let filler7 = 0;
page.on("websocket", ws => ws.on("framereceived", f => {
  if (typeof f.payload !== "string" && f.payload[0] === 7 && Buffer.from(f.payload.slice(1)).toString().includes('"type": "filler"')) filler7++;
}));
await fetch(base + "/api/filler?mode=ticker", { method: "POST" });
await page.goto(base, { waitUntil: "networkidle" });
await page.getByTestId("sel-agent-type").waitFor();
await page.getByTestId("btn-connect").click();
await page.getByText("Live: speak as the customer.").waitFor({ timeout: 90000 });
res.checks.indicator_present = await page.getByTestId("ticker-indicator").count();
res.checks.label = (await page.locator("text=ticker (model only)").count()) > 0;
const t0 = Date.now();
let shotFill = false, offAt = null, backAt = null;
while ((Date.now() - t0) / 1000 < callS) {
  const st = await page.getByTestId("ticker-indicator").getAttribute("data-state").catch(() => null);
  const t = (Date.now() - t0) / 1000;
  if (st && (!res.states.length || res.states[res.states.length - 1][1] !== st)) res.states.push([+t.toFixed(1), st]);
  if (st === "filling" && !shotFill) { await page.screenshot({ path: `${out}/ticker_filling.png` }); shotFill = true; }
  if (offAt === null && t > callS * 0.55) { await fetch(base + "/api/filler?mode=off", { method: "POST" }); offAt = t; }
  if (offAt !== null && backAt === null && t > offAt + 5) {
    await page.screenshot({ path: `${out}/ticker_off.png` });
    await fetch(base + "/api/filler?mode=ticker", { method: "POST" }); backAt = t;
  }
  await page.waitForTimeout(100);
}
const seen = new Set(res.states.map(s => s[1]));
res.checks.saw_filling = seen.has("filling");
res.checks.saw_idle = seen.has("idle");
res.checks.saw_off_after_toggle = res.states.some(([t, s]) => s === "off" && t >= offAt);
res.checks.back_from_off = res.states.some(([t, s]) => s !== "off" && t > backAt);
res.checks.filler_frames = filler7;
res.off_at = offAt; res.back_at = backAt;
await browser.close();
fs.writeFileSync(`${out}/ticker_result.json`, JSON.stringify(res, null, 1));
console.log(JSON.stringify(res, null, 1));
const ok = res.checks.indicator_present && res.checks.label && res.checks.saw_filling && res.checks.saw_idle &&
  res.checks.saw_off_after_toggle && res.checks.back_from_off && res.console_errors.length === 0;
console.log(ok ? "TICKER UI OK" : "TICKER UI FAILED");
process.exit(ok ? 0 : 1);
