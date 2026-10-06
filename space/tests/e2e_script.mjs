// Headless check of the "Script: what to say" panel (D-SCRIPT-PANEL, 2026-10-06) + the top-right turn-filler toggle,
// against the Space backend + real worker_server --mock + tests/fake_runpod.py (run_e2e_script.sh starts them).
//   1. desktop 1440x1000, food_08 g1: panel right of the live text (same row), customer lines, tags, expected write,
//      split note; toggle Off -> /api/filler off and the ticker indicator goes "off"; Ticker -> back. Screenshots.
//   2. narrow 390x844, food_08 g1: panel stacked under the text, no horizontal page scroll. Screenshot.
//   3. narrow, food_18 g1 (call dropped at generation): "no script for this pairing". Screenshot.
// Usage: cd /root/e2e && node <this> [baseUrl] [outDir]
import { chromium } from "/root/e2e/node_modules/playwright/index.mjs";
import fs from "node:fs";

const base = process.argv[2] ?? "http://localhost:18870";
const out = process.argv[3] ?? "/tmp/out_script";
fs.mkdirSync(out, { recursive: true });
const res = { base, console_errors: [], checks: {} };
const browser = await chromium.launch({ args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
  "--autoplay-policy=no-user-gesture-required"] });

async function startCall(viewport, agentType, recordId, pairing) {
  const ctx = await browser.newContext({ viewport });
  await ctx.grantPermissions(["microphone"], { origin: base });
  const page = await ctx.newPage();
  page.on("pageerror", e => res.console_errors.push("pageerror: " + String(e).slice(0, 300)));
  await page.goto(base, { waitUntil: "networkidle" });
  await page.getByTestId("sel-agent-type").selectOption(agentType);
  await page.getByTestId("sel-record").selectOption(recordId);
  await page.getByTestId("sel-pairing").getByRole("button", { name: new RegExp(`^${pairing} `) }).click();
  await page.getByTestId("btn-connect").click();
  await page.getByText("Live: speak as the customer.").waitFor({ timeout: 120000 });
  return { ctx, page };
}

async function endCall(ctx, page) {
  await page.getByRole("button", { name: "Disconnect" }).click().catch(() => undefined);
  await page.waitForTimeout(1500);
  await ctx.close();
}

// ---- 0. defaults (2026-10-06): food_23 g1 preselected, research-demo disclaimer
{
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  await ctx.grantPermissions(["microphone"], { origin: base });
  const page = await ctx.newPage();
  page.on("pageerror", e => res.console_errors.push("pageerror: " + String(e).slice(0, 300)));
  await page.goto(base, { waitUntil: "networkidle" });
  await page.getByTestId("samples-panel").waitFor({ timeout: 15000 });
  res.checks.default_agent_type = await page.getByTestId("sel-agent-type").inputValue();
  res.checks.default_record = await page.getByTestId("sel-record").inputValue();
  const dis = await page.getByTestId("disclaimer").first().innerText();
  res.checks.disclaimer_v4 = dis.includes("V4_A2 step 600") && dis.includes("30 held-out calls") && !dis.includes("step 200");
  res.checks.disclaimer_compound = dis.includes("compound calls");
  await page.screenshot({ path: `${out}/defaults_setup.png`, fullPage: true });
  await page.getByTestId("btn-connect").click();                     // connect with the defaults (food_23, g1)
  await page.getByText("Live: speak as the customer.").waitFor({ timeout: 120000 });
  await page.getByTestId("script-panel").getByTestId("script-customer").first().waitFor({ timeout: 15000 });
  res.checks.default_call_note = (await page.getByTestId("script-note").innerText()).includes("food_23_g1");
  res.checks.default_split_badge_absent = (await page.getByTestId("script-split").count()) === 0;
  await page.waitForTimeout(6000);
  await page.screenshot({ path: `${out}/defaults_call.png`, fullPage: true });
  await endCall(ctx, page);
}

// ---- 1. desktop
{
  await fetch(base + "/api/filler?mode=ticker", { method: "POST" });
  const { ctx, page } = await startCall({ width: 1440, height: 1000 }, "food_delivery_support", "food_08", "g1");
  const panel = page.getByTestId("script-panel");
  await panel.getByTestId("script-customer").first().waitFor({ timeout: 15000 });
  const tb = await page.getByTestId("text-panel").boundingBox();
  const sb = await panel.boundingBox();
  res.checks.desktop_same_row_right = !!(tb && sb && sb.x > tb.x + tb.width - 2 && Math.abs(sb.y - tb.y) < 4);
  res.checks.n_customer = await panel.getByTestId("script-customer").count();
  res.checks.n_agent = await panel.getByTestId("script-agent").count();
  res.checks.first_customer_text = (await panel.getByTestId("script-customer").first().innerText()).slice(0, 120);
  res.checks.tag_check = await panel.getByTestId("script-tag-check_line").count();
  res.checks.tag_write = await panel.getByTestId("script-tag-confirm_write").count();
  res.checks.write_shown = (await panel.innerText()).includes("change_delivery_address(");
  res.checks.split_badge_removed = (await page.getByTestId("script-split").count()) === 0;   // 2026-10-06: badge removed
  res.checks.note = (await page.getByTestId("script-note").innerText()).slice(0, 200);
  res.checks.n_next = await panel.locator("[data-next='1']").count();
  await page.waitForTimeout(12000);                                  // let the mock text stream run
  await page.screenshot({ path: `${out}/script_desktop.png` });
  // the top-right toggle (public/filler-toggle.js) is still there and still works
  const tog = page.locator("body > div").filter({ hasText: "Turn filler:" });
  res.checks.toggle_present = (await tog.count()) === 1;
  await tog.getByRole("button", { name: "Off" }).click();
  await page.waitForTimeout(500);
  res.checks.toggle_off_api = (await (await fetch(base + "/api/filler")).json()).mode === "off";
  let st = null;
  for (let i = 0; i < 60 && st !== "off"; i++) {
    st = await page.getByTestId("ticker-indicator").getAttribute("data-state").catch(() => null);
    await page.waitForTimeout(100);
  }
  res.checks.indicator_off_after_toggle = st === "off";
  await page.screenshot({ path: `${out}/script_desktop_toggle_off.png` });
  await tog.getByRole("button", { name: "Ticker" }).click();
  await page.waitForTimeout(500);
  res.checks.toggle_back_ticker = (await (await fetch(base + "/api/filler")).json()).mode === "ticker";
  await endCall(ctx, page);
}

// ---- 2. narrow, script available
{
  const { ctx, page } = await startCall({ width: 390, height: 844 }, "food_delivery_support", "food_08", "g1");
  const panel = page.getByTestId("script-panel");
  await panel.getByTestId("script-customer").first().waitFor({ timeout: 15000 });
  const tb = await page.getByTestId("text-panel").boundingBox();
  const sb = await panel.boundingBox();
  res.checks.narrow_stacked = !!(tb && sb && sb.y >= tb.y + tb.height - 2);
  res.checks.narrow_no_hscroll = await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1);
  await page.waitForTimeout(4000);
  await page.screenshot({ path: `${out}/script_narrow.png`, fullPage: true });
  await panel.scrollIntoViewIfNeeded();
  await page.screenshot({ path: `${out}/script_narrow_panel.png` });
  await endCall(ctx, page);
}

// ---- 3. narrow, dropped call
{
  const { ctx, page } = await startCall({ width: 390, height: 844 }, "food_delivery_support", "food_18", "g1");
  await page.getByTestId("script-none").waitFor({ timeout: 15000 });
  res.checks.none_text = (await page.getByTestId("script-none").innerText()).slice(0, 120);
  await page.getByTestId("script-panel").scrollIntoViewIfNeeded();
  await page.screenshot({ path: `${out}/script_none_narrow.png` });
  await endCall(ctx, page);
}

await browser.close();
fs.writeFileSync(`${out}/script_result.json`, JSON.stringify(res, null, 1));
console.log(JSON.stringify(res, null, 1));
const c = res.checks;
const ok = c.default_agent_type === "food_delivery_support" && c.default_record === "food_23" && c.disclaimer_v4 &&
  c.disclaimer_compound && c.default_call_note && c.default_split_badge_absent && c.desktop_same_row_right && c.n_customer >= 3 && c.n_agent >= 3 && c.tag_check >= 1 && c.tag_write >= 1 &&
  c.write_shown && c.split_badge_removed && c.n_next === 1 && c.toggle_present && c.toggle_off_api &&
  c.indicator_off_after_toggle && c.toggle_back_ticker && c.narrow_stacked && c.narrow_no_hscroll &&
  /No script for this pairing/.test(c.none_text ?? "") && res.console_errors.length === 0;
console.log(ok ? "SCRIPT UI OK" : "SCRIPT UI FAILED");
process.exit(ok ? 0 : 1);
