// T2 review: client vs the REAL GPU server. Checks role-prompt equality (preview == session event == /internal/session),
// inject rendering (amber spans == forced text events), mute in metrics, reconnect reset, second-tab behaviour.
// Usage: cd /root/e2e && node e2e_real.mjs BASE INTERNAL OUT RECORD_TYPE PAIRING WAV LIVE_S
import { chromium } from "/root/e2e/node_modules/playwright/index.mjs";
import fs from "node:fs";

const [base, internal, out, agentType, pairing, wav, liveS] = process.argv.slice(2);
fs.mkdirSync(out, { recursive: true });
const res = { base, agentType, pairing, wav, console_errors: [], console_unknown: 0, checks: {}, s1: {}, s2: {}, tab2: {} };
const browser = await chromium.launch({
  args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
    `--use-file-for-fake-audio-capture=${wav}`, "--autoplay-policy=no-user-gesture-required"],
});
const ctx = await browser.newContext({ ignoreHTTPSErrors: true, viewport: { width: 1440, height: 1000 } });
await ctx.grantPermissions(["microphone"], { origin: base });
const post = async (p, body) => (await fetch(internal + p, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) })).json();
const getj = async (p) => (await fetch(internal + p)).json();

const mkPage = async (tag) => {
  const page = await ctx.newPage();
  const st = { k7: 0, forced: 0, texts: 0, sessions: [], ends: [], actions: 0, injects: [], k2: 0, audio: 0 };
  page.on("console", m => {
    const t = m.text();
    if (m.type() === "error") res.console_errors.push(`${tag}: ` + t.slice(0, 300));
    if (/Unknown message/i.test(t)) res.console_unknown++;
  });
  page.on("pageerror", e => res.console_errors.push(`${tag} pageerror: ` + String(e).slice(0, 300)));
  page.on("websocket", ws => {
    ws.on("framereceived", f => {
      if (typeof f.payload === "string") return;
      const b = Buffer.from(f.payload);
      if (b[0] === 1) st.audio++;
      if (b[0] === 2) st.k2++;
      if (b[0] !== 7) return;
      st.k7++;
      const ev = JSON.parse(b.subarray(1).toString("utf8"));
      if (ev.type === "text") { st.texts++; if (ev.forced) st.forced++; }
      if (ev.type === "session") st.sessions.push(ev);
      if (ev.type === "session_end") st.ends.push(ev);
      if (ev.type === "action") st.actions++;
      if (ev.type === "inject") st.injects.push(ev);
    });
  });
  return { page, st };
};

const setup = async (page) => {
  await page.goto(base, { waitUntil: "networkidle" });
  await page.getByTestId("sel-agent-type").selectOption(agentType);
  const second = await page.locator('[data-testid="sel-record"] option').nth(1).getAttribute("value");
  await page.getByTestId("sel-record").selectOption(second);
  await page.getByTestId("sel-pairing").getByText(new RegExp(`^${pairing} `)).click();
  await page.waitForTimeout(800);
  return { record: second, preview: await page.getByTestId("role-prompt-preview").innerText() };
};

const t0 = Date.now();
// ---------------- session 1
const A = await mkPage("s1");
const s1 = await setup(A.page);
res.record_id = s1.record;
res.s1.preview_prompt = s1.preview;
await A.page.getByTestId("btn-connect").click();
const tc = Date.now();
await A.page.getByText("Live: speak as the customer.").waitFor({ timeout: 120000 });
res.s1.connect_to_live_s = (Date.now() - tc) / 1000;
await A.page.waitForTimeout(1500);
const isess = await getj("/internal/session");
res.s1.internal_session = { session_id: isess.session_id, record_id: isess.record_id, pairing: isess.pairing, voice: isess.voice, seed: isess.seed };
await A.page.getByTestId("session-panel").getByText("show role prompt").click();
const sessPrompt = await A.page.getByTestId("session-role-prompt").innerText();
const ev0 = A.st.sessions[0] ?? {};
res.checks.prompt_preview_eq_session_event = s1.preview === ev0.role_prompt;
res.checks.prompt_panel_eq_session_event = sessPrompt === ev0.role_prompt;
res.checks.prompt_eq_internal_session = isess.role_prompt === ev0.role_prompt;
res.s1.session_event = { ...ev0, role_prompt: undefined };
await A.page.screenshot({ path: `${out}/s1_live.png`, fullPage: true });

// inject a short phrase ~2 s ahead
const words = [[0, 0], ["aapka", 2], [0, 0], ["order", 2], [0, 0], ["ban", 2], [0, 0], ["raha", 2], [0, 0], ["hai", 4]];
res.s1.inject_plan = await post("/internal/inject", { words, start_frame: 0 });
await A.page.waitForTimeout(4000);
res.s1.mute_on = await post("/internal/mute", { on: true });
await A.page.waitForTimeout(2600);
res.s1.metrics_while_muted = (await A.page.getByTestId("metrics-panel").innerText()).replace(/\s+/g, " ").match(/inject \/ mute.{0,40}/)?.[0];
res.s1.mute_off = await post("/internal/mute", { on: false });
await A.page.waitForTimeout(1500);
res.s1.amber_spans = await A.page.locator('[data-testid="text-panel"] span.bg-amber-200').count();
res.s1.forced_events = A.st.forced;
res.checks.amber_eq_forced = res.s1.amber_spans === A.st.forced && A.st.forced > 0;
res.s1.forced_in_header = await A.page.getByTestId("text-panel").locator("h2 + span").innerText();
await A.page.screenshot({ path: `${out}/s1_after_inject.png`, fullPage: true });

// keep live until LIVE_S, watch for actions
const liveEnd = Date.now() + Number(liveS) * 1000;
let shot = false;
while (Date.now() < liveEnd) {
  await A.page.waitForTimeout(2000);
  if (!shot && (await A.page.getByTestId("action-group").count()) > 0) { await A.page.screenshot({ path: `${out}/s1_action.png`, fullPage: true }); shot = true; }
}
res.s1.action_groups_dom = await A.page.getByTestId("action-group").count();
res.s1.action_events_ws = A.st.actions;
res.s1.actions_header = await A.page.getByTestId("actions-panel").locator("h2 + span").innerText();
res.s1.text_segs = await A.page.getByTestId("text-seg").count();
res.s1.text_events = A.st.texts;
res.s1.metrics_text = (await A.page.getByTestId("metrics-panel").innerText()).replace(/\s+/g, " ").slice(0, 500);
res.s1.text_panel_header = await A.page.getByTestId("text-panel").locator("h2 + span").innerText();
await A.page.screenshot({ path: `${out}/s1_before_disconnect.png`, fullPage: true });
// disconnect
await A.page.getByRole("button", { name: "Disconnect" }).click();
await A.page.waitForTimeout(3000);
res.s1.phase_after_disconnect = await A.page.getByTestId("phase").innerText();
res.s1.session_end_ws = A.st.ends;
res.s1.download_link = await A.page.getByText("Download stereo recording").count();
await A.page.screenshot({ path: `${out}/s1_end.png`, fullPage: true });

// ---------------- session 2: "New Conversation" (reload) + reconnect in the same tab
await A.page.getByRole("button", { name: "New Conversation" }).click();
await A.page.getByTestId("btn-connect").waitFor({ timeout: 20000 });
const st1 = { ...A.st }; A.st.sessions = []; A.st.texts = 0; A.st.forced = 0; A.st.actions = 0; A.st.ends = [];
await A.page.getByTestId("sel-agent-type").selectOption(agentType);
await A.page.getByTestId("sel-record").selectOption(s1.record);
await A.page.getByTestId("sel-pairing").getByText(new RegExp(`^${pairing} `)).click();
await A.page.waitForTimeout(500);
await A.page.getByTestId("btn-connect").click();
await A.page.getByText("Live: speak as the customer.").waitFor({ timeout: 120000 });
await A.page.waitForTimeout(1200);
res.s2.text_segs_at_start = await A.page.getByTestId("text-seg").count();
res.s2.action_groups_at_start = await A.page.getByTestId("action-group").count();
res.s2.session_id = A.st.sessions[0]?.session_id;
res.checks.reconnect_new_session_id = !!res.s2.session_id && res.s2.session_id !== ev0.session_id;

// ---------------- second tab while session 2 is live
const B = await mkPage("tab2");
await setup(B.page);
await B.page.getByTestId("btn-connect").click();
await B.page.waitForTimeout(20000);
res.tab2.phase_after_20s = await B.page.getByTestId("phase").innerText();
res.tab2.k7 = B.st.k7;
res.tab2.audio = B.st.audio;
await B.page.screenshot({ path: `${out}/tab2_waiting.png`, fullPage: true });
res.s2.still_live = await A.page.getByTestId("phase").innerText();
await B.page.close();
await A.page.waitForTimeout(2000);
res.s2.text_events = A.st.texts;
await A.page.getByRole("button", { name: "Disconnect" }).click();
await A.page.waitForTimeout(3000);
res.s2.phase_after_disconnect = await A.page.getByTestId("phase").innerText();
res.s2.session_end_ws = A.st.ends;
res.s1_ws = { k7: st1.k7, k2: st1.k2, audio: st1.audio, injects: st1.injects };
res.wall_s = (Date.now() - t0) / 1000;
await browser.close();
fs.writeFileSync(`${out}/e2e_real_result.json`, JSON.stringify(res, null, 1));
console.log(JSON.stringify(res, null, 1));
