import { chromium } from "/root/e2e/node_modules/playwright/index.mjs";
const browser = await chromium.launch({ args: ["--use-fake-ui-for-media-stream","--use-fake-device-for-media-stream"] });
const ctx = await browser.newContext({ ignoreHTTPSErrors: true });
await ctx.grantPermissions(["microphone"], { origin: "https://localhost:9998" });
const out = {};
for (const typed of ["-1", "", "-", "42"]) {
  const page = await ctx.newPage();
  const urls = [];
  page.on("websocket", ws => urls.push(ws.url()));
  await page.goto("https://localhost:9998", { waitUntil: "networkidle" });
  await page.getByTestId("seed").fill("");
  if (typed) await page.getByTestId("seed").pressSequentially(typed);
  out[typed || "(empty)"] = { field: await page.getByTestId("seed").inputValue() };
  await page.getByTestId("btn-connect").click();
  await page.waitForTimeout(1500);
  out[typed || "(empty)"].seed_param = urls.length ? new URL(urls[0]).searchParams.get("seed") : null;
  await page.close();
  await new Promise(r => setTimeout(r, 4000));
}
console.log(JSON.stringify(out));
await browser.close();
