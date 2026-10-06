# Serverless client fork (DESIGN.md 5.4)

Fork of `/root/deploy/client` (DEP1; see README_DEP1.md). Build on runpod2 only: `npm ci && npm run build`. The
output in `client/dist` is copied to `space/static` (`ops/build_client.sh`). `.env.local` is unchanged
(`VITE_QUEUE_API_PATH=/api`).

Changes vs DEP1:
- `pages/Conversation/Conversation.tsx`
  - `buildURL` uses `window.location.host`. DEP1 used `hostname + ":" + port`, which gives `host:` on hf.space.
  - It appends `sid`.
  - The phase line shows the `session_end` reason, with text for `time_limit`, `worker_lost`, `worker_shutdown` and
    `relay_error`, plus the relay's `detail`.
  - Disconnect and "New Conversation" also send `DELETE /api/session/{sid}`.
- `pages/Queue/Queue.tsx`: Connect does four things in order:
  1. gets the AudioContext and microphone (user gesture);
  2. sends `POST /api/session`;
  3. shows the `dep1/Warmup.tsx` panel, which polls `GET /api/session/{sid}` every 1.5 s and has a Cancel button that
     sends DELETE;
  4. mounts `<Conversation>` only at `ready`.
  A `pagehide` handler sends DELETE (`fetch` with `keepalive`).
- `dep1/SessionSetup.tsx`
  - A passcode field appears when `/api/config.passcode_required` is set.
  - Errors from `POST /api/session` are shown (403, 429 capacity or rate, 400, 503).
  - The stock free-prompt panel is shown only when `/api/config.allow_free_prompt` is set.
- `dep1/Panels.tsx`
  - The latency panel reads the socket's 0x07 `metrics` events. It no longer polls HTTP `/metrics` every 1 s.
  - VRAM is shown without the hard-coded 24 GB total.
  - The samples hint points to `space/samples/`.
- `dep1/api.ts`, `dep1/types.ts`: the Space session API (`createSession`, `getSession`, `deleteSession`,
  `fetchConfig`, `fetchMetrics(sid)`) and `END_REASON_TEXT`.

Unchanged: the protocol (byte-identical to DEP1), the audio path, the text, actions and session panels, and the
disclaimer.
