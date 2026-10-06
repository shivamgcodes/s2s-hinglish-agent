# DEP1 web client (Track 2)

Fork of assets/personaplex/client (stock Vite/React). Build output: client/dist (served by server/server.py on :8998).

Build (pod2, Node v24.21.0): cd /root/deploy/client && npm ci && npm run build   (.env.local = VITE_QUEUE_API_PATH=/api only)

Changes vs stock:
- src/protocol/{types,encoder}.ts: kind 0x07 decoded as {type:"event", data:JSON}; unknown kinds no longer throw.
- src/pages/Queue/Queue.tsx: stock homepage replaced by src/dep1/SessionSetup.tsx (agent type -> demo record ->
  pairing g1..g4 -> voice NATF2/NATM1; role prompt preview = server session_configs[gN] from /api/records/{id};
  record facts; seed). Stock free-prompt session kept in a collapsed panel.
- src/pages/Conversation/Conversation.tsx: ws URL adds record_id, pairing, agent_type, seed, events=1 and sends
  text_prompt= and voice_prompt= EMPTY (server builds them, INTERFACE.md section 3). New layout with
  src/dep1/Panels.tsx: live text stream (0x07 text events, forced tokens amber, trigger markers; 0x02 fallback),
  actions panel (grouped by trigger_id, per-stage latency_ms / since_trigger_ms), latency readout (/metrics poll 1 s),
  session panel, handpicked samples (/api/samples; 5 placeholders when empty), disclaimer, session_end reason.
- No backend logic beyond display and session config.

Test-only mock (no GPU): client/mock/mock_server.py (venv-pp; https, self-signed cert generated into mock/certs/).
  /root/deploy/venv-pp/bin/python client/mock/mock_server.py --port 8998   # replays tests/out/V3_A/V3 calls,
  scripted [mock] router chains (executed / unbound / error). Protocol check: client/mock/ws_check.py.
Browser test: Playwright chromium in /root/e2e: cd /root/e2e && node /root/deploy/client/e2e/e2e.mjs https://localhost:8998 OUTDIR [agent_type] [pairing] [max_wait_s]
Laptop: ssh -N -L 8998:localhost:8998 runpod2  then open https://localhost:8998 (accept the self-signed cert).
Playwright lives outside /root/deploy (not persisted). Reinstall after a pod reset:
  mkdir -p /root/e2e && cd /root/e2e && npm init -y && npm i playwright@1 && npx playwright install --with-deps chromium
Results of the Track 2 runs: client/e2e/out (cab_02 g2, client mock), out2 (food_02 g1, with a temp sample), out_t1mock and
out_t1mock_action (against Track 1 server/mock_server.py on :9998; actions POSTed to /internal/action by curl).
