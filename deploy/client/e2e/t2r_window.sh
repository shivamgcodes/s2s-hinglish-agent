#!/bin/bash
# T2-review GPU window: waits for /root/gpu.lock, declares the window, runs the real server on 9978/9979.
export HF_HOME=/root/hf HF_HUB_OFFLINE=1 DEP1_TORCH_THREADS=4
cd /root/deploy/server
s=$(date -u +%FT%TZ); e=$(date -u -d "+20 min" +%FT%TZ)
printf "\n## D-T2R-W %s T2-review: declared GPU window\n- Owner: Track 2 adversarial reviewer. Start %s, expected end %s. Real server (server.py) on test ports 9978 (https) / 9979 (internal), static = client/dist; Trelis asr_service on 8976 and router_service on 8975 (no lock, D-T3-1 arrangement); Playwright e2e_real.mjs + ws_check.py.\n" "$s" "$s" "$e" >> /root/deploy/DECISIONS.md
exec /root/deploy/venv-pp/bin/python -u server.py --port 9978 --internal-port 9979 --session-log /root/deploy/logs/t2review_sessions.jsonl
