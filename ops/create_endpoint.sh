#!/bin/bash
# S2S serverless: create the RunPod endpoint (LB primary or queue fallback) through the RunPod REST API v2.
# DRY RUN BY DEFAULT: prints the request (secrets masked) and the console steps. Nothing is sent without --yes.
#
#   bash ops/create_endpoint.sh [--mode lb|queue] [--max-workers 2] [--dc AP-JP-1[,US-TX-3]] [--name NAME]
#                               [--secret-ref] [--list-gpus] [--yes]
#   env: IMAGE (+TAG, default v1)  S2S_SESSION_SECRET  S2S_AUDIENCE (default hinglish-lb-1 / hinglish-q-1)
#        RUNPOD_API_KEY (only with --yes / --list-gpus)  RUNPOD_REGISTRY_AUTH_ID (private image only)
#
# API: POST https://api.runpod.io/v2/serverless  ([API-V2] in ops/ENDPOINT_SETTINGS.md). REST v1 (rest.runpod.io/v1)
# has no endpoint-type field, so it cannot create an LB endpoint; GraphQL saveEndpoint(type:"LB") is the other option
# (it takes the key in the URL query, so it is not used here).
# NOT settable by any documented API (2026-10-04): the cached Model (nvidia/personaplex-7b-v1 + HF token). This script
# prints that manual console step; the endpoint cannot load the model until you do it.
# --secret-ref puts {{ RUNPOD_SECRET_s2s_session_secret }} in the env instead of the value (documented for Pods only;
#   unverified for serverless; create the RunPod secret "s2s_session_secret" first).
# --list-gpus: read-only GET of the GPU catalog to confirm the pool ids (needs RUNPOD_API_KEY; sends a GET only).
set -euo pipefail
MODE=lb; MAXW=2; DC=""; NAME=""; YES=0; SECRET_REF=0; LIST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --mode) MODE=$2; shift 2 ;;
    --max-workers) MAXW=$2; shift 2 ;;
    --dc) DC=$2; shift 2 ;;
    --name) NAME=$2; shift 2 ;;
    --secret-ref) SECRET_REF=1; shift ;;
    --list-gpus) LIST=1; shift ;;
    --yes) YES=1; shift ;;
    -h|--help) sed -n 2,20p "$0"; exit 0 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
case "$MODE" in lb|queue) ;; *) echo "--mode must be lb or queue" >&2; exit 2 ;; esac
API=${RUNPOD_API_V2:-https://api.runpod.io/v2}

if [ $LIST = 1 ]; then
  echo "# read-only: GPU catalog (pool ids)"
  echo "curl -s -H \"Authorization: Bearer \$RUNPOD_API_KEY\" $API/gpu-types"
  if [ $YES = 1 ]; then
    [ -n "${RUNPOD_API_KEY:-}" ] || { echo "RUNPOD_API_KEY not set" >&2; exit 2; }
    curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" "$API/gpu-types"; echo
  else
    echo "# (dry run; add --yes to send this GET)"
  fi
  exit 0
fi

IMAGE_REF="${IMAGE:-<registry>/s2s-worker}:${TAG:-v1}"
[ -n "$NAME" ] || NAME="s2s-hinglish-$MODE"
AUD_DEFAULT=hinglish-lb-1; [ "$MODE" = queue ] && AUD_DEFAULT=hinglish-q-1
export CE_MODE=$MODE CE_MAXW=$MAXW CE_DC=$DC CE_NAME=$NAME CE_IMAGE=$IMAGE_REF CE_SECRET_REF=$SECRET_REF \
       CE_AUD=${S2S_AUDIENCE:-$AUD_DEFAULT} CE_REG=${RUNPOD_REGISTRY_AUTH_ID:-}

body() {  # $1 = mask|real ; prints the JSON body
  CE_SHOW=$1 python3 - <<'PY'
import json, os
mode = os.environ["CE_MODE"]
sec = os.environ.get("S2S_SESSION_SECRET", "")
if os.environ["CE_SECRET_REF"] == "1":
    sec_val = "{{ RUNPOD_SECRET_s2s_session_secret }}"
elif os.environ["CE_SHOW"] == "mask":
    sec_val = f"<S2S_SESSION_SECRET: {'set, %d chars' % len(sec) if sec else 'NOT SET'}>"
else:
    sec_val = sec
env = {"S2S_MODE": mode, "S2S_SESSION_SECRET": sec_val, "S2S_AUDIENCE": os.environ["CE_AUD"],
       "CALL_MAX_S": "300", "CLAIM_TTL_S": "90"}
if mode == "lb":
    env.update({"PORT": "80", "PORT_HEALTH": "80"})   # PORT_HEALTH explicit: runpod/docs#853
else:
    # job timer runs through the model load: exec 900 = load 420 + claim retry 30 + claim TTL 90 + call 300 + 30 + 30
    # (REVIEW-ops-3; rp_handler derives the load cap from S2S_QUEUE_EXEC_TIMEOUT_S, REVIEW-worker-3, set both explicitly)
    env.update({"PORT": "8765", "S2S_QUEUE_EXEC_TIMEOUT_S": "900", "S2S_QUEUE_LOAD_TIMEOUT_S": "420"})
b = {
    "name": os.environ["CE_NAME"],
    "type": "LOAD_BALANCER" if mode == "lb" else "QUEUE",
    "image": os.environ["CE_IMAGE"],
    # DESIGN 6.1 priority; AMPERE_24 (L4/A5000/3090) deliberately absent (R4). Confirm ids with --list-gpus.
    "gpu": {"pools": ["AMPERE_48", "ADA_24", "ADA_48_PRO"], "count": 1, "minCudaVersion": "13.0"},
    "workers": {"min": 0, "max": int(os.environ["CE_MAXW"]), "idleTimeout": 120},
    # LB: one worker per in-flight call. Queue: QUEUE_DELAY, because API-V2 says idleTimeout is "not applicable to
    # queue-based endpoints scaling on requestCount" (we want the 120 s warm tail there too).
    "scaling": ({"type": "REQUEST_COUNT", "requestCount": 1} if mode == "lb" else {"type": "QUEUE_DELAY", "queueDelay": 1}),
    "timeout": 330000 if mode == "lb" else 900000,   # queue: REVIEW-ops-3 arithmetic in ENDPOINT_SETTINGS.md 2
    "flashboot": "FLASHBOOT",                         # API default is OFF
    "disk": 40,
    "env": env,
    "ports": ["80/http"] if mode == "lb" else ["8765/tcp"],
}
if os.environ["CE_DC"]:
    b["dataCenterIds"] = [x for x in os.environ["CE_DC"].split(",") if x]
if os.environ["CE_REG"]:
    b["registry"] = os.environ["CE_REG"]
print(json.dumps(b, indent=1))
PY
}

echo "# ---- RunPod endpoint ($MODE) ----------------------------------------------------------------"
echo "# POST $API/serverless   (Authorization: Bearer \$RUNPOD_API_KEY)"
body mask
cat <<EOF
# equivalent curl (the key stays in your shell variable; nothing is echoed):
#   curl -sS -X POST $API/serverless -H "Authorization: Bearer \$RUNPOD_API_KEY" \\
#        -H "Content-Type: application/json" --data @body.json
#
# ---- MANUAL STEP after creation (no API field for it) -----------------------------------------
#  RunPod console -> Serverless -> $NAME -> Manage -> Edit endpoint:
#   1. Model: nvidia/personaplex-7b-v1   + Hugging Face access token (read-only, fine-grained, gated repo)
#      ([CACHE]: one cached model per endpoint; do NOT attach a network volume too)
#   2. Check: GPU priority A6000/A40 > 4090 PRO > L40/L40S (no L4/A5000/3090 pool), CUDA >= 13.0,
#      idle timeout 120 s, FlashBoot on, max workers $MAXW$( [ "$MODE" = queue ] && echo ", Expose TCP port 8765, execution timeout 900 s")
#   3. Space variables: RUNPOD_ENDPOINT_ID=<id>  S2S_MODE=$MODE  S2S_AUDIENCE=$CE_AUD  MAX_CONCURRENT_CALLS=$MAXW
EOF

if [ $YES != 1 ]; then
  echo "# DRY RUN: nothing sent. Re-run with --yes (and IMAGE, S2S_SESSION_SECRET, RUNPOD_API_KEY set) to create it."
  exit 0
fi
miss=""
[ -n "${IMAGE:-}" ] || miss="$miss IMAGE"
[ -n "${RUNPOD_API_KEY:-}" ] || miss="$miss RUNPOD_API_KEY"
SEC=${S2S_SESSION_SECRET:-}
[ $SECRET_REF = 1 ] || [ ${#SEC} -ge 32 ] || miss="$miss S2S_SESSION_SECRET(>=32 chars)"
[ -z "$miss" ] || { echo "refusing --yes: missing$miss" >&2; exit 2; }
command -v curl >/dev/null || { echo "curl not found" >&2; exit 2; }
TMPB=$(mktemp); chmod 600 "$TMPB"; trap 'rm -f "$TMPB"' EXIT
body real > "$TMPB"
echo "# sending ..."
RESP=$(curl -sS -X POST "$API/serverless" -H "Authorization: Bearer $RUNPOD_API_KEY" \
       -H "Content-Type: application/json" --data @"$TMPB" -w '\n%{http_code}')
CODE=${RESP##*$'\n'}; RESP=${RESP%$'\n'*}
echo "HTTP $CODE"
echo "$RESP" | python3 -c "import json,sys
t=sys.stdin.read()
try:
  d=json.loads(t)
except Exception:
  print(t[:2000]); sys.exit()
if isinstance(d,dict) and isinstance(d.get('env'),dict):
  d['env']={k:('<masked>' if 'SECRET' in k else v) for k,v in d['env'].items()}
print(json.dumps(d,indent=1)); print('ENDPOINT_ID =', d.get('id') if isinstance(d,dict) else None)"
[ "${CODE:0:1}" = 2 ] || exit 1
