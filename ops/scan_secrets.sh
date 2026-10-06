#!/bin/bash
# Secret + size scan of one or more trees before a push (from the old make_repos.sh; D-MONOREPO 2026-10-06).
#   bash ops/scan_secrets.sh DIR [DIR...]      exit 1 on any problem; never prints a secret value
# Checks: files > 50 MB; token-shaped strings (HF, RunPod, GitHub, OpenAI/Anthropic, Docker Hub PAT, private keys);
# forbidden files (.env, .env.local, secrets.env, cert.pem, key.pem); and the REAL secret values (read, never printed)
# from ~/.config/s2s/secrets.env (all values, incl. the Space passcode), RUNPOD_API_KEY*/DOCKERHUB_TOKEN in
# ~/Desktop/S2S/.env, and the HF token in /workspace/hf/token when readable (pod1).
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: scan_secrets.sh DIR [DIR...]" >&2; exit 2; }
bad=0
for R in "$@"; do
  n=$(find "$R" -type f ! -path '*/.git/*' ! -path '*/node_modules/*' | wc -l); sz=$(du -sh --exclude=.git --exclude=node_modules "$R" | cut -f1)
  echo "$R: $n files, $sz"
  big=$(find "$R" -type f ! -path '*/.git/*' ! -path '*/node_modules/*' -size +50M)
  [ -z "$big" ] || { echo "  FILES > 50 MB (GitHub warns at 50, rejects at 100): $big"; bad=1; }
  hits=$(grep -rIEn --exclude-dir=.git --exclude-dir=node_modules \
    -e 'hf_[A-Za-z0-9]{30,}' -e 'rpa_[A-Za-z0-9]{20,}' -e 'gh[pousr]_[A-Za-z0-9]{30,}' -e 'sk-(ant-)?[A-Za-z0-9_-]{20,}' -e 'dckr_pat_[A-Za-z0-9_-]{10,}' \
    -e 'BEGIN [A-Z ]*PRIVATE KEY' "$R" || true)
  [ -z "$hits" ] || { echo "  TOKEN-LIKE STRINGS:"; echo "$hits" | cut -c1-160; bad=1; }
  grep -rIEn --exclude-dir=.git --exclude-dir=node_modules -e '(RUNPOD_API_KEY|S2S_SESSION_SECRET|HF_TOKEN|S2S_PASSCODE)\s*=\s*["'"'"']?[A-Za-z0-9_-]{16,}' "$R" \
    | cut -c1-160 | sed 's/^/  review (test fixtures use obvious fake values): /' || true
  for f in .env .env.local secrets.env cert.pem key.pem; do
    found=$(find "$R" -name "$f" ! -path '*/.git/*' ! -path '*/node_modules/*')
    [ -z "$found" ] || { echo "  FORBIDDEN FILE: $found"; bad=1; }
  done
done
PATS=$(mktemp); chmod 600 "$PATS"; trap 'rm -f "$PATS"' EXIT
{ [ -f "$HOME/.config/s2s/secrets.env" ] && sed -n 's/^[A-Z0-9_]*=//p' "$HOME/.config/s2s/secrets.env" | tr -d '"'"'"' \r'
  [ -f "$HOME/Desktop/S2S/.env" ] && sed -n -E 's/^(RUNPOD_API_KEY[A-Z0-9_]*|DOCKERHUB_TOKEN[A-Z0-9_]*)=//p' "$HOME/Desktop/S2S/.env" | tr -d '"'"'"' \r'
  [ -r /workspace/hf/token ] && head -n1 /workspace/hf/token | tr -d ' \r'; } 2>/dev/null | grep -E '.{8,}' > "$PATS" || true
# the Space passcode is short (< 8 chars), so it is added on its own (any length >= 4)
[ -f "$HOME/.config/s2s/secrets.env" ] && sed -n 's/^S2S_PASSCODE=//p' "$HOME/.config/s2s/secrets.env" | tr -d '"'"'"' \r' | grep -E '.{4,}' >> "$PATS" || true
if [ -s "$PATS" ]; then
  for R in "$@"; do
    n=$( (grep -rlF --exclude-dir=.git --exclude-dir=node_modules -f "$PATS" "$R" || true) | wc -l)
    [ "$n" = 0 ] || { echo "  $R: $n FILE(S) CONTAIN A REAL SECRET VALUE"; bad=1; }
  done
  echo "real-secret scan: $(wc -l < "$PATS") value(s) checked"
else
  echo "real-secret scan: no local secret values found to check against (token-shape scan only)"
fi
[ $bad = 0 ] && echo "scan: OK (no token-like strings, no forbidden files, no file > 50 MB)" || { echo "scan: PROBLEMS (see above)"; exit 1; }
