# S2S serverless: create the HF Space and push the front (for the user; NOT executed by the build)

> **2026-10-06 (D-MONOREPO):** the Space contents are no longer a separate repo. `bash ops/push_space.sh --build-client`
> stages `space/` + `common/` + `client/dist` from the monorepo (`ops/stage_space.sh`), scans it and uploads it (`--yes`).
> The manual `s2s-space` git steps below are historical.

Sources: [HF-DOCKER] https://huggingface.co/docs/hub/spaces-sdks-docker, [HF-OV] https://huggingface.co/docs/hub/spaces-overview,
[HF-CFG] https://huggingface.co/docs/hub/spaces-config-reference (fetched 2026-10-04 by the architect; DESIGN.md 0 and 5.2).

## 0. Before you start

- A **Docker-SDK Space needs a paid plan** (PRO for a personal account). Free accounts get Static Spaces plus up to
  2 ZeroGPU Gradio Spaces ([HF-OV]). Without PRO, use launcher (b) (section 5, untested) or launcher (c) (section 6).
- Create a **write** token for the push (huggingface.co → Settings → Access Tokens). Use it only on your machine, and
  never put it in the Space, the repo or the RunPod endpoint.
- Have the endpoint id ready (ops/create_endpoint.sh output, or the console), plus `S2S_SESSION_SECRET` and the
  RunPod API key (ENV.md).

## 1. Stage the folder (on runpod2, where the client is built)

```bash
cd /root/deploy_serverless
bash ops/build_client.sh            # client/dist -> space/static  (or --copy-only if client/dist is current)
bash space/stage_common.sh          # ../common/{s2s_token.py,session.py,data/records_v4.json} -> space/common
ls space/static/index.html space/common/s2s_token.py space/common/data/records_v4.json
```

Copy `space/` to the machine you push from, for example the laptop:
`rsync -a --exclude tests/ --exclude __pycache__ runpod2:/root/deploy_serverless/space/ ./s2s-space/`.
The Space repo is the **contents** of `space/`, without `tests/`, because `.dockerignore` drops it from the image anyway.

## 2. Create the Space (web UI, simplest)

1. huggingface.co → New → Space.
2. Owner `<you>`, name e.g. `hinglish-agent`, SDK **Docker** (blank template), hardware **CPU basic**.
3. **Visibility: Public**, or **Protected** if your plan offers it. **Never Private**: a private Space returns 404 to
   visitors and to their websockets (R1, DESIGN 5.2).

Or use the CLI (`pip install -U huggingface_hub`; uses the write token from `hf auth login`):

```bash
export HF_SPACE=<you>/hinglish-agent
hf repo create "$HF_SPACE" --repo-type space --space_sdk docker        # public by default; do not add --private
```

(Check the flag spelling with `hf repo create --help`; it has changed between huggingface_hub releases.)

## 3. Set Variables and Secrets BEFORE the first push

Space → Settings → **Variables and secrets** (ENV.md section 2 has every option):

| kind | name | value |
|---|---|---|
| Secret | `RUNPOD_API_KEY` | RunPod key |
| Secret | `S2S_SESSION_SECRET` | same as the endpoint |
| Secret | `S2S_PASSCODE` | a passcode you share with testers (strongly recommended) |
| Variable | `RUNPOD_ENDPOINT_ID` | the endpoint id |
| Variable | `S2S_MODE` | `lb` |
| Variable | `S2S_AUDIENCE` | same as the endpoint (e.g. `hinglish-lb-1`) |
| Variable | `MAX_CONCURRENT_CALLS` | = endpoint max workers |

Changing a variable or secret restarts the Space. Sessions are in memory, so a restart drops any call in progress.

## 4. Push

**Use `hf upload` (primary).** The Space folder contains binary files (`static/assets/*.wasm`, the favicon PNGs).
HF rejects binary files pushed over plain git whatever their size ("Your push was rejected because it contains
binary files"), and `hf upload` stores them through Xet/LFS automatically (REVIEW-ops-1).

```bash
cd s2s-space
hf upload "$HF_SPACE" . . --repo-type space --exclude "tests/*" --commit-message "S2S serverless Space front"
```

Plain git works only if LFS tracks every binary **before the first commit**:

```bash
cd s2s-space
git init -b main && git lfs install
git lfs track "*.wasm" "*.png" "*.ico" "*.jpg" "*.wav" "*.mp3" "*.ogg"
git remote add origin https://huggingface.co/spaces/$HF_SPACE
git add .gitattributes && git add -A && git commit -m "S2S serverless Space front"
git push -u origin main            # username = your HF user, password = the write token
```

Watch the build: Space page → **Logs** (build, then container). Then open the **direct** URL
`https://<you>-hinglish-agent.hf.space/` (not the huggingface.co/spaces page, R8) and run TEST_PLAN.md step 1.

## 5. Launcher (b): free account, ZeroGPU Gradio Space (UNTESTED, R3)

Edit `README.md`'s YAML header: `sdk: gradio`, `sdk_version: <current gradio>`, `app_file: app.py`, and remove
`app_port`. Push the same folder. `app.py` runs the same aiohttp app on :7860. HF may still require at least one
`@spaces.GPU` function (the shim declares a dummy one when `SPACES_ZERO_GPU` is set) and may proxy only Gradio routes.
Test #1 (ws echo) tells you in a minute whether it works.

## 6. Launcher (c): any container host (fallback front)

```bash
bash space/stage_common.sh && bash ops/build_client.sh --copy-only
docker build -t s2s-space space/
docker run -d -p 7860:7860 --env-file s2s-space.env s2s-space     # s2s-space.env: the variables + secrets of section 3
```

Put HTTPS in front of it, for example Caddy or a Cloudflare tunnel: browsers allow the microphone only on https.
`ops/build_and_push.sh --space-image <registry>/s2s-space` builds and pushes the same image.

## 7. Updating

Push again. HF rebuilds, which takes a few minutes, and the Space restarts. Calls in progress drop: tell testers first.
