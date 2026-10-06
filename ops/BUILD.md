# S2S serverless: build, push and deploy (index, for the user to run later)

> **Superseded 2026-10-06:** RunPod now builds the worker from the GitHub repo `s2s-worker`, and nothing is baked.
> See `GO_LIVE.md` and DECISIONS "D-GOLIVE-PREP". This file is kept for reference only.

Nothing here was run by the build stage: no image was built, nothing was pushed, no endpoint or Space was created.
`build_and_push.sh` and `create_endpoint.sh` have only been dry-run, by the ops builder and the ops reviewer: fake
curl/docker/git/hf/npm were first on PATH and were called 0 times. The critic stage could not re-run them because a
permission block stopped it. They have never run for real, so read the plan they print before you add `--push` /
`--yes`. A bundle made by `prepare_build_context.sh --bundle` was rehearsed (CRIT-2): it extracts, and
`check_dockerfile.py --root <extracted>` passes.
Every script prints its commands by default and acts only with an explicit flag. DESIGN.md Amendment C1 maps these
files to the DESIGN section 1 names.

| step | where | file | acts only with |
|---|---|---|---|
| 1. assets into the build context | runpod2 | `ops/prepare_build_context.sh` (`--check`, `--bundle FILE.tar`) | local copy only (no remote effect) |
| 2. web client → `space/static` | runpod2 | `ops/build_client.sh` (`--copy-only`, `--dry-run`) | local build only |
| 3. build + push the worker image | a docker host (≥ 60 GB disk; the pods have no docker daemon) | `ops/build_and_push.sh` | `--run`, `--push` |
| 4. create the RunPod endpoint | anywhere | `ops/create_endpoint.sh` (`--mode lb|queue`) + the **Model** console step | `--yes` |
| 5. create + push the HF Space | the laptop | `ops/space_push.md` | you |
| 6. test | the laptop / India | `tests/TEST_PLAN.md` | `--space URL` or `--direct …` |

Reference: `ops/ENV.md` (every variable), `ops/ENDPOINT_SETTINGS.md` (every console setting), `ops/COSTS.md`.

## Short path (LB mode)

```bash
# runpod2
cd /root/deploy_serverless
bash ops/build_client.sh --copy-only && bash space/stage_common.sh   # stage the Space first, so the bundle has it
bash ops/prepare_build_context.sh --bundle /root/s2s_build_context.tar

# docker host (example: copy the bundle from runpod2)
scp runpod2:/root/s2s_build_context.tar . && mkdir s2s && tar -xf s2s_build_context.tar -C s2s && cd s2s
docker login                                    # registry of your choice; make the repo PRIVATE (see build_and_push.sh)
IMAGE=docker.io/<you>/s2s-worker TAG=v1 bash ops/build_and_push.sh            # read the plan
IMAGE=docker.io/<you>/s2s-worker TAG=v1 bash ops/build_and_push.sh --push     # about 14 GB image, 20-40 min

# endpoint (secrets only in your shell)
export S2S_SESSION_SECRET=$(python3 -c "import secrets;print(secrets.token_urlsafe(48))")   # keep it; the Space needs it too
export S2S_AUDIENCE=hinglish-lb-1 IMAGE=docker.io/<you>/s2s-worker TAG=v1
bash ops/create_endpoint.sh --mode lb --max-workers 1                  # read the request
read -s RUNPOD_API_KEY && export RUNPOD_API_KEY
bash ops/create_endpoint.sh --mode lb --max-workers 1 --yes            # prints ENDPOINT_ID
#   then the console MODEL step: nvidia/personaplex-7b-v1 + HF token (no API for it)

# Space: ops/space_push.md (Variables/Secrets incl. RUNPOD_ENDPOINT_ID, then push)
# Tests: tests/TEST_PLAN.md
```

## Where the image is built

The pods are containers without a docker daemon, and the laptop should not do heavy jobs. Options:
- (a) any Linux box or cloud VM with docker, ≥ 60 GB disk and a fast uplink, since the push is about 14 GB. Copy
  `/root/s2s_build_context.tar` (about 0.5 GB) there.
- (b) the laptop, if you accept a 20-40 min build. Docker Desktop or the docker engine with BuildKit.
- (c) RunPod's own "deploy from GitHub" build is NOT set up: `worker/build/` (470 MB of weights) is not in any git
  repo, and must not go into a public one.

## If runpod2 is gone (CRIT-2)

runpod2's disk is ephemeral. The laptop copy (`/home/shivam/Desktop/S2S/hinglish/deploy_serverless`) has all the code
but not the baked assets. A copy of them is parked on pod1's persistent network volume:

- `runpod:/workspace/hinglish/deploy_serverless_assets/s2s_build_assets.tar`
- 466 MB, md5 `f620575d6357e9a20fd743213c803c0b`
- contains `worker/build/{v3_adapter/{config.json,lora.safetensors},tuned_full.cact,cactus-needle/v3/3.0.2/{libneedle.so,needle3.cact}}`
  and `common/data/records_v4.json`

To use it on the docker host:
```bash
rsync -a <laptop>:/home/shivam/Desktop/S2S/hinglish/deploy_serverless/ s2s/        # code (incl. space/static)
scp runpod:/workspace/hinglish/deploy_serverless_assets/s2s_build_assets.tar . && tar -xf s2s_build_assets.tar -C s2s
cd s2s && bash space/stage_common.sh && python3 worker/tests/check_dockerfile.py    # must print OK
```
Rehearsed on 2026-10-05: the laptop copy plus this tar, then stage_common.sh, then `check_dockerfile.py --root .`
printed OK, and `space/static/index.html` was present.

The original sources, if you need to rebuild the tar, are on pod1 `/workspace`:
- `/workspace/runs/V3_A/checkpoints/checkpoint_000200/consolidated/` (V3 LoRA; its md5 must match `96d13850…`);
- `/workspace/hinglish/needle/finetune/sweep/tuned_full.cact`.

The Needle native lib cache (`~/.cache/cactus-needle/v3/3.0.2`) exists only on runpod2 and inside the tar. Pass the
sources to `prepare_build_context.sh` with `SRC_ADAPTER=… SRC_CACT=… SRC_NEEDLE_LIB=… SRC_RECORDS=…`.
