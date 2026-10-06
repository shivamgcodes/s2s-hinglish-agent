# S2S serverless: PersonaPlex on a network volume (fallback when the Model cache is not offered)

Written 2026-10-05 by the completeness critic (CRIT-1). Nothing here was run: no volume, pod or endpoint was created.

## When you need this

The primary plan caches `nvidia/personaplex-7b-v1` with the endpoint's **Model** field (ENDPOINT_SETTINGS.md 1).
The RunPod model-caching page does not say whether that field exists for **load-balancing** endpoints. Neither the
load-balancing "build a worker" page nor the endpoint-configurations page mentions it either (all three re-read on
2026-10-05). Use this page if one of these happens:
- the LB endpoint's settings have no Model field; or
- the worker logs `PersonaPlex not found` and `/ping` answers 500 although the Model step was done (R12).

The worker finds the files on its own. `worker/resolve_models.py` step 3 looks in
`$S2S_VOLUME_HF/models--nvidia--personaplex-7b-v1/snapshots/<refs/main or newest>/`, which defaults to
`/runpod-volume/hf/hub/...`. That is the same snapshot lookup the model cache uses, and the worker tests cover it.
No image change is needed.

## Trade-offs (DESIGN 6.4 C)

- The endpoint is **pinned to the volume's data center** ([EP-CFG] "Network volumes"). Combined with the 48 GB and
  CUDA 13.0+ filters, that data center may have few or no eligible hosts (R5). Pick it with care (step 1).
- Loads are slower than the model cache (network storage). Expect the PersonaPlex read (about 16 GB) to add
  tens of seconds to every cold start. Measure it with `tests/cold_start_timer.py`.
- Storage cost is about $0.07/GB/month, so about $1.2/month for 17 GB ([PRICE]; COSTS.md).
- **Do not set the Model field as well.** Both use the `/runpod-volume` mount ([CACHE]).

## Steps

1. **Choose the data center.** Console → Serverless → New Endpoint → GPU configuration. With the CUDA 13.0+ filter
   set, note which data centers offer the 48 GB A6000/A40 (and 4090 / L40S) pools. Choose one that offers several of
   them. AP-JP-1 is closest to India but small (ENDPOINT_SETTINGS 3).
2. **Create the volume.** Console → Storage → Network Volume, in that data center. Size: **30 GB** (17 GB of files
   plus headroom).
3. **Fill it from a cheap pod.**
   - Deploy the cheapest pod in that data center with the volume attached. A CPU pod is enough if the data center
     offers one. **On a pod the volume is mounted at `/workspace`; on serverless it is mounted at `/runpod-volume`.**
   - In the pod's web terminal:
     ```bash
     pip install -U "huggingface_hub[hf_xet]"
     read -s HF_TOKEN && export HF_TOKEN       # read-only token of the account that accepted the licence
     export HF_HOME=/workspace/hf              # -> /workspace/hf/hub = /runpod-volume/hf/hub on the worker
     export HF_XET_CACHE=/tmp/hf-xet           # keep the download's chunk cache OFF the volume
     hf download nvidia/personaplex-7b-v1 \
        model.safetensors tokenizer-e351c8d8-checkpoint125.safetensors tokenizer_spm_32k_3.model voices.tgz config.json
     unset HF_TOKEN
     ls -L /workspace/hf/hub/models--nvidia--personaplex-7b-v1/snapshots/*/
     cat /workspace/hf/hub/models--nvidia--personaplex-7b-v1/refs/main
     ls /workspace/hf/token 2>/dev/null && echo "REMOVE IT: rm /workspace/hf/token"
     du -sh /workspace/hf                      # about 17 GB expected; delete any xet/ or other cache dir under it
     ```
   - **Do not run `hf auth login` here.** With `HF_HOME` on the volume, it would save your token to
     `/workspace/hf/token`, and every worker that mounts the volume could read it. The `ls` line above checks for
     that file.
   - The snapshot files are relative symlinks into `../../blobs/`, so they still resolve under `/runpod-volume`.
   - **Terminate the pod** when done. The volume keeps the files.
4. **Endpoint.**
   - Network volume: this volume.
   - Model field: **empty**.
   - Data center: follows the volume.
   - Everything else as ENDPOINT_SETTINGS.md 1 (or 2 for queue).
   - No extra env is needed: `S2S_VOLUME_HF` defaults to `/runpod-volume/hf/hub`. If you used another layout, set
     `S2S_PP_DIR` to the snapshot directory itself (resolver step 1).
5. **Check.**
   - Run TEST_PLAN step 3 (cold start).
   - The worker log shows the resolver's JSON line with `"source": "network_volume"` and the file sizes.
   - A `/ping` 500 with `PersonaPlex not found` lists every path it tried.

## Pre-merged variant (DESIGN 6.4 A on the volume)

- Run `worker/tools/premerge.py --out <dir>` on a GPU box (about 16 GB VRAM). This tool is untested.
  `<dir>` then holds the merged `model.safetensors` plus copies of the Mimi, tokenizer and voices files.
- Copy that directory to the volume, for example `/workspace/pp_v3merged/`.
- Set endpoint env `S2S_PP_DIR=/runpod-volume/pp_v3merged` and `S2S_ADAPTER=premerged`.
- This keeps NVIDIA's weights off any registry or HF repo. Check the licence before you re-host merged weights
  anywhere else.
