# packages/: shared code, one copy (D-SINGLE-SOURCE, D-LEAN-HF, 2026-10-07)

Code that more than one place needs lives here and nowhere else.
- `research/` and `deploy/` use it in place.
- Two of the folders are also **pip-installable from GitHub**. Since D-LEAN-HF the Hugging Face model repos hold only
  weights, configs, examples and the model card, so the model cards install these packages.

```bash
pip install "git+https://github.com/shivamgcodes/s2s-hinglish-agent#subdirectory=packages/needle_router"
pip install "git+https://github.com/shivamgcodes/s2s-hinglish-agent#subdirectory=packages/personaplex_lora"
```
The GitHub repo is private until release.

| dir | pip name / import | what | used by |
|---|---|---|---|
| `needle_router/` | `needle-router` / `needle_router` (deps: `cactus-needle==3.0.6`) | Needle v2 router: `v2/` (`router_v2`: route, route_raw, resolve_calls, prepare_transcript; `resolver_v2`, `tools_v2` (7 agent types), `targets`, `n1path`), `numconv/`, and the N1 router + helpers v2 uses in `n1/` (`router`, `resolver`, `romanise` + lexicon, `tools` + `tools_json/`, `build_data.system_text`, `src/records.json`) | live worker (`deploy/worker/router/router_core.py`; `S2S_ROUTER=v2`, rollback `n1`), `research/needle/`, `research/inference/needle_router/example.py` |
| `personaplex_lora/` | `personaplex-lora` / `personaplex_lora` (torch, safetensors and PersonaPlex's `moshi` are installed separately) | `infer/lora_merge.py` (load_adapter, merge_into), `trainer/merge_lora.py` (the `merge_into(lm, path)` hook) + `trainer/voice_codes/`, `role_prompt.py` (record/pairing → role prompt, voice) | live worker (`paths.MERGE_LORA`), `deploy/common/session.py` (role prompt), research eval harness + training, `research/inference/personaplex_lora/run_offline.py` |
| `hinglish_text/` | (not packaged) | `hindi_share.py` (Hindi-share counter / token classifier) + project-made `lexicon/{hindi,english}_extra.txt` | research/data_gen + eval harness, the N1 data build (`build_data`, imported lazily) |
| `selftest.py` | | stdlib CPU self-test: imports without the engine, lazy hindi_share, role-prompt wiring, both wheels build with their data | `deploy/run_all_cpu_tests.sh` |

**Flat modules.** The router modules use bare imports (`import router_v2`, `import build_data`, ...), as in the code of
record and in the former HF quickstart.
- `needle_router/__init__.py` puts `v2/`, `n1/` and `numconv/` on `sys.path` and re-exports the API.
- The worker and the research scripts put the same folders on `sys.path` directly.
- So an installed package and a checkout run the same files.

**How the worker image gets them.** The image COPYs `needle_router/` and `personaplex_lora/` to `/opt/s2s/packages/`
and uses them in place. `deploy/common/session.py` finds `role_prompt.py` there. The Space build stages
`role_prompt.py` next to `session.py` (`deploy/space/stage_common.sh`).

## Runtime vs data-build dependencies
- The routers use only `build_data.system_text` / `PINNED_DATE`. `build_data` imports `hindi_share` **lazily**, so the
  worker image and the pip package need neither `hinglish_text/` nor any word list. `selftest.py` and the image build
  assert this.
- `hindi_share` reads two Google English word lists (`google-20k-en.txt`, `google-10000-en.txt`, from
  github.com/first20hours/google-10000-english). They have **no clear licence**, so they are **not in this repo**.
  To run a data build or the validators, put them in a directory and set `HINDI_SHARE_LEXICON=<dir>`.

## Where the code came from
All of these sources were byte-identical on the laptop and pod1 on 2026-10-07.

| package code | source |
|---|---|
| `needle_router/n1/*` | `hinglish/needle/`. It was also `worker/needle/` and the HF repo's `needle/`. |
| `needle_router/v2/*` | `hinglish/needle_v2/schema/` |
| `numconv/` | `hinglish/needle_v2/numconv/` |
| `personaplex_lora/{infer,trainer}` | pod1 `hinglish/{infer,trainer}` |
| `personaplex_lora/role_prompt.py` | DEP1 `common/session.py` + `tests/tcommon.py` `ascii_prompt`. It was also the HF repo's `role_prompt.py`. |
| `hinglish_text/hindi_share.py` | pod1 `hinglish/gen/hindi_share.py` (= the `needle/vendor` copy) |

Edits were limited to paths. Dev-machine paths were replaced by paths relative to the file or by env vars with the
same meaning, and each module's docstring names its origin. Two other changes were made:
- the lazy `hindi_share` import;
- the package scaffolding (`pyproject.toml`, `__init__.py`).

The laptop folders `hinglish/needle/` and `hinglish/needle_v2/` are now **archives** (results, data, reports). They
are no longer a code source.
