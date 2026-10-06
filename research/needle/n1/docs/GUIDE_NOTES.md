# Needle 3 guide notes for N1 (read 2026-10-04)

Sources:
- The guides are not shipped in the wheel. The installed `cactus-needle` 3.0.6 README (METADATA) only links to them.
- "Fine-tuning Needle": https://cactuscompute.com/blog/finetuning-needle
- "How to design tools for Needle 3": https://cactuscompute.com/blog/designing-tools-for-needle
- "Needle Python docs": https://cactuscompute.com/blog/needle-python-docs (system facts, response shape)
- Installed code, which is the authority where it is more exact than the blogs:
  - `/mnt/drive2/venv/needle-exp0/lib/python3.11/site-packages/needle/model/finetune.py` (`render_example`, `read_examples`, `from_chat`, `finetune_local`)
  - `needle/cli.py` (flags)
  - `needle/agent/tools.py` (`build_schema`)
  - `needle/__init__.py` (`Needle`, `_with_date_fact`)
- The quotes below are from the blog pages, fetched through a summariser, so treat their wording as near-verbatim, not exact.

## 1. Training data format (local `needle finetune`)

One JSON object per line.

| field | required | meaning |
|---|---|---|
| `query` | yes | the user text |
| `tools` | yes | list of tool schemas: a list of dicts, or a JSON string |
| `answers` | yes | list of `{"name", "arguments"}`; `[]` means no tool applies. `function_calls` is accepted as an alias. |
| `reasoning` | no | "one short line deriving each argument from its span in the query", e.g. `"'kitchen' -> room; 'to 10' -> brightness"` |
| `system` | no | a system turn, the same text as `Needle(system=...)` at inference |

Chat format is also accepted (`messages` plus OpenAI-form `tools`). It must be exactly one user turn and one assistant turn, with an optional system turn. Multi-turn lines are skipped.

What goes over the wire, from `finetune.render_example`:

```
<|im_start|>system\n{system}<|im_end|>\n            (only if system is non-empty)
<|im_start|>user\n<tools>{tools JSON, compact separators}</tools>\n{query}<|im_end|>\n<|im_start|>assistant\n
--- target (loss only here) ---
<think>\n{reasoning}\n</think>\n                   (only if reasoning is non-empty)
<tool_call>{answers JSON, compact}</tool_call><|im_end|>
```

- The loss covers only the target: the reasoning line plus the call.
- The JSON uses `ensure_ascii=False`, so Devanagari or other non-ASCII text goes through as-is.
- Sequences are truncated at `--max-len`, 1024 tokens by default, with no warning. Padding fits the longest example, bucketed at 128, 256 and so on.
- Built-in validation score: a held-out split decoded greedily through the 4-bit export numerics, `max_new_tokens=96`. A row counts as correct when the list of (name, sorted-args JSON) matches exactly.

Data rules from the guide:
1. Arguments contain only values present in the query. Optional fields with no evidence are omitted.
2. Include off-topic examples with `answers: []`. The guide says about 1 in 8. **N1 overrides this:** the spec asks for positives to negatives at about 1:1.5, and the spec wins.
3. Include ambiguous queries, resolved to the correct tool.
4. Reasoning lines plus varied phrasings and values are what fix wrong argument values. Tool choice improves with a few hundred clean examples. Argument grounding needs on the order of thousands.

## 2. System facts format

- Write facts, not instructions: `key: value` pairs joined by `"; "`.
- Recognised keys: `date`, `locale`, `device`, `battery`, `network`, `location`, `user`, `assistant`.
- Example: `date: 2026-07-21 Tue 14:30; locale: en-US; device: phone; battery: 62%`.
- "Instructions don't steer Needle; facts do."
- `Needle(system=...)` puts `date: %Y-%m-%d %a %H:%M` in front automatically (`_with_date_fact`), unless:
  - the text already has a `date:` key or an ISO stamp, or
  - the text starts with `{`, or
  - `auto_date=False` is set.
- The minute in that date changes outputs (N0). **Every N1 eval passes its own fixed `date:` fact and sets `auto_date=False`.** The training `system` field must carry the same `date: ...; user: ...` shape.
- N1 facts shape (spec §2), for example: `date: 2026-10-04 Sun 10:00; user: orders: FD1565 Behrouz Biryani ...; older order FD4065 Wow! Momo ...`. Needle does not resolve IDs from it.

## 3. CLI (cactus-needle 3.0.6, from `cli.py`)

`needle finetune <jsonl>`:

| flag | default |
|---|---|
| `--checkpoint` | auto-downloads `needle3.safetensors` |
| `--epochs` | 3 |
| `--batch-size` | 16 |
| `--lr` | 1e-4 (warmup, then cosine decay; gradient clip at norm 1) |
| `--lora-rank` | 16 (32 "doubles adapter capacity" for grounding-heavy tasks) |
| `--lora-alpha` | 32 |
| `--max-len` | 1024 |
| `--val-split` | 0.1 |
| `--seed` | 0 |
| `--generate` | 0 (N extra examples through OpenRouter) |
| `--model` | (OpenRouter model) |
| `--workers` | (OpenRouter workers) |
| `--checkpoint-dir` | `checkpoints` |
| `--out` | `adapter.safetensors` |

- **`--val-split` is a random row split inside the file. It is not a split by scenario.** Two ways to get N1's by-scenario val:
  - train with `--val-split 0` and compute val loss/accuracy on `val.jsonl` ourselves, or
  - put `val.jsonl` rows first and use a matching `--val-split`. This does **not** work: `finetune_local` permutes all rows with the seeded RNG before it takes the first `n_val`.
- The loss starts near 1.0 because the call boilerplate is already predicted. Judge it by the trend: stop when val rises while train falls. Small datasets need 10-30 epochs.

`needle build [checkpoint]`:

| flag | meaning |
|---|---|
| `--lora <adapter>` | merges the adapter before export |
| `--layers N` | the N-layer rung, 2..20; default is the full 20 |
| `--out <x.cact>` | output path |
| `--platform <folder>` | also fetches that engine |
| `--upload` | pushes to `$NEEDLE_HF_REPO` |

- A local build is 4-bit CQ.

Run a tuned model with `needle.Needle(tools=..., system=..., weights="tuned.cact")`. It runs in a worker. `confidence` is `None` for local fine-tunes because the head is untouched.

Platform fine-tune (only if `NEEDLE_API_KEY` is set):
```
needle platform finetune train.jsonl validation.jsonl test.jsonl --suffix X --out ./models [--max-depth N --depth N]
```
- Trains the full model, mixed with Needle's own data, with a calibrated confidence head, 2-bit.
- Takes 100 to 10,000 examples.

## 4. Tool-design rules ("How to design tools for Needle 3")

- **Every argument is a span.** A call holds only values evidenced by the request.
  - An optional field with no span is omitted.
  - A required field with no span sends the call to `suppressed_calls`, so `function_calls` comes back empty.
  - Do not require arguments users won't say. Use a schema `default` for a required field the model shouldn't invent.
- **One tool per action.** Use narrow tools, not `control_home(device, action, value)`. Describe the action ("Turn a room's lights on or off, or dim them"), not a category ("Lighting control").
- **Names users would say.**
  - Enum values should be conversational words.
  - Avoid values that hide inside common words (`office` contains "off").
  - Polar pairs are separate tools or one enum.
- **Formats in descriptions.** Per-argument descriptions are read literally: `"City, ST"`, `"e.g. T-1042"`, `"ISO date"`, `"the place after 'from'"`. The model restores quoted titles, completes phone numbers and converts units when the description asks.
- **Constraints in the grammar, not prose.** These compile into the decode grammar, so out-of-range values cannot be produced:
  - `enum`, `minimum`/`maximum`, `exclusiveMinimum`/`exclusiveMaximum`
  - `minLength`/`maxLength`, `minItems`/`maxItems`
  - The guide does not list `pattern` among what compiles. `Field` accepts it.
- **Triggers.** An optional `"triggers": [regex, ...]` per tool. A match restricts decoding to the tools declaring it and forces a call. Use negative lookaheads to keep multi-action requests possible.
- **Toolset size.** Five or fewer tools render directly. **Above five, retrieval engages:** all schemas are embedded and only the 5 closest go into context. For big catalogues, pick the tool in one pass, then run a second pass with only that tool.
- **System facts, not instructions** (see §2).
- **Testing.** `needle.environments.<env>.run_tests()` covers exact calls, `[]` for missing required values, out-of-bounds values, negations and multi-call.

Tool JSON (what `needle.agent.tools.build_schema` emits; `Needle(tools=[dict,...])` takes the dicts unchanged):
```json
{"name": "change_delivery_address",
 "description": "Change the delivery address of an order to a new address.",
 "parameters": {"type": "object",
   "properties": {"order_ref": {"type": "string", "description": "..."},
                  "address":   {"type": "string", "description": "the new address as spoken, ..."}},
   "required": ["address"]},
 "triggers": ["..."]}
```
- `triggers` is optional.
- A parameter is required when it has no default and is not `Optional`.

## 5. Response shape and behaviour (Python docs)

- `complete()` / `run()` return:
  - `type`, `success`, `error`, `error_code`
  - `function_calls: [{name, arguments}]`, `suppressed_calls`
  - `reasoning`, `confidence`
  - `prefill_tps`, `decode_tps`, `peak_ram_mb`
- A request no tool can serve returns `[]`.
- Arguments are repaired deterministically:
  - names are split
  - phone numbers are completed
  - dates are grounded against the facts
  - ungrounded optional values are removed
- One toolset per agent instance. `reset()` rewinds the conversation.
- The tokenizer handles non-English text at about 1.7x the token rate. Fine-tuning does not change it.

## 6. Consequences for N1 (see DECISIONS.md)

- **`order_ref` is optional; the real arguments are required.** Real write turns rarely name the entity.
- **cab_ride_support has 6 write tools, so retrieval (top 5) engages for that agent.** Names are fixed by the spec and none were dropped. Watch the cab tool confusion in eval.
- **Eval pins the system text.** Use `auto_date=False` plus an explicit `date:` fact.
- **By-scenario val needs `--val-split 0` plus our own val scoring,** because the CLI split is by row.
