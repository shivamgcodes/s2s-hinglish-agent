"""Recover the token columns behind PersonaPlex voice-prompt .pt embeddings (CPU only).

voices/<V>.pt = {"embeddings": [51,1,1,4096] bf16 = LMModel.embed_codes(input column c) for c=0..50,
                 "cache": [1,17,4] LMGen cache after the voice prompt}.
embed_codes(col) = sum_{k=0..15} emb[k](col[1+k]) + text_emb(col[0]) (bf16, that order; lm.py embed_codes).
Greedy matching pursuit over the 17 tables, then exact bf16 re-embedding check.
Output: <out>/<V>.columns.pt with int64 [17, 51] columns + residual stats, and
        <out>/<V>.frames.pt : agent-stream (moshi) codes per voice frame vf[f] f=0..51, [8, 52] (-1 = unknown).
"""
import sys, json, torch
from safetensors import safe_open
S = "/workspace/hf/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309"
OUT = sys.argv[1]
f = safe_open(f"{S}/model.safetensors", "pt")
tabs = [f.get_tensor(f"emb.{k}.weight") for k in range(16)] + [f.get_tensor("text_emb.weight")]  # bf16
tabs32 = [t.float() for t in tabs]
norms = [(t * t).sum(-1) for t in tabs32]
delays = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]

def embed(col):  # col: list of 17 ints [text, a0..a15]; replicate embed_codes order/dtype
    x = None
    for k in range(16):
        e = tabs[k][col[1 + k]]
        x = e if x is None else x + e
    return x + tabs[16][col[0]]

report = {}
for v in sys.argv[2:]:
    d = torch.load(f"{S}/voices/{v}.pt", map_location="cpu")
    E = d["embeddings"][:, 0, 0]  # [51,4096] bf16
    cols = torch.full((17, E.shape[0]), -9, dtype=torch.long)
    maxerr = []
    for c in range(E.shape[0]):
        r = E[c].float().clone()
        left = set(range(17))
        chosen = {}
        for _ in range(17):
            best = None
            for t in left:
                score = norms[t] - 2 * (tabs32[t] @ r)  # ||r-e||^2 - ||r||^2
                i = int(score.argmin()); s = float(score[i])
                if best is None or s < best[0]:
                    best = (s, t, i)
            _, t, i = best
            chosen[t] = i; left.discard(t); r -= tabs32[t][i]
        col = [chosen[16]] + [chosen[k] for k in range(16)]
        cols[:, c] = torch.tensor(col)
        err = (embed(col).float() - E[c].float()).abs().max().item()
        maxerr.append(err)
    cache = d["cache"][0]
    # voice frames: col c (c>=2) has d0 streams = vf[c], d1 streams = vf[c-1]  (see RECON.md)
    vf = torch.full((8, 52), -1, dtype=torch.long)
    for c in range(1, E.shape[0]):
        vf[0, c] = cols[1, c]
    for c in range(2, E.shape[0]):
        vf[1:, c - 1] = cols[2:9, c]
    # cache columns: col 51 -> index 51%4=3 (d0: vf[51], d1: vf[50]); col 52 -> index 0 (d1: vf[51])
    vf[0, 51] = cache[1, 3]; vf[1:, 50] = cache[2:9, 3]; vf[1:, 51] = cache[2:9, 0]
    torch.save({"columns": cols, "max_abs_err": maxerr}, f"{OUT}/{v}.columns.pt")
    torch.save({"vf": vf, "note": "agent-stream codes per voice frame; vf[:,0] unknown (-1)"}, f"{OUT}/{v}.frames.pt")
    cons = sum(int(cols[1, c] == vf[0, c]) for c in range(1, 51))
    report[v] = {
        "max_abs_err_all_cols": max(maxerr), "n_cols_exact": sum(e == 0 for e in maxerr),
        "text_row": sorted(set(cols[0].tolist())), "col0": cols[:, 0].tolist(),
        "user_rows_col5": cols[9:, 5].tolist(), "user_rows_col1": cols[9:, 1].tolist(),
        "agent_d1_col1": cols[2:9, 1].tolist(),
        "cache_matches_col_ring": [int((cache[:, 3] >= 0).all())],
        "vf_first": vf[:, 1].tolist(), "vf_last": vf[:, 51].tolist(),
    }
print(json.dumps(report, indent=1))
