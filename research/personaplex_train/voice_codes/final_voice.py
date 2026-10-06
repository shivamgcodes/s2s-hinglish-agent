"""Final solve: text/user rows fixed to their known values, agent rows searched until bf16 re-embedding is exact."""
import torch, itertools, json
from safetensors import safe_open
S = "/workspace/hf/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309"
D = "/workspace/hinglish/trainer/voice_codes"
f = safe_open(f"{S}/model.safetensors", "pt")
tabs = [f.get_tensor(f"emb.{k}.weight") for k in range(16)] + [f.get_tensor("text_emb.weight")]
t32 = [t.float() for t in tabs]
SINE = [430, 1268, 381, 1611, 1095, 1495, 56, 472]
def embed(col):
    x = None
    for k in range(16):
        e = tabs[k][col[1 + k]]; x = e if x is None else x + e
    return x + tabs[16][col[0]]
def err(col, e): return (embed(col).float() - e.float()).abs().max().item()
rep = {}
for v in ["NATF2", "NATM1"]:
    d = torch.load(f"{S}/voices/{v}.pt", map_location="cpu"); E = d["embeddings"][:, 0, 0]; cache = d["cache"][0]
    cols = torch.load(f"{D}/{v}.columns.pt")["columns"].clone()
    errs = []
    for c in range(E.shape[0]):
        col = cols[:, c].tolist()
        if c >= 1:
            col[0] = 3; col[9:] = [430] + ([2048] * 7 if c == 1 else SINE[1:])
            if c == 1: col[2:9] = [2048] * 7
        for rnd in range(4):
            if err(col, E[c]) == 0: break
            for _ in range(4):  # coordinate descent
                for tab in range(8):
                    if c == 1 and tab > 0: continue
                    rest = sum(t32[j][col[1 + j]] for j in range(16) if j != tab) + t32[16][col[0]]
                    col[1 + tab] = int(((t32[tab] - (E[c].float() - rest)) ** 2).sum(-1).argmin())
            if err(col, E[c]) == 0: break
            K = 3 + 2 * rnd
            cands = []
            for tab in range(8):
                rest = sum(t32[j][col[1 + j]] for j in range(16) if j != tab) + t32[16][col[0]]
                dist = ((t32[tab] - (E[c].float() - rest)) ** 2).sum(-1)
                cands.append(dist.topk(K, largest=False).indices.tolist() if not (c == 1 and tab > 0) else [2048])
            best = None
            for combo in itertools.product(*cands):
                cc = col[:]; cc[1:9] = list(combo); e = err(cc, E[c])
                if best is None or e < best[0]: best = (e, cc)
                if e == 0: break
            col = best[1]
        cols[:, c] = torch.tensor(col); errs.append(err(col, E[c]))
    vf = torch.full((8, 52), -1, dtype=torch.long)
    for c in range(1, 51): vf[0, c] = cols[1, c]
    for c in range(2, 51): vf[1:, c - 1] = cols[2:9, c]
    vf[0, 51] = cache[1, 3]; vf[1:, 50] = cache[2:9, 3]; vf[1:, 51] = cache[2:9, 0]
    torch.save({"columns": cols, "max_abs_err": errs}, f"{D}/{v}.columns.pt")
    torch.save({"vf": vf, "note": "agent-stream Mimi codes per LMGen voice frame f=0..51 ([8,52]); vf[:,0] unknown (-1; overwritten by the initial token at inference). Training prefix agent rows = vf[:,1:52] (51 frames)."}, f"{D}/{v}.frames.pt")
    rep[v] = {"n_exact": sum(e == 0 for e in errs), "n_cols": len(errs), "max_err": max(errs), "inexact_cols": [i for i, e in enumerate(errs) if e > 0],
              "cache49_eq": torch.equal(cache[:, 1], cols[:, 49]), "cache50_eq": torch.equal(cache[:, 2], cols[:, 50])}
print(json.dumps(rep))
