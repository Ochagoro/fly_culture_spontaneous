"""The culture test, applied to the simulated fly brain: does the leading
eigenvector of between-burst functional coupling predict where bursts ignite
better than the between-burst firing rate does?

Mirrors culture analysis C3(v): units = cell type x hemisphere groups with
>= 30 between-burst spikes; Pearson correlation of 20-ms spike counts in
between-burst periods (bursts +-100 ms excluded); negatives set to 0,
diagonal 0; leading eigenvector by power iteration. Ignition = excess spikes
per neuron in the 20 ms before burst onset (from ignition.py). Also compares
the structural Perron vector of the excitatory wiring (structure.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from spont import load_connectome  # noqa: E402
from variants import load_annotations  # noqa: E402
from bursts import groups_of  # noqa: E402

ROOT = HERE.parents[1]
RES = ROOT / "results" / "fly"


def leading_eig(C: np.ndarray, iters: int = 500) -> np.ndarray:
    v = np.ones(C.shape[0]) / np.sqrt(C.shape[0])
    for _ in range(iters):
        w = C @ v
        nv = np.linalg.norm(w)
        if nv == 0:
            break
        w /= nv
        if np.linalg.norm(w - v) < 1e-10:
            v = w
            break
        v = w
    return np.abs(v)


def run(tag: str, codes, names, sizes, variant: str):
    d = np.load(RES / "video" / f"spikes_{tag}.npz")
    ig = pd.read_csv(RES / f"ignition_{tag}.csv")
    t_ms = d["t"].astype(np.int64) // 10
    n = d["n"].astype(np.int64)
    T = len(d["pop"])
    keep_t = np.ones(T, bool)
    keep_t[:300] = False
    for a, b in zip(d["on"], d["off"]):
        keep_t[max(0, a - 100): b + 100] = False
    sel = keep_t[np.minimum(t_ms, T - 1)]
    g = codes[n[sel]]
    tb = t_ms[sel] // 20
    G = len(names)
    cnt = np.bincount(g, minlength=G)
    units = np.flatnonzero(cnt >= 30)
    lut = -np.ones(G, np.int64)
    lut[units] = np.arange(len(units))
    nb = T // 20 + 1
    X = np.zeros((len(units), nb), np.float32)
    m = lut[g] >= 0
    np.add.at(X, (lut[g[m]], tb[m]), 1)
    # keep only bins fully in between-burst periods
    okb = keep_t[: (T // 20) * 20].reshape(-1, 20).all(1)
    X = X[:, : len(okb)][:, okb]
    X = X[X.std(1) > 0] if False else X
    sd = X.std(1)
    good = sd > 0
    Xz = (X[good] - X[good].mean(1, keepdims=True)) / sd[good, None]
    C = (Xz @ Xz.T) / Xz.shape[1]
    C[C < 0] = 0
    np.fill_diagonal(C, 0)
    ev = leading_eig(C.astype(np.float64))
    u = units[good]
    df = pd.DataFrame(dict(group=names[u], n=sizes[u], eig=ev,
                           rate_ib=cnt[u] / sizes[u] / (okb.sum() * 0.02)))
    df = df.merge(ig[["group", "excess_spikes_per_burst", "enrichment", "n_bursts"]], on="group")
    df["ignition_per_neuron"] = df.excess_spikes_per_burst / df.n
    st = pd.read_csv(RES / f"structure_groups_{variant}.csv")
    df = df.merge(st[["group", "perron", "recip"]], on="group", how="left")
    df["perron_per_neuron"] = df.perron / df.n
    out = {}
    for pred in ["eig", "rate_ib", "perron_per_neuron", "recip"]:
        r, p = stats.spearmanr(df[pred], df["ignition_per_neuron"], nan_policy="omit")
        out[pred] = (r, p)
    # partial: eig given rate
    from scipy.stats import rankdata
    def resid(a, b):
        a, b = rankdata(a), rankdata(b)
        return a - np.polyval(np.polyfit(b, a, 1), b)
    rp = stats.pearsonr(resid(df.eig, df.rate_ib), resid(df.ignition_per_neuron, df.rate_ib))[0]
    df.to_csv(RES / f"functional_eig_{tag}.csv", index=False)
    top = df.sort_values("eig", ascending=False).head(10)[["group", "n", "eig", "rate_ib", "ignition_per_neuron"]]
    return dict(tag=tag, n_units=len(df), n_bursts=int(ig.n_bursts.iloc[0]),
                **{f"rho_{k}": v[0] for k, v in out.items()}, **{f"p_{k}": v[1] for k, v in out.items()},
                partial_eig_given_rate=rp), top


def main():
    con = load_connectome("783")
    ann = load_annotations(con)
    codes, names = groups_of(ann)
    sizes = np.bincount(codes, minlength=len(names))
    rows = []
    for f in sorted((RES / "video").glob("spikes_*.npz")):
        tag = f.stem.replace("spikes_", "")
        if not (RES / f"ignition_{tag}.csv").exists():
            continue
        variant = tag.split("_g")[0]
        r, top = run(tag, codes, names, sizes, variant)
        rows.append(r)
        print(r)
        print(top.round(4).to_string())
    pd.DataFrame(rows).to_csv(RES / "functional_eig_summary.csv", index=False)


if __name__ == "__main__":
    main()
