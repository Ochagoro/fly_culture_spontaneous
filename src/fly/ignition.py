"""Where do spontaneous bursts ignite? Spike-level analysis of whole-brain recordings.

For every burst (onset = first 1-ms bin where the 20-ms smoothed population
rate crosses the burst threshold) and every cell type x hemisphere group:

* pre-onset enrichment: spikes in [onset - 20 ms, onset) versus the number
  expected from the group's baseline rate (bins >= 200 ms away from any burst);
  summed over bursts, with a one-sided Poisson p value.
* crossing time: first time (1 ms resolution) at which the group's spike count
  in a sliding 5-ms window exceeds baseline + 5 sqrt(baseline) and >= 3 spikes,
  searched from onset - 50 ms; relative to the population onset.

Results: results/fly/ignition_<variant>_g<gain>.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from spont import load_connectome  # noqa: E402
from variants import load_annotations  # noqa: E402
from bursts import groups_of  # noqa: E402

ROOT = HERE.parents[1]
VD = ROOT / "results" / "fly" / "video"


def onsets_exact(pop: np.ndarray, on: np.ndarray, base: float) -> np.ndarray:
    """Refine onsets: last bin before the detected crossing where the 5-ms
    smoothed rate was still below 1.25x baseline + 10."""
    x = np.convolve(pop, np.ones(5) / 5, mode="same")
    out = []
    for a in on:
        k = a
        while k > 0 and x[k] > 1.25 * base + 10:
            k -= 1
        out.append(k)
    return np.array(out)


def analyse(path: Path, codes, names, sizes):
    d = np.load(path)
    t_ms = d["t"].astype(np.int64) // 10
    n = d["n"].astype(np.int64)
    pop = d["pop"]
    T = len(pop)
    base = float(np.median(pop))
    on = onsets_exact(pop, d["on"], base)
    G = len(names)
    # group x 1-ms count matrix (sparse via bincount on flattened index)
    g = codes[n]
    M = np.bincount(g * T + t_ms, minlength=G * T).reshape(G, T).astype(np.float32)
    far = np.ones(T, bool)
    for a, b in zip(d["on"], d["off"]):
        far[max(0, a - 200): b + 200] = False
    far[:300] = False
    lam = M[:, far].mean(1)                      # baseline spikes per ms per group
    obs = np.zeros(G)
    for a in on:
        if a >= 20:
            obs += M[:, a - 20: a].sum(1)
    exp = lam * 20 * len(on)
    p = stats.poisson.sf(obs - 1, np.maximum(exp, 1e-12))
    # crossing time
    W = 5
    cs = np.cumsum(np.c_[np.zeros(G), M], 1)
    lam5 = lam * W
    thr = np.maximum(lam5 + 5 * np.sqrt(lam5), 3)
    cross = np.full((len(on), G), np.nan)
    for i, a in enumerate(on):
        lo, hi = max(W, a - 50), min(T, a + 60)
        win = cs[:, lo:hi] - cs[:, lo - W: hi - W]   # count in (t-W, t]
        hit = win > thr[:, None]
        first = np.where(hit.any(1), hit.argmax(1), -1)
        cross[i] = np.where(first >= 0, lo + first - a, np.nan)
    df = pd.DataFrame(dict(group=names, n=sizes, baseline_hz_per_neuron=lam * 1000 / np.maximum(sizes, 1),
                           pre_obs=obs, pre_exp=exp, enrichment=(obs + 1) / (exp + 1), p=p,
                           excess_spikes_per_burst=(obs - exp) / max(len(on), 1),
                           cross_ms_median=np.nanmedian(cross, 0) if len(on) else np.nan,
                           frac_bursts_crossed=np.mean(~np.isnan(cross), 0) if len(on) else np.nan))
    df["type"] = df.group.str.split("|").str[0]
    df["side"] = df.group.str.split("|").str[1]
    df["n_bursts"] = len(on)
    return df


def main():
    con = load_connectome("783")
    ann = load_annotations(con)
    codes, names = groups_of(ann)
    sizes = np.bincount(codes, minlength=len(names))
    for f in sorted(VD.glob("spikes_*.npz")):
        df = analyse(f, codes, names, sizes)
        tag = f.stem.replace("spikes_", "")
        df.to_csv(ROOT / "results" / "fly" / f"ignition_{tag}.csv", index=False)
        sig = df[(df.p < 1e-6) & (df.pre_obs >= 10)]
        print(f"== {tag}: {df.n_bursts.iloc[0]} bursts; {len(sig)} groups enriched (p<1e-6)")
        print(sig.sort_values("excess_spikes_per_burst", ascending=False).head(20)[
            ["group", "n", "baseline_hz_per_neuron", "enrichment", "excess_spikes_per_burst",
             "cross_ms_median", "frac_bursts_crossed"]].round(3).to_string())
        early = df[(df.frac_bursts_crossed >= 0.5)].sort_values("cross_ms_median").head(15)
        print("-- earliest crossing (crossed in >= 50% of bursts):")
        print(early[["group", "n", "cross_ms_median", "frac_bursts_crossed", "enrichment"]].round(3).to_string())


if __name__ == "__main__":
    main()
