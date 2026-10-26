"""Spontaneous-activity metrics per brain region in real flies (Turner et al. 2021).

Data: 20 flies (nSyb>GCaMP6s), ~1.2 Hz volumetric imaging, no stimulus, raw F
per Ito-atlas region. For each fly and region we compute dF/F (baseline = 8th
percentile in a 60 s sliding window) and:

  sd        standard deviation of dF/F (fluctuation size)
  kurt      excess kurtosis (event-like, bursty fluctuations give high values)
  r_global  correlation with the mean of all other regions (participation in
            brain-wide fluctuations)
  event     mean z-scored dF/F in the frames where the global signal is in its
            top 5 % (participation in large spontaneous events)
  lead      lag (s) of the peak cross-correlation with the global signal;
            positive = the region leads
  ac1       lag-1 autocorrelation

Output: results/fly/turner_region_metrics.csv (fly x region rows) and a
per-region summary (median across flies).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import percentile_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from turner_load import load  # noqa: E402

ROOT = HERE.parents[1]
TD = ROOT / "data" / "fly_imaging" / "turner2021" / "data"
FS = 1.2  # Hz (Turner et al. 2021)


def region_names():
    c = pd.read_csv(TD / "ito_68_atlas" / "Original_Index_panda_full.csv")
    return dict(zip(c["num"], c["name"]))


def dff(F: np.ndarray, win_s: float = 60.0) -> np.ndarray:
    w = int(win_s * FS) | 1
    base = percentile_filter(F, 8, size=(1, w), mode="nearest")
    return (F - base) / np.maximum(base, 1e-6)


def metrics_one(df: pd.DataFrame, names: dict) -> pd.DataFrame:
    df = df.loc[[i for i in df.index if isinstance(names.get(i), str)]]
    X = dff(df.to_numpy(np.float64))
    Z = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    rows = []
    g_all = Z.mean(0)
    top = g_all >= np.quantile(g_all, 0.95)
    maxlag = 5
    for k, rid in enumerate(df.index):
        others = np.delete(Z, k, 0).mean(0)
        x = Z[k]
        cc = [np.corrcoef(x[max(0, -L): len(x) - max(0, L)],
                          others[max(0, L): len(x) - max(0, -L)])[0, 1] for L in range(-maxlag, maxlag + 1)]
        lag = (np.argmax(cc) - maxlag) / FS   # >0: region at t best matches others at t+lag (region leads)
        rows.append(dict(region=names[rid], sd=X[k].std(), kurt=stats.kurtosis(X[k]),
                         r_global=np.corrcoef(x, others)[0, 1], event=x[top].mean(),
                         lead=lag, ac1=np.corrcoef(x[:-1], x[1:])[0, 1]))
    return pd.DataFrame(rows)


def main():
    names = region_names()
    out = []
    for fn in sorted((TD / "ito_responses").glob("ito_*.pkl")):
        df = load(fn)
        m = metrics_one(df, names)
        m["fly"] = fn.stem
        out.append(m)
    res = pd.concat(out)
    dst = ROOT / "results" / "fly"
    dst.mkdir(parents=True, exist_ok=True)
    res.to_csv(dst / "turner_region_metrics.csv", index=False)
    summ = res.groupby("region").agg(n_flies=("fly", "nunique"), sd=("sd", "median"),
                                     kurt=("kurt", "median"), r_global=("r_global", "median"),
                                     event=("event", "median"), lead=("lead", "median"),
                                     ac1=("ac1", "median"))
    summ.to_csv(dst / "turner_region_summary.csv")
    print(res.fly.nunique(), "flies")
    print(summ.sort_values("event", ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
