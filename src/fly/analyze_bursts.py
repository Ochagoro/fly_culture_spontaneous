"""Analyse the spontaneous-burst runs of bursts.py.

A1 burst statistics per variant x gain
A2 ignition ranking per cell type x hemisphere (pre-onset excess, latency)
A3 recruitment of sleep / clock circuits by bursts
A4 neuropil-level burst participation vs real-fly spontaneous metrics (Turner 2021)
A5 'virtual MEA' metrics (same as for the cultures): out-of-burst spike
   fraction and branching ratio m (MR estimator, 4 ms bins)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from spont import load_connectome, mr_estimate  # noqa: E402
from variants import load_annotations  # noqa: E402
from bursts import groups_of  # noqa: E402

ROOT = HERE.parents[1]
RES = ROOT / "results" / "fly"
BD = RES / "bursts"
VARIANTS = ["fixAL", "typevote", "nomod"]

SLEEP_TYPES = {
    "ER5 (sleep drive)": r"^ER5$",
    "dFB FB6/FB7 (sleep-promoting)": r"^FB[67]",
    "ExR1 helicon": r"^ExR1$",
    "DN1a/DN1p clock": r"^DN1",
    "l-LNv clock (wake)": r"^l-LNv$",
    "PPL1 DAN (wake)": r"^PPL1",
    "MBON": r"^MBON",
    "Kenyon cells": r"^KC",
    "AL projection neurons": None,
}


def load_runs():
    runs = []
    for f in sorted(BD.glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        v = f.name.split("_g")[0]
        runs.append((v, float(d["gain"]), int(f.stem.split("_s")[-1]), d))
    return runs


def burst_stats(runs, n_neurons):
    rows = []
    for v, g, s, d in runs:
        pop = d["pop"].astype(np.float64)
        on, off = d["on"], d["off"]
        inb = np.zeros(len(pop), bool)
        for a, b in zip(on, off):
            inb[a:b] = True
        T = len(pop) / 1000
        rows.append(dict(variant=v, gain=g, seed=s, bursts_per_min=len(on) / T * 60,
                         dur_ms=float(np.median(off - on)) if len(on) else np.nan,
                         peak_per_ms=float(np.median([pop[a:b].max() for a, b in zip(on, off)])) if len(on) else np.nan,
                         frac_spikes_in_bursts=float(pop[inb].sum() / pop.sum()),
                         mean_rate_hz=float(pop.sum() / n_neurons / T)))
    return pd.DataFrame(rows)


def ignition_table(runs, min_bursts=5):
    out = []
    for v in VARIANTS:
        pre, lat, sizes, names = [], [], None, None
        for vv, g, s, d in runs:
            if vv != v or len(d["on"]) < 1 or d["pre_ex"].ndim < 2:
                continue
            pre.append(d["pre_ex"])
            lat.append(d["lat"])
            sizes, names = d["sizes"], d["names"]
        if not pre:
            continue
        pre = np.concatenate(pre)
        lat = np.concatenate(lat)
        if len(pre) < min_bursts:
            continue
        per_neuron = pre.mean(0) / np.maximum(sizes, 1)
        frac_pos = (pre > 0).mean(0)
        lat_m = np.nanmedian(np.where(lat >= 0, lat, np.nan), 0)
        df = pd.DataFrame(dict(variant=v, group=names, n=sizes, pre_excess_per_neuron=per_neuron,
                               pre_excess_total=pre.mean(0), frac_bursts_preceded=frac_pos,
                               latency_ms=lat_m, n_bursts=len(pre)))
        df["type"] = df.group.str.split("|").str[0]
        df["side"] = df.group.str.split("|").str[1]
        df["rank_total"] = df.pre_excess_total.rank(ascending=False)
        out.append(df)
    return pd.concat(out)


def sleep_recruitment(runs, ann, codes, names):
    """Peak in-burst rate / baseline rate per sleep-circuit group (pooled across sides)."""
    rows = []
    ct = ann["cell_type"].fillna("")
    for v, g, s, d in runs:
        if len(d["on"]) == 0 or d["peak"].ndim < 2:
            continue
        peak = d["peak"].mean(0)  # spikes per 2 ms bin, per group
        base = d["base_g"]
        for label, pat in SLEEP_TYPES.items():
            if pat is None:
                m = (ann["cell_class"] == "ALPN").to_numpy()
            else:
                m = ct.str.contains(pat).to_numpy()
            gids = np.unique(codes[m])
            if len(gids) == 0:
                continue
            ratio = (peak[gids].sum() + 1e-9) / (base[gids].sum() + 1e-9)
            rows.append(dict(variant=v, gain=g, seed=s, circuit=label, n_neurons=int(m.sum()),
                             peak_over_base=float(ratio)))
    return pd.DataFrame(rows)


def neuropil_participation(runs, ann, codes, names, con):
    M = sp.load_npz(ROOT / "data" / "flywire" / "neuron_neuropil_frac.npz")
    npn = pd.read_csv(ROOT / "data" / "flywire" / "neuropil_names.csv").iloc[:, 0].tolist()
    rows = []
    for v in VARIANTS:
        acc, n = None, 0
        for vv, g, s, d in runs:
            if vv != v or len(d["on"]) < 3 or d["peak"].ndim < 2:
                continue
            sizes = np.maximum(d["sizes"], 1)
            excess = np.clip(d["peak"].mean(0) - d["base_g"], 0, None) / sizes   # per neuron
            pre = np.clip(d["pre_ex"].mean(0), 0, None) / sizes
            x_burst = excess[codes]
            x_pre = pre[codes]
            part = (M.T @ x_burst) / np.maximum(np.asarray(M.sum(0)).ravel(), 1e-9)
            ign = (M.T @ x_pre) / np.maximum(np.asarray(M.sum(0)).ravel(), 1e-9)
            a = np.vstack([part, ign])
            acc = a if acc is None else acc + a
            n += 1
        if acc is None:
            continue
        acc /= n
        rows.append(pd.DataFrame(dict(variant=v, region=npn, burst_participation=acc[0],
                                      ignition=acc[1])))
    return pd.concat(rows) if rows else pd.DataFrame()


def compare_turner(npart):
    real = pd.read_csv(RES / "turner_region_summary.csv")
    rows = []
    for v, sub in npart.groupby("variant"):
        m = real.merge(sub, on="region")
        for model_col in ["burst_participation", "ignition"]:
            for real_col in ["sd", "kurt", "r_global", "event", "ac1"]:
                r, p = stats.spearmanr(m[model_col], m[real_col])
                rows.append(dict(variant=v, model=model_col, real=real_col, n_regions=len(m), rho=r, p=p))
    return pd.DataFrame(rows)


def virtual_mea(runs):
    rows = []
    for v, g, s, d in runs:
        t = d["ev_t"].astype(np.int64)  # 0.1 ms steps
        if len(t) < 100:
            continue
        T_ms = int(d["t_run"])
        a1 = np.bincount(t // 10, minlength=T_ms)[:T_ms]   # 1 ms bins
        inb = np.zeros(T_ms, bool)
        for a, b in zip(d["on"], d["off"]):
            inb[a:b] = True
        frac_out = 1 - a1[inb].sum() / a1.sum()
        a4 = a1[: T_ms // 4 * 4].reshape(-1, 4).sum(1)
        m, rk, b = mr_estimate(a4[75:], kmax=50)
        # inter-burst only
        keep = ~inb[: T_ms // 4 * 4].reshape(-1, 4).any(1)
        m_ib = np.nan
        if keep.sum() > 1000:
            m_ib, _, _ = mr_estimate(a4[keep], kmax=50)
        rows.append(dict(variant=v, gain=g, seed=s, mea_spikes=len(t), mea_rate_hz=len(t) / 1024 / (T_ms / 1000),
                         frac_spikes_outside_bursts=frac_out, m=m, tau_ms=-4 / np.log(m) if 0 < m < 1 else np.nan,
                         m_interburst=m_ib))
    return pd.DataFrame(rows)


def main():
    con = load_connectome("783")
    ann = load_annotations(con)
    codes, names = groups_of(ann)
    runs = load_runs()
    print(len(runs), "runs")
    bs = burst_stats(runs, con.n)
    bs.to_csv(RES / "burst_stats.csv", index=False)
    print(bs.groupby(["variant", "gain"]).mean(numeric_only=True).round(3).to_string())
    ig = ignition_table(runs)
    ig.to_csv(RES / "ignition_groups.csv", index=False)
    for v, sub in ig.groupby("variant"):
        print("==", v, "n_bursts", sub.n_bursts.iloc[0])
        print(sub.sort_values("pre_excess_total", ascending=False).head(15)[
            ["group", "n", "pre_excess_total", "pre_excess_per_neuron", "frac_bursts_preceded", "latency_ms"]].round(3).to_string())
    sr = sleep_recruitment(runs, ann, codes, names)
    sr.to_csv(RES / "sleep_recruitment.csv", index=False)
    print(sr.groupby(["variant", "circuit"])["peak_over_base"].median().round(2).unstack(0).to_string())
    npart = neuropil_participation(runs, ann, codes, names, con)
    npart.to_csv(RES / "neuropil_participation.csv", index=False)
    cmp = compare_turner(npart)
    cmp.to_csv(RES / "turner_vs_model.csv", index=False)
    print(cmp.round(3).to_string())
    vm = virtual_mea(runs)
    vm.to_csv(RES / "virtual_mea.csv", index=False)
    print(vm.groupby(["variant", "gain"]).mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
