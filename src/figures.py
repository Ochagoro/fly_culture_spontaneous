"""Figures for the paper (Nature style; see style.py). Usage: python figures.py fig1 fig2 ...

Outputs PDF, SVG and PNG to figures/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec

import style
from style import (AQUA, BLUE, DOUBLE, GRID, INK, INK2, MM, MUTED, ORANGE, SEQ_BLUE, SINGLE,
                   VIOLET, YELLOW, MAGENTA, GREEN, RED, panel_label)

ROOT = Path(__file__).resolve().parents[1]
RC = ROOT / "results" / "culture"
RF = ROOT / "results" / "fly"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
style.apply()

CULT_COLORS = [BLUE, ORANGE, AQUA, VIOLET, YELLOW, MAGENTA, GREEN]
VAR_COLORS = {"fixAL": BLUE, "fixALlit": VIOLET, "typevote": ORANGE, "nomod": AQUA}
VAR_LABEL = {"fixAL": "Minimal correction", "fixALlit": "Literature-consistent", "typevote": "Cell-type vote", "nomod": "No fast monoamines"}
VAR_SHORT = {"fixAL": "MC", "fixALlit": "LC", "typevote": "CV", "nomod": "NM"}
V4 = ["fixAL", "fixALlit", "typevote", "nomod"]
HEAT = mcolors.LinearSegmentedColormap.from_list("heat", ["#ffffff", "#fde0c5", "#f6a46b", "#e2572b", "#a3220f", "#4a0b05"])


def save(fig, name):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print("saved", name)


# ------------------------------------------------------------------ Fig. 1
def fig1():
    ex = np.load(RC / "example_dandi_sub-23312_ses-23312-240504.npz")
    maps = np.load(RC / "ignition_maps_dandi.npz")
    ses = pd.read_csv(RC / "sessions_dandi.csv")
    eig = pd.read_csv(RC / "c3_eigen_dandi.csv")
    pairs = pd.read_csv(RC / "ignition_stability_pairs_dandi.csv")

    fig = plt.figure(figsize=(DOUBLE, 128 * MM))
    gs = GridSpec(3, 6, figure=fig, height_ratios=[1.15, 0.5, 1.0], hspace=0.62, wspace=0.9)

    # a: raster
    ax = fig.add_subplot(gs[0, :4])
    t = ex["spike_frame_20kHz"].astype(np.float64) / 20000.0
    ch = ex["spike_channel_index"].astype(np.int64)
    x = ex["channel_x_um"]
    act = ex["channel_active"]
    t0, t1 = 870.0, 930.0
    m = (t >= t0) & (t < t1) & act[ch]
    order = np.argsort(np.argsort(x))
    rng = np.random.default_rng(0)
    show = rng.random(len(x)) < 0.5
    mm = m & show[ch]
    for a, b in zip(ex["burst_start_s"], ex["burst_end_s"]):
        if b > t0 and a < t1:
            ax.axvspan(max(a, t0) - t0, min(b, t1) - t0, color="#e9eff9", lw=0, zorder=0)
    ax.scatter(t[mm] - t0, order[ch[mm]], s=0.05, c=INK, lw=0, rasterized=True)
    ax.set_xlim(0, t1 - t0)
    ax.set_ylabel("Electrode (sorted by x)")
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    panel_label(ax, "a", x=-0.06)
    ax.set_title("Rat cortical culture on a MaxOne HD-MEA (spontaneous activity)", loc="left", fontsize=6.5, color=INK2)

    # b: ignition points
    ax = fig.add_subplot(gs[0, 4:])
    xa, ya = ex["channel_x_um"][act], ex["channel_y_um"][act]
    ax.scatter(xa / 1000, ya / 1000, s=0.6, c=GRID, lw=0, rasterized=True)
    ax.scatter(ex["ign_x_um"] / 1000, ex["ign_y_um"] / 1000, s=1.2, c=RED, lw=0, alpha=0.35, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_xlim(-0.1, 3.95)
    ax.set_ylim(-0.1, 2.2)
    ax.text(0.98, 1.02, f"{len(ex['ign_x_um']):,} bursts", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6, color=INK2)
    panel_label(ax, "b", x=-0.22)

    # c: ignition maps across days, culture 23312
    sel = np.flatnonzero(maps["culture"] == "23312")
    days = maps["day"][sel]
    pick = [d for d in [9, 11, 13, 16, 18, 20] if d in set(days)]
    for k, d in enumerate(pick):
        i = sel[np.flatnonzero(days == d)[0]]
        ax = fig.add_subplot(gs[1, k])
        pm = maps["ign_pmap"][i].T.copy()
        sup = maps["support"][i].T
        pm[~sup] = np.nan
        ax.imshow(pm / np.nanmax(pm), origin="lower", cmap=HEAT, vmin=0, vmax=1,
                  extent=[0, 3.9, 0, 2.1], interpolation="nearest")
        ax.set_facecolor("#f2f2f0")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.4)
            s.set_color(MUTED)
        ax.set_title(f"Day {d}", fontsize=6, pad=2)
        if k == 0:
            panel_label(ax, "c", x=-0.32, y=1.12)
        if k == len(pick) - 1:
            ax.text(1.0, -0.12, "Ignition probability, culture 23312 (normalized; grey, no active electrode)",
                    transform=ax.transAxes, ha="right", va="top", fontsize=5.5, color=INK2)

    # d: stability
    ax = fig.add_subplot(gs[2, 0:2])
    cons = pairs[pairs.kind == "consecutive"]
    betw = pairs[pairs.kind != "consecutive"]
    data = [cons.r_ign.dropna(), betw.r_ign.dropna()]
    parts = ax.violinplot(data, positions=[0, 1], widths=0.7, showextrema=False)
    for pc, c in zip(parts["bodies"], [BLUE, MUTED]):
        pc.set_facecolor(c)
        pc.set_alpha(0.35)
        pc.set_edgecolor("none")
    for k, dd in enumerate(data):
        ax.plot([k - 0.18, k + 0.18], [np.median(dd)] * 2, color=INK, lw=1)
    ax.axhline(0, color=GRID, lw=0.5, zorder=0)
    ax.set_xticks([0, 1], [f"Same culture,\nnext session\n(n = {len(data[0])})", f"Different\ncultures\n(n = {len(data[1]):,})"])
    ax.set_ylabel("Ignition-map correlation (r)")
    panel_label(ax, "d", x=-0.3)

    # e: predictors
    ax = fig.add_subplot(gs[2, 2:4])
    e = eig.dropna(subset=["rho_bin_eig", "rho_bin_ibrate_same"]).copy()
    cults = sorted(e.culture.astype(str).unique())
    for k, c in enumerate(cults):
        s = e[e.culture.astype(str) == c]
        for _, r in s.iterrows():
            ax.plot([0, 1], [r.rho_bin_ibrate_same, r.rho_bin_eig], color=CULT_COLORS[k % 7], lw=0.35, alpha=0.6)
    for k, col in enumerate(["rho_bin_ibrate_same", "rho_bin_eig"]):
        ax.plot([k - 0.15, k + 0.15], [e[col].median()] * 2, color=INK, lw=1.2)
    ax.axhline(0, color=GRID, lw=0.5, zorder=0)
    ax.set_xticks([0, 1], ["Between-burst\nfiring rate", "Leading coupling\neigenvector"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel("Spearman ρ with ignition probability")
    ax.text(0.5, 0.98, f"{len(e)} sessions, {len(cults)} cultures", transform=ax.transAxes, ha="center", va="top",
            fontsize=5.5, color=INK2)
    panel_label(ax, "e", x=-0.3)

    # f: development
    ax = fig.add_subplot(gs[2, 4:6])
    s = ses[~ses.excluded_empty.astype(bool)].copy()
    for k, (c, sub) in enumerate(s.groupby("culture")):
        sub = sub.sort_values("day")
        ax.plot(sub.day, sub.m_all, color=CULT_COLORS[k % 7], lw=0.7, marker="o", ms=1.5)
    ax.set_xlabel("Days since first recording")
    ax.set_ylabel("Branching ratio m")
    ax.set_ylim(0.94, 1.008)
    panel_label(ax, "f", x=-0.32)
    save(fig, "fig1")




# ------------------------------------------------------------------ fly helpers
def _fly_meta():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent / "fly"))
    from spont import load_connectome
    from variants import load_annotations
    from bursts import groups_of
    con = load_connectome("783")
    ann = load_annotations(con)
    codes, names = groups_of(ann)
    return con, ann, codes, names


def _brain_xy():
    xy = np.load(ROOT / "video" / "brain_xy.npy")
    return xy[:, 0] * 4e-3, xy[:, 1] * 4e-3   # 4-nm voxels -> µm


# ------------------------------------------------------------------ Fig. 2
def fig2():
    con, ann, codes, names = _fly_meta()
    bs = pd.read_csv(RF / "burst_stats.csv")
    ig = pd.read_csv(RF / "ignition_fixAL_g2.50.csv")
    d = np.load(RF / "bursts" / "fixAL_g2.50_s0.npz", allow_pickle=True)

    fig = plt.figure(figsize=(DOUBLE, 118 * MM))
    gs = GridSpec(2, 6, figure=fig, height_ratios=[1, 1.15], hspace=0.55, wspace=1.0)

    # a: burst rate vs gain
    ax = fig.add_subplot(gs[0, 0:2])
    for v in V4:
        s = bs[bs.variant == v].groupby("gain").bursts_per_min
        ax.plot(s.mean().index, s.mean().values, color=VAR_COLORS[v], marker="o", ms=2.5, lw=1, label=VAR_LABEL[v])
    ax.axvline(1.0, color=MUTED, lw=0.6, ls="--")
    ax.text(1.03, 5, "Calibrated\nmodel", fontsize=5.5, color=INK2, va="bottom")
    ax.set_xlabel("Synaptic gain (× calibrated)")
    ax.set_ylabel("Spontaneous bursts per min")
    ax.legend(loc="upper left", fontsize=5.5)
    panel_label(ax, "a", x=-0.32)

    # b: virtual MEA raster with whole-brain population rate on top
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # noqa: F401
    sub = gs[0, 2:6].subgridspec(2, 1, height_ratios=[0.35, 1], hspace=0.08)
    axp = fig.add_subplot(sub[0])
    popr = d["pop"].astype(float)
    tt = np.arange(len(popr)) / 1000
    w0, w1 = 2.0, 7.0
    mm_ = (tt >= w0) & (tt < w1)
    axp.plot(tt[mm_] - w0, np.convolve(popr, np.ones(5) / 5, mode="same")[mm_], color=INK, lw=0.4)
    axp.set_xlim(0, w1 - w0)
    axp.set_xticks([])
    axp.spines["bottom"].set_visible(False)
    axp.set_ylabel("Spikes\nper ms", fontsize=5.5)
    axp.set_title("FlyWire whole-brain model, noise + short-term depression, gain 2.5 (minimal correction)", loc="left", fontsize=6.5, color=INK2)
    panel_label(axp, "b", x=-0.08, y=1.0)
    ax = fig.add_subplot(sub[1])
    t = d["ev_t"].astype(np.float64) / 10.0   # ms
    n = d["ev_n"].astype(np.int64)
    mea = d["mea_idx"]
    xs, _ = _brain_xy()
    order = {j: k for k, j in enumerate(mea[np.argsort(xs[mea])])}
    t0, t1 = 2000, 7000
    m = (t >= t0) & (t < t1)
    for a, b in zip(d["on"], d["off"]):
        if b > t0 and a < t1:
            ax.axvspan((max(a, t0) - t0) / 1000, (min(b, t1) - t0) / 1000, color="#e9eff9", lw=0, zorder=0)
    ax.scatter((t[m] - t0) / 1000, [order[j] for j in n[m]], s=0.25, c=INK, lw=0, rasterized=True)
    ax.set_xlim(0, (t1 - t0) / 1000)
    ax.set_ylim(-5, 1030)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylabel("Virtual MEA: 1,024 random\nneurons (sorted by x)")
    ax.set_xlabel("Time (s)")

    # c: brain map of pre-onset excess spikes per neuron
    ax = fig.add_subplot(gs[1, 0:3])
    xs, ys = _brain_xy()
    grp = names[codes]
    e = ig.set_index("group")
    per = (e.excess_spikes_per_burst / e.n).reindex(grp).to_numpy()
    pv = e.p.reindex(grp).to_numpy()
    sig = (pv < 1e-6) & (per > 0.02) & np.isfinite(per)
    ax.scatter(xs[~sig], ys[~sig], s=0.02, c="#d4d4d0", lw=0, rasterized=True)
    o = np.argsort(per[sig])
    sc = ax.scatter(xs[sig][o], ys[sig][o], s=2.5, c=np.log10(per[sig][o]), cmap=HEAT, vmin=-1.7, vmax=0,
                    lw=0, rasterized=True)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    j = np.flatnonzero(np.isin(grp, ["lLN1_bc|L", "lLN1_bc|R", "lLN2X12|L", "lLN2X12|R"]))
    for side_sel in [xs[j] < np.median(xs), xs[j] >= np.median(xs)]:
        jj = j[side_sel]
        if len(jj):
            ax.annotate("AL cholinergic\nlocal neurons", (np.median(xs[jj]), np.median(ys[jj])), xytext=(0, 26),
                        textcoords="offset points", ha="center", fontsize=5.5, color=INK,
                        arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
    cax = ax.inset_axes([0.3, -0.06, 0.4, 0.03])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("Excess spikes per neuron, 20 ms before onset (log10)", fontsize=5.5)
    cb.ax.tick_params(labelsize=5)
    ax.set_title("Where bursts start (frontal view; grey, not enriched)", loc="left", fontsize=6.5, color=INK2)
    panel_label(ax, "c", x=-0.02, y=1.0)

    # d: earliest crossing groups
    ax = fig.add_subplot(gs[1, 4:6])
    cand = ig[(ig.frac_bursts_crossed >= 0.5)].sort_values("cross_ms_median").head(12).iloc[::-1]
    yy = np.arange(len(cand))
    cols = [RED if g.startswith(("lLN1_bc", "lLN2X")) else MUTED for g in cand.group]
    ax.barh(yy, cand.cross_ms_median, color=cols, height=0.6)
    ax.axvline(0, color=INK, lw=0.5)
    ax.set_yticks(yy, [g.replace("|", " ") for g in cand.group], fontsize=5.5)
    ax.set_xlabel("Threshold crossing relative to\nbrain-wide burst onset (ms)")
    ax.text(0.98, 0.04, f"{int(ig.n_bursts.iloc[0])} bursts; groups crossing\nin ≥ 50% of bursts",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5, color=INK2)
    panel_label(ax, "d", x=-0.42)
    save(fig, "fig2")


# ------------------------------------------------------------------ Fig. 3
CLS_ORDER = [("AL cholinergic LNs", None), ("Other AL LNs", None), ("AL PNs", "ALPN"), ("Kenyon cells", "Kenyon_Cell"),
             ("Other", None)]
CLS_COL = {"AL cholinergic LNs": RED, "Other AL LNs": "#f2a58f", "AL PNs": ORANGE, "Kenyon cells": BLUE, "Other": GRID}


def _eln_mask(ann):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from variants import transmitter_table
    nt = transmitter_table(ann)
    return ((ann["cell_class"] == "ALLN") & (nt == "acetylcholine")).to_numpy()


def _class_of(ann, eln):
    cls = np.full(len(ann), "Other", dtype=object)
    cc = ann["cell_class"].to_numpy()
    cls[cc == "ALLN"] = "Other AL LNs"
    cls[eln] = "AL cholinergic LNs"
    cls[cc == "ALPN"] = "AL PNs"
    cls[cc == "Kenyon_Cell"] = "Kenyon cells"
    return cls


def fig3():
    con, ann, codes, names = _fly_meta()
    eln = _eln_mask(ann)
    cls = _class_of(ann, eln)
    fig = plt.figure(figsize=(DOUBLE, 112 * MM))
    gs = GridSpec(2, 6, figure=fig, hspace=0.75, wspace=1.1)

    # a: Perron share by class
    ax = fig.add_subplot(gs[0, 0:2])
    vs = V4
    bottom = np.zeros(len(vs))
    for lab in CLS_COL:
        vals = []
        for v in vs:
            d = pd.read_parquet(RF / f"structure_neurons_{v}.parquet")
            vals.append(d.perron.to_numpy()[cls == lab].sum() / d.perron.sum())
        ax.bar(range(len(vs)), vals, bottom=bottom, color=CLS_COL[lab], width=0.6, label=lab, lw=0)
        bottom += np.array(vals)
    ax.set_xticks(range(len(vs)), [VAR_SHORT[v] for v in vs], fontsize=5.5)
    ax.set_ylabel("Share of Perron-mode weight")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    panel_label(ax, "a", x=-0.35)

    # b: ladder
    ax = fig.add_subplot(gs[0, 3:6])
    lad = pd.read_csv(RF / "ladder.csv")
    for v in [v for v in vs if v in set(lad.variant)]:
        s = lad[lad.variant == v].groupby("gain")[["n_core", "n_AL_LN", "n_KC"]].mean()
        ax.plot(s.index, s.n_core + 1, color=VAR_COLORS[v], lw=1, marker="o", ms=2, label=VAR_LABEL[v] + " (all)")
        ax.plot(s.index, s.n_AL_LN + 1, color=VAR_COLORS[v], lw=0.8, ls="--", marker="s", ms=1.8,
                label=VAR_LABEL[v] + " (AL LNs)")
    ax.set_yscale("log")
    ax.axvline(1, color=MUTED, lw=0.5, ls=":")
    ax.set_xlabel("Synaptic gain (× calibrated)")
    ax.set_ylabel("Self-sustaining neurons + 1")
    ax.legend(fontsize=4.8, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.32))
    panel_label(ax, "b", x=-0.2)

    # c: spontaneous rates
    ax = fig.add_subplot(gs[1, 0:2])
    er = pd.read_csv(RF / "eln_rates.csv")
    for v in [v for v in vs if v in set(er.variant)]:
        s = er[er.variant == v]
        ax.plot(s.gain, s.lLN1_bc, color=VAR_COLORS[v], marker="o", ms=2, lw=1)
        ax.plot(s.gain, s.brain_median, color=VAR_COLORS[v], lw=0.6, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("Synaptic gain (× calibrated)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.text(1.0, 60, "lLN1_bc", fontsize=5.5, color=INK2)
    ax.text(1.0, 0.27, "Brain median", fontsize=5.5, color=INK2)
    panel_label(ax, "c", x=-0.35)

    # d: silencing with matched controls
    ax = fig.add_subplot(gs[1, 2:5])
    mt = pd.read_csv(RF / "silencing_matched.csv")
    mt["set"] = mt.silenced.str.replace(r"_#\d", "", regex=True)
    sc = pd.read_csv(RF / "silencing_confirm.csv")
    sc["set"] = sc.silenced.str.replace(r"_#\d", "", regex=True)
    sc = sc[sc.set.isin(["random_111", "other_ALLN_111"])]

    def rel(df):
        it = df[df.set == "intact"].groupby(["variant", "gain"]).bursts_per_min.mean()
        return df.assign(rel=lambda d: d.bursts_per_min / [it.get((a, b), np.nan) for a, b in zip(d.variant, d.gain)])
    mt = rel(mt)
    sc = rel(pd.concat([sc, pd.read_csv(RF / "silencing_confirm.csv").assign(
        set=lambda d: d.silenced.str.replace(r"_#\d", "", regex=True)).query("set == 'intact'")]))
    sets = [("lLN1_bc", "lLN1_bc", mt), ("degree_matched_30", "Degree-\nmatched", mt), ("high_drive_30", "High-\ndrive", mt),
            ("AL_cholinergic_LNs", "AL chol.\nLNs", mt), ("degree_matched_111", "Degree-\nmatched", mt),
            ("high_drive_111", "High-\ndrive", mt), ("random_111", "Random", sc), ("other_ALLN_111", "Other\nAL LNs", sc)]
    for k, (key, lab, src) in enumerate(sets):
        vv = [v for v in V4 if len(src[(src.set == key) & (src.variant == v)])]
        for j, v in enumerate(V4):
            s_ = src[(src.set == key) & (src.variant == v)]
            if not len(s_):
                continue
            vals = s_.groupby(["silenced", "gain"]).rel.mean().to_numpy()
            x = k + (j - 1.5) * 0.19
            strong = key in ("lLN1_bc", "AL_cholinergic_LNs")
            ax.bar(x, np.mean(vals), width=0.17, color=VAR_COLORS[v], alpha=0.9 if strong else 0.35, lw=0)
            ax.scatter(np.full(len(vals), x), vals, s=1.5, color=INK, lw=0, zorder=3)
    ax.axhline(1, color=INK, lw=0.5)
    ax.axvline(2.5, color=GRID, lw=0.6)
    ax.text(1, 1.95, "30 neurons", ha="center", fontsize=5.5, color=INK2)
    ax.text(5, 1.95, "111 neurons", ha="center", fontsize=5.5, color=INK2)
    ax.set_ylim(0, 2.05)
    ax.set_xticks(range(len(sets)), [l for _, l, _ in sets], fontsize=5)
    ax.set_ylabel("Bursts relative to intact")
    panel_label(ax, "d", x=-0.17)

    # e: brain-wide mean rate change (matched runs, highest gain)
    ax = fig.add_subplot(gs[1, 5])
    for j, v in enumerate(V4):
        s_ = mt[mt.variant == v]
        if not len(s_):
            continue
        g = s_.gain.max()
        r = s_[s_.gain == g].groupby("set").brain_rate.mean()
        ax.bar(j - 0.18, 100 * (r["lLN1_bc"] / r["intact"] - 1), width=0.34, color=VAR_COLORS[v])
        ax.bar(j + 0.18, 100 * (r["degree_matched_30"] / r["intact"] - 1), width=0.34, color=VAR_COLORS[v], alpha=0.35)
    ax.axhline(0, color=INK, lw=0.5)
    ax.set_xticks(range(4), [VAR_SHORT[v] for v in V4], fontsize=5.5)
    ax.set_ylabel("Change in mean rate (%)")
    ax.set_title("dark, lLN1_bc\nlight, degree-matched", fontsize=4.8, color=INK2, loc="left")
    panel_label(ax, "e", x=-0.7)
    save(fig, "fig3")


# ------------------------------------------------------------------ Fig. 4
def fig4():
    fig = plt.figure(figsize=(DOUBLE, 62 * MM))
    gs = GridSpec(1, 4, figure=fig, wspace=0.75)
    summ = pd.read_csv(RF / "turner_region_summary.csv")
    ax = fig.add_subplot(gs[0, 0])
    al = summ.region.str.startswith("AL_")
    ax.scatter(summ.sd[~al], summ.ac1[~al], s=5, color=MUTED, lw=0)
    ax.scatter(summ.sd[al], summ.ac1[al], s=12, color=ORANGE, lw=0)
    for _, r in summ[al].iterrows():
        ax.annotate(r.region.replace("_", " "), (r.sd, r.ac1), xytext=(3, -6), textcoords="offset points", fontsize=5.5, color=ORANGE)
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([0.02, 0.05, 0.1, 0.2]))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Fluctuation size (s.d. of ΔF/F)")
    ax.set_ylabel("Lag-1 autocorrelation")
    ax.set_title("74 regions, 20 flies", loc="left", fontsize=6, color=INK2)
    panel_label(ax, "a", x=-0.38)

    ax = fig.add_subplot(gs[0, 1])
    x = pd.read_csv(RF / "turner_AL_vs_others.csv")
    rng = np.random.default_rng(1)
    for k, (a, b, lab) in enumerate([("ac1_AL", "ac1_oth", "Autocorrelation"), ("sd_AL", "sd_oth", "Fluctuation size"),
                                     ("rg_AL", "rg_oth", "Global coupling")]):
        rr = np.log2(x[a] / x[b])
        ax.scatter(k + rng.uniform(-0.12, 0.12, len(rr)), rr, s=5, color=ORANGE, lw=0, alpha=0.8)
        ax.plot([k - 0.2, k + 0.2], [rr.median()] * 2, color=INK, lw=1)
        ax.text(k, 1.55, f"{(rr > 0).sum()}/{len(rr)}", ha="center", fontsize=5.5, color=INK2)
    ax.axhline(0, color=INK, lw=0.5)
    ax.set_ylim(-1.2, 1.7)
    ax.set_xticks(range(3), ["Autocorr.", "Size", "Global r"], fontsize=5.5)
    ax.set_ylabel("Antennal lobes / other regions\n(log2, per fly)")
    panel_label(ax, "b", x=-0.38)

    ax = fig.add_subplot(gs[0, 2])
    eig = pd.read_csv(RC / "c3_eigen_dandi.csv").dropna(subset=["rho_bin_eig", "rho_bin_ibrate_same"])
    fe = pd.read_csv(RF / "functional_eig_summary.csv")
    fe = fe[fe.tag.str.startswith("fixAL_g2.50")]
    items = [("Culture: firing rate", eig.rho_bin_ibrate_same, MUTED, "c"),
             ("Culture: activity mode", eig.rho_bin_eig, BLUE, "c"),
             ("Fly: firing rate", fe.rho_rate_ib, MUTED, "f"),
             ("Fly: activity mode", fe.rho_eig, BLUE, "f"),
             ("Fly: wiring mode", fe.rho_perron_per_neuron, RED, "f")]
    for k, (lab, vals, col, kind) in enumerate(items):
        y = len(items) - 1 - k
        if kind == "c":
            q1, q2, q3 = np.percentile(vals, [25, 50, 75])
            ax.barh(y, q2, color=col, height=0.6)
            ax.plot([q1, q3], [y, y], color=INK, lw=0.8)
        else:
            ax.barh(y, np.median(vals), color=col, height=0.6, alpha=0.6)
            ax.scatter(vals, np.full(len(vals), y), s=5, color=INK, lw=0, zorder=3)
    ax.set_yticks(range(len(items))[::-1], [i[0] for i in items], fontsize=5.5)
    ax.axvline(0, color=INK, lw=0.5)
    ax.set_xlabel("Spearman ρ with ignition")
    panel_label(ax, "c", x=-0.95)

    ax = fig.add_subplot(gs[0, 3])
    ses = pd.read_csv(RC / "sessions_dandi.csv")
    ses = ses[~ses.excluded_empty.astype(bool)]
    ax.scatter(ses.frac_out_of_burst, ses.m_all, s=4, color=BLUE, alpha=0.5, lw=0, label="Cultures (HD-MEA)")
    vm = pd.read_csv(RF / "virtual_mea.csv")
    s = vm[vm.variant == "fixAL"].groupby("gain").mean(numeric_only=True)
    ax.plot(s.frac_spikes_outside_bursts, s.m, color=RED, marker="D", ms=3, lw=0.8, label="Fly model (gain 1–2.5)")
    for gg, r in s.iterrows():
        if gg in (1.0, 2.5):
            ax.annotate(f"g = {gg:g}", (r.frac_spikes_outside_bursts, r.m), xytext=(-30, -2 if gg == 1.0 else 2), textcoords="offset points", fontsize=5, color=RED)
    ax.set_xlabel("Fraction of spikes outside bursts")
    ax.set_ylabel("Branching ratio m")
    ax.set_ylim(0.94, 1.01)
    ax.legend(fontsize=5, loc="lower left")
    panel_label(ax, "d", x=-0.38)
    save(fig, "fig4")


# ------------------------------------------------------------------ Extended Data
def ed1():
    d = pd.read_csv(RC / "sessions_dandi.csv")
    d = d[~d.excluded_empty.astype(bool)]
    w = pd.read_csv(RC / "sessions_wagenaar.csv")
    w = w[~w.excluded_empty.astype(bool)]
    sens = pd.read_csv(RC / "sensitivity_dandi.csv").set_index("variant")
    sensw = pd.read_csv(RC / "sensitivity_wagenaar.csv").set_index("variant")
    fig = plt.figure(figsize=(DOUBLE, 95 * MM))
    gs = GridSpec(2, 4, figure=fig, hspace=0.7, wspace=0.65)
    for row, (df, lab, xl) in enumerate([(d, "HD-MEA (7 cultures)", "Days since first recording"),
                                         (w, "59-electrode (30 cultures)", "Days in vitro")]):
        for col, (key, yl, logy) in enumerate([("burst_rate_per_min", "Bursts per min", True),
                                               ("frac_out_of_burst", "Spikes outside bursts", False),
                                               ("m_all", "Branching ratio m", False)]):
            ax = fig.add_subplot(gs[row, col])
            if row == 0:
                for k, (c, sub) in enumerate(df.groupby("culture")):
                    sub = sub.sort_values("day")
                    ax.plot(sub.day, sub[key] + (0.01 if logy else 0), color=CULT_COLORS[k % 7], lw=0.6, marker="o", ms=1.2)
            else:
                g = df.groupby("day")[key]
                ax.scatter(df.day, df[key] + (0.01 if logy else 0), s=1.5, color=MUTED, lw=0, alpha=0.5)
                ax.plot(g.median().index, g.median().values + (0.01 if logy else 0), color=BLUE, lw=1)
            if logy:
                ax.set_yscale("log")
            if key == "m_all":
                ax.set_ylim(0.93, 1.01)
            ax.set_xlabel(xl, fontsize=6)
            ax.set_ylabel(yl)
            if col == 0:
                ax.set_title(lab, loc="left", fontsize=6.5, color=INK2)
                panel_label(ax, "ab"[row], x=-0.35)
    ax = fig.add_subplot(gs[0, 3])
    vals = [d.h2_m_ib - d.h1_m_ib, w.h2_m_ib - w.h1_m_ib]
    for k, v in enumerate(vals):
        v = v.dropna()
        v = v[np.abs(v) < 0.05]
        ax.hist(v, bins=40, color=[BLUE, ORANGE][k], alpha=0.5, label=["HD-MEA", "59-electrode"][k], density=True)
    ax.axvline(0, color=INK, lw=0.5)
    ax.set_xlabel("Δm between-burst (2nd − 1st 15 min)", fontsize=6)
    ax.set_ylabel("Density")
    ax.legend(fontsize=5)
    panel_label(ax, "c", x=-0.35)
    ax = fig.add_subplot(gs[1, 3])
    labs = ["thr_x0.5", "base", "thr_x2"]
    for k, (S, c, lab) in enumerate([(sens, BLUE, "HD-MEA"), (sensw, ORANGE, "59-electrode")]):
        if all(l in S.index for l in labs):
            ax.plot([0.5, 1, 2], S.loc[labs, "frac_out"].to_numpy(), color=c, marker="o", ms=2.5, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.5, 1, 2], ["×0.5", "×1", "×2"])
    ax.set_xlabel("Burst threshold")
    ax.set_ylabel("Spikes outside bursts")
    ax.legend(fontsize=5)
    panel_label(ax, "d", x=-0.35)
    save(fig, "ed1")


def _early_list(ax, tag, title):
    ig = pd.read_csv(RF / f"ignition_{tag}.csv")
    cand = ig[(ig.frac_bursts_crossed >= 0.5)].sort_values("cross_ms_median").head(12).iloc[::-1]
    yy = np.arange(len(cand))
    cols = [RED if g.startswith(("lLN1_bc", "lLN2X")) else MUTED for g in cand.group]
    ax.barh(yy, cand.cross_ms_median, color=cols, height=0.6)
    ax.axvline(0, color=INK, lw=0.5)
    ax.set_yticks(yy, [f"{g.replace('|', ' ')} ({n})" for g, n in zip(cand.group, cand.n)], fontsize=5)
    ax.set_xlabel("Crossing vs onset (ms)")
    ax.set_title(title + f" ({int(ig.n_bursts.iloc[0])} bursts)", loc="right", fontsize=6, color=INK2)


def ed2():
    lad = pd.read_csv(RF / "ladder.csv")
    lc = pd.read_csv(RF / "ladder_classes.csv")
    fig = plt.figure(figsize=(DOUBLE, 120 * MM))
    gs = GridSpec(2, 6, figure=fig, wspace=2.2, hspace=0.6)
    ax = fig.add_subplot(gs[0, 0:3])
    for v, c in [("shiu", MUTED)] + [(v, VAR_COLORS[v]) for v in V4]:
        if v not in set(lad.variant):
            continue
        s_ = lad[lad.variant == v].groupby("gain").n_core.mean()
        ax.plot(s_.index, s_.values + 1, color=c, marker="o", ms=2, lw=1, label={"shiu": "Published signs"}.get(v, VAR_LABEL.get(v)))
    ax.set_yscale("log")
    ax.axvline(1, color=MUTED, lw=0.5, ls=":")
    ax.set_xlabel("Synaptic gain")
    ax.set_ylabel("Self-sustaining neurons + 1")
    ax.legend(fontsize=5)
    panel_label(ax, "a", x=-0.2)
    ax = fig.add_subplot(gs[0, 3:6])
    rows = []
    for v in ["shiu"] + V4:
        mc = lad[lad.variant == v].groupby("gain").n_core.mean()
        mc = mc[mc > 20]
        if not len(mc):
            continue
        g0 = mc.index.min()
        sub = lc[(lc.variant == v) & (lc.gain == g0)].groupby("cell_class").n.sum().sort_values(ascending=False)
        rows.append((v, g0, sub / sub.sum()))
    allc = ["ALLN", "ALPN", "Kenyon_Cell", "LHLN"]
    for j, (v, g0, sub) in enumerate(rows):
        bottom = 0
        for k, c in enumerate(allc + ["other"]):
            val = sub.get(c, 0) if c != "other" else 1 - sum(sub.get(x, 0) for x in allc)
            ax.bar(j, val, bottom=bottom, color=[RED, ORANGE, BLUE, AQUA, GRID][k], width=0.6,
                   label=["AL LNs", "AL PNs", "Kenyon cells", "LH LNs", "Other"][k] if j == 0 else None)
            bottom += val
        ax.text(j, 1.02, f"g={g0:g}", ha="center", fontsize=5)
    ax.set_xticks(range(len(rows)), [{"shiu": "Published"}.get(r[0], VAR_SHORT.get(r[0])) for r in rows], fontsize=5.5)
    ax.set_ylabel("Share of first core (> 20 neurons)")
    ax.legend(fontsize=4.8, loc="upper left", bbox_to_anchor=(1, 1))
    panel_label(ax, "b", x=-0.2)
    for k, (tag, title, lab) in enumerate([("fixALlit_g2.50", "Literature-consistent, g = 2.5", "c"),
                                          ("typevote_g2.00", "Cell-type vote, g = 2", "d"),
                                          ("nomod_g2.50", "No fast monoamines, g = 2.5", "e")]):
        ax = fig.add_subplot(gs[1, 2 * k: 2 * k + 2])
        if (RF / f"ignition_{tag}.csv").exists():
            _early_list(ax, tag, title)
        panel_label(ax, lab, x=-0.55, y=1.1)
    save(fig, "ed2")


def ed3():
    fig = plt.figure(figsize=(DOUBLE, 62 * MM))
    gs = GridSpec(1, 4, figure=fig, wspace=0.9)
    ax = fig.add_subplot(gs[0, 0])
    nl = pd.read_csv(RF / "crossing_null.csv")
    nl = nl[nl.file.str.startswith("spikes_fixAL_")]
    tags = [("spikes_fixAL_g2.50", "Base"), ("spikes_fixAL_g2.50-h0.15", "Drive ×0.5"), ("spikes_fixAL_g2.50-h0.6", "Drive ×2"),
            ("spikes_fixAL_g2.50-U0.1tau1000", "Stronger STD"), ("spikes_fixAL_g2.50-seed6", "Seed 2"),
            ("spikes_fixAL_g2.50-seed7", "Seed 3")]
    for k, (t, lab) in enumerate(tags):
        sub = nl[nl.file == t]
        for j, (g, c) in enumerate([("lLN1_bc|L", BLUE), ("lLN1_bc|R", ORANGE), ("lLN2X12|L", AQUA)]):
            r = sub[sub.group == g]
            if not len(r):
                continue
            r = r.iloc[0]
            x = k + (j - 1) * 0.22
            ax.plot([x, x], [r.frac_before_random, r.frac_before_burst], color=c, lw=0.8)
            ax.scatter(x, r.frac_before_burst, s=10, color=c, lw=0, zorder=3, marker="o")
            ax.scatter(x, r.frac_before_random, s=8, facecolor="white", edgecolor=c, lw=0.6, zorder=3)
            if r.p_fisher < 0.05:
                ax.text(x, r.frac_before_burst + 0.03, "*", ha="center", fontsize=6, color=c)
    ax.set_xticks(range(len(tags)), [l for _, l in tags], rotation=35, ha="right", fontsize=5.3)
    ax.set_ylabel("Threshold crossed in 50 ms before\nburst (filled) or random time (open)")
    ax.set_ylim(0, 0.85)
    ax.text(0.02, 0.97, "blue, lLN1_bc L; orange, lLN1_bc R;\naqua, lLN2X12 L; *, Fisher P < 0.05", transform=ax.transAxes,
            fontsize=4.6, color=INK2, va="top")
    panel_label(ax, "a", x=-0.42)
    ctl = RF / "eln_pn_control.csv"
    c = pd.read_csv(ctl)
    ax = fig.add_subplot(gs[0, 1])
    for j, v in enumerate(["fixAL", "nomod"]):
        s_ = c[c.variant == v].set_index("test").value
        ax.bar(j - 0.18, s_.get("perron_share_AL_cholinergic_LNs", np.nan), width=0.34, color=RED)
        ax.bar(j + 0.18, s_.get("perron_share_KC", np.nan), width=0.34, color=BLUE)
    ax.set_xticks([0, 1], ["MC", "NM"])
    ax.set_ylabel("Perron weight share\n(no eLN→PN synapses)")
    ax.text(0.02, 0.95, "red, AL cholinergic LNs\nblue, Kenyon cells", transform=ax.transAxes, fontsize=4.8, color=INK2, va="top")
    panel_label(ax, "b", x=-0.45)
    ax = fig.add_subplot(gs[0, 2])
    for j, v in enumerate(["fixAL", "nomod"]):
        s_ = c[c.variant == v].groupby("test").value.mean()
        for k, (nm, col) in enumerate([("intact", INK), ("lLN1_bc", RED), ("random30", MUTED)]):
            ax.bar(j + (k - 1) * 0.25, s_.get(f"bursts_per_min_{nm}", np.nan), width=0.23, color=col,
                   label=["Intact", "lLN1_bc silenced", "30 random silenced"][k] if j == 0 else None)
    ax.set_xticks([0, 1], ["MC", "NM"])
    ax.set_ylabel("Bursts per min (g = 2.5,\nno eLN→PN synapses)")
    ax.legend(fontsize=4.8, loc="upper right")
    panel_label(ax, "c", x=-0.45)
    ax = fig.add_subplot(gs[0, 3])
    tm = pd.read_csv(RF / "perron_two_modes.csv")
    for j, v in enumerate(V4):
        s_ = tm[tm.variant == v].set_index("mode")
        for m, dx in [(1, -0.2), (2, 0.2)]:
            ax.bar(j + dx, s_.loc[m, "left"], width=0.36, color=BLUE)
            ax.bar(j + dx, s_.loc[m, "right"], bottom=s_.loc[m, "left"], width=0.36, color=ORANGE)
            ax.text(j + dx, 1.01, str(m), ha="center", fontsize=5)
    ax.set_xticks(range(4), [VAR_SHORT[v] for v in V4])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Weight of leading modes 1 and 2\n(blue, left; orange, right)")
    panel_label(ax, "d", x=-0.45)
    save(fig, "ed3")


def ed4():
    fig = plt.figure(figsize=(DOUBLE, 62 * MM))
    gs = GridSpec(1, 4, figure=fig, wspace=1.0)
    ax = fig.add_subplot(gs[0, 0])
    sr = pd.read_csv(RF / "sleep_recruitment_norm.csv")
    t = sr.groupby(["circuit", "variant"]).pct.median().unstack()
    order = [c for c in t.index if c != "AL projection neurons"]
    yy = np.arange(len(order))
    for j, v in enumerate(["fixAL", "typevote", "nomod"]):
        ax.scatter(t.loc[order, v], yy + (j - 1) * 0.18, s=8, color=VAR_COLORS[v], lw=0, label=VAR_LABEL[v])
    ax.axvline(0.5, color=MUTED, lw=0.5, ls="--")
    ax.set_yticks(yy, [o.replace(" (", "\n(") for o in order], fontsize=4.8)
    ax.set_xlabel("Percentile vs random types")
    ax.set_xlim(-0.02, 1.02)
    ax.legend(fontsize=4.5, loc="lower left", bbox_to_anchor=(0, 1.0))
    panel_label(ax, "a", x=-1.0)
    ax = fig.add_subplot(gs[0, 1])
    cmp = pd.read_csv(RF / "turner_vs_model.csv")
    piv = cmp.pivot_table(index=["model", "real"], columns="variant", values="rho")
    im = ax.imshow(piv.to_numpy(), cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_yticks(range(len(piv)), [f"{a.replace('burst_participation', 'participation')} × {b}" for a, b in piv.index], fontsize=4.5)
    ax.set_xticks(range(piv.shape[1]), [{"fixAL": "MC", "typevote": "CV", "nomod": "NM"}[c] for c in piv.columns], fontsize=5.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.05)
    cb.ax.tick_params(labelsize=5)
    cb.set_label("Spearman ρ (73 regions)", fontsize=5)
    panel_label(ax, "b", x=-1.1)
    ax = fig.add_subplot(gs[0, 2])
    fe = pd.read_csv(RF / "functional_eig_summary.csv")
    fe = fe[fe.n_bursts >= 10]
    x = np.arange(len(fe))
    for k, (col, c, lab) in enumerate([("rho_rate_ib", MUTED, "Rate"), ("rho_eig", BLUE, "Activity mode"),
                                       ("rho_perron_per_neuron", RED, "Wiring mode")]):
        ax.bar(x + (k - 1) * 0.25, fe[col], width=0.23, color=c, label=lab)
    ax.axhline(0, color=INK, lw=0.5)
    ax.set_xticks(x, [t.replace("_g", "\ng=") for t in fe.tag], fontsize=5)
    ax.set_ylabel("Spearman ρ with ignition")
    ax.legend(fontsize=5)
    panel_label(ax, "c", x=-0.45)
    ax = fig.add_subplot(gs[0, 3])
    vm = pd.read_csv(RF / "virtual_mea.csv")
    for v in ["fixAL", "typevote", "nomod"]:
        s = vm[vm.variant == v].groupby("gain").mean(numeric_only=True)
        ax.plot(s.index, s.m, color=VAR_COLORS[v], marker="o", ms=2, lw=1, label=VAR_LABEL[v])
    ax.set_xlabel("Synaptic gain")
    ax.set_ylabel("Virtual-MEA branching ratio m")
    ax.legend(fontsize=4.8)
    panel_label(ax, "d", x=-0.45)
    save(fig, "ed4")



if __name__ == "__main__":
    for name in sys.argv[1:]:
        globals()[name]()
