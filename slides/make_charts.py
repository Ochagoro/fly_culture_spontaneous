"""Slide-sized charts for slides/slides.html (1280 x 720 px slides).

Run: .venv/bin/python slides/make_charts.py [name ...]
Reads results/ and data/ (read-only); writes slides/img/*.png only.
Figure sizes are given in slide pixels; text sizes in slide pixels (>= 16 px).
Colours follow the Digital Agency design-system tokens (slides/dads/tokens.css):
minimal correction = Blue-800, cell-type vote = Orange-800, no fast monoamines = Green-700,
hub (AL cholinergic local neurons) = Red-900, neutral = Solid Gray.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
RC, RF = ROOT / "results" / "culture", ROOT / "results" / "fly"
OUT = ROOT / "slides" / "img"
OUT.mkdir(exist_ok=True)

MC, CV, NM, MCL = "#0031D8", "#C74700", "#1D8B56", "#6F23D0"
HUB, HUB_LIGHT = "#CE0000", "#FFBBBB"
BLUE, BLUE_LIGHT = "#0031D8", "#9DB7F9"
INK, INK2, NEUTRAL, LIGHT, GRID, FAINT = "#1A1A1A", "#4D4D4D", "#7F7F7F", "#B3B3B3", "#CCCCCC", "#E6E6E6"
VAR_COL = {"fixAL": MC, "fixALlit": MCL, "typevote": CV, "nomod": NM}
VARS = ["fixAL", "fixALlit", "typevote", "nomod"]
VAR_JA = {"fixAL": "最小補正", "fixALlit": "文献整合の最小補正", "typevote": "細胞タイプ多数決", "nomod": "モノアミンなし"}
VAR_AB = {"fixAL": "MC", "fixALlit": "LC", "typevote": "CV", "nomod": "NM"}
HEAT = mcolors.LinearSegmentedColormap.from_list("heat", ["#ffffff", "#ffdfca", "#ff8d44", "#e25100", "#ac3e00", "#5a1e00"])
PX = 0.72  # points per slide pixel at 100 dpi


def px(n):
    return n * PX


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
    "font.size": px(18),
    "axes.labelsize": px(18),
    "xtick.labelsize": px(17),
    "ytick.labelsize": px(17),
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "axes.unicode_minus": True,
    "legend.frameon": False,
    "legend.fontsize": px(16),
    "savefig.dpi": 200,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "lines.solid_capstyle": "round",
})


def fig_px(w, h):
    return plt.figure(figsize=(w / 100, h / 100), dpi=100)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("saved", name)


def thousands(axis):
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))


# ------------------------------------------------------------------ fly helpers (lazy)
_META = {}


def fly_meta():
    if not _META:
        sys.path.insert(0, str(ROOT / "src"))
        sys.path.insert(0, str(ROOT / "src" / "fly"))
        from sim import load_connectome
        from variants import load_annotations, transmitter_table
        from bursts import groups_of
        con = load_connectome("783")
        ann = load_annotations(con)
        codes, names = groups_of(ann)
        nt = transmitter_table(ann)
        eln = ((ann["cell_class"] == "ALLN") & (nt == "acetylcholine")).to_numpy()
        xy = np.load(ROOT / "video" / "brain_xy.npy") * 4e-3
        _META.update(ann=ann, codes=codes, names=names, eln=eln, x=xy[:, 0], y=xy[:, 1])
    return _META


def classes():
    m = fly_meta()
    ann, eln = m["ann"], m["eln"]
    cls = np.full(len(ann), "other", dtype=object)
    cc = ann["cell_class"].to_numpy()
    cls[cc == "ALLN"] = "ALLN_other"
    cls[eln] = "ALLN_chol"
    cls[cc == "ALPN"] = "ALPN"
    cls[cc == "Kenyon_Cell"] = "KC"
    return cls


# ================================================================== CULTURE
def culture_raster():
    ex = np.load(RC / "example_dandi_sub-23312_ses-23312-240504.npz")
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
    fig = fig_px(700, 300)
    ax = fig.add_axes([0.07, 0.2, 0.9, 0.76])
    for a, b in zip(ex["burst_start_s"], ex["burst_end_s"]):
        if b > t0 and a < t1:
            ax.axvspan(max(a, t0) - t0, min(b, t1) - t0, color="#D9E6FF", lw=0, zorder=0)
    ax.scatter(t[mm] - t0, order[ch[mm]], s=0.12, c=INK, lw=0, rasterized=True)
    ax.set_xlim(0, t1 - t0)
    ax.set_ylim(order[act].min() - 5, order[act].max() + 5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylabel("電極", labelpad=8)
    ax.set_xlabel("時間 (s)")
    save(fig, "culture_raster")


def ign_maps():
    maps = np.load(RC / "ignition_maps_dandi.npz")
    sel = np.flatnonzero(maps["culture"] == "23312")
    days = maps["day"][sel]
    pick = [d for d in [9, 11, 13, 16, 18, 20] if d in set(days)]
    fig = fig_px(600, 290)
    W, H = 0.31, 0.36
    for k, d in enumerate(pick):
        i = sel[np.flatnonzero(days == d)[0]]
        r, c = divmod(k, 3)
        ax = fig.add_axes([0.01 + c * 0.333, 0.53 - r * 0.49, W, H])
        pm = maps["ign_pmap"][i].T.copy()
        sup = maps["support"][i].T
        pm[~sup] = np.nan
        ax.imshow(pm / np.nanmax(pm), origin="lower", cmap=HEAT, vmin=0, vmax=1, extent=[0, 3.9, 0, 2.1],
                  interpolation="nearest")
        ax.set_facecolor("#F2F2F2")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.8)
            s.set_color(LIGHT)
        ax.set_title(f"{d} 日目", fontsize=px(17), pad=3)
    save(fig, "ign_maps")


def ign_stability():
    pairs = pd.read_csv(RC / "ignition_stability_pairs_dandi.csv")
    cons = pairs[pairs.kind == "consecutive"].r_ign.dropna()
    betw = pairs[pairs.kind != "consecutive"].r_ign.dropna()
    fig = fig_px(400, 360)
    ax = fig.add_axes([0.22, 0.22, 0.74, 0.72])
    parts = ax.violinplot([cons, betw], positions=[0, 1], widths=0.75, showextrema=False)
    for pc, c in zip(parts["bodies"], [BLUE_LIGHT, FAINT]):
        pc.set_facecolor(c)
        pc.set_alpha(1)
        pc.set_edgecolor("none")
    for k, dd in enumerate([cons, betw]):
        med = np.median(dd)
        ax.plot([k - 0.22, k + 0.22], [med] * 2, color=INK, lw=2.5)
        ax.text(k + 0.27, med, f"{med:.2f}".replace("-", "−"), va="center", fontsize=px(18), fontweight="bold",
                color=BLUE if k == 0 else INK2)
    ax.axhline(0, color=GRID, lw=1, zorder=0)
    ax.set_xticks([0, 1], [f"同じ培養の\n次の記録日\n(n = {len(cons)})", f"別の培養\n\n(n = {len(betw):,})"])
    ax.set_xlim(-0.55, 1.65)
    ax.set_ylim(-0.75, 1.05)
    ax.set_ylabel("点火マップの相関 r")
    save(fig, "ign_stability")


def predictors():
    eig = pd.read_csv(RC / "c3_eigen_dandi.csv").dropna(subset=["rho_bin_eig", "rho_bin_ibrate_same"])
    fig = fig_px(470, 410)
    ax = fig.add_axes([0.2, 0.19, 0.62, 0.75])
    for _, r in eig.iterrows():
        up = r.rho_bin_eig > r.rho_bin_ibrate_same
        ax.plot([0, 1], [r.rho_bin_ibrate_same, r.rho_bin_eig], color=BLUE_LIGHT if up else LIGHT, lw=1.0, alpha=0.9)
    meds = [eig.rho_bin_ibrate_same.median(), eig.rho_bin_eig.median()]
    for k, med in enumerate(meds):
        ax.plot([k - 0.16, k + 0.16], [med] * 2, color=INK if k == 0 else BLUE, lw=3.5, zorder=4)
    ax.axhline(0, color=GRID, lw=1, zorder=0)
    ax.set_xticks([0, 1], ["発火率\n(バースト間)", "結合の主モード\n(バースト間)"])
    ax.get_xticklabels()[1].set_color(BLUE)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("点火確率との順位相関 ρ")
    ax.text(0.5, 1.0, f"{len(eig)} セッション・7 培養", transform=ax.transAxes, ha="center", va="top",
            fontsize=px(16), color=INK2)
    save(fig, "predictors")


def m_days():
    ses = pd.read_csv(RC / "sessions_dandi.csv")
    s = ses[~ses.excluded_empty.astype(bool)].copy()
    fig = fig_px(520, 380)
    ax = fig.add_axes([0.2, 0.19, 0.76, 0.75])
    ax.axhspan(1.0, 1.012, color="#FDEEEE", lw=0, zorder=0)
    ax.text(31.5, 1.006, "m > 1：暴走", ha="right", va="center", fontsize=px(16), color="#CE0000")
    for c, sub in s.groupby("culture"):
        sub = sub.sort_values("day")
        ax.plot(sub.day, sub.m_all, color=BLUE, lw=1.4, alpha=0.55, marker="o", ms=3)
    ax.axhline(1.0, color="#CE0000", lw=1.2, ls=(0, (4, 3)))
    ax.set_ylim(0.94, 1.012)
    ax.set_xlim(-1, 32)
    ax.set_xlabel("最初の記録からの日数")
    ax.set_ylabel("分岐比 m")
    save(fig, "m_days")


def dev3():
    d = pd.read_csv(RC / "sessions_dandi.csv")
    d = d[~d.excluded_empty.astype(bool)]
    fig = fig_px(1150, 300)
    items = [("burst_rate_per_min", "バースト頻度 (回/分)", True),
             ("frac_out_of_burst", "バースト外スパイクの割合", False),
             ("m_all", "分岐比 m", False)]
    for k, (key, yl, logy) in enumerate(items):
        ax = fig.add_axes([0.07 + k * 0.33, 0.22, 0.25, 0.72])
        for c, sub in d.groupby("culture"):
            sub = sub.sort_values("day")
            ax.plot(sub.day, sub[key] + (0.01 if logy else 0), color=BLUE, lw=1.2, alpha=0.5, marker="o", ms=2.5)
        if logy:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        if key == "m_all":
            ax.set_ylim(0.93, 1.01)
            ax.axhline(1, color="#CE0000", lw=1, ls=(0, (4, 3)))
        ax.set_xlabel("最初の記録からの日数")
        ax.set_ylabel(yl)
    save(fig, "dev3")


# ================================================================== FLY MODEL
def _burst_stats():
    bs = pd.read_csv(RF / "burst_stats.csv")
    if "fixALlit" not in set(bs.variant):
        import ast
        rows = []
        for line in open(RF / "bursts_fixALlit_log.txt"):
            line = line.strip()
            if line.startswith("{") and "burst_rate_per_min" in line:
                d = ast.literal_eval(line)
                rows.append(dict(variant=d["variant"], gain=d["gain"], seed=d["seed"], bursts_per_min=d["burst_rate_per_min"]))
        bs = pd.concat([bs, pd.DataFrame(rows)], ignore_index=True)
    return bs


def fly_gain():
    bs = _burst_stats()
    fig = fig_px(560, 420)
    ax = fig.add_axes([0.17, 0.18, 0.6, 0.76])
    ends = {}
    for v in VARS:
        s = bs[bs.variant == v].groupby("gain").bursts_per_min.mean()
        ax.plot(s.index, s.values, color=VAR_COL[v], marker="o", ms=6, lw=2.5)
        ends[v] = s.values[-1]
        print(v, s.round(1).to_dict())
    order = sorted(ends, key=ends.get)
    ys = {}
    last = -1e9
    for v in order:
        y = max(ends[v], last + 26)
        ys[v] = y
        last = y
    for v in VARS:
        ax.text(2.56, ys[v], VAR_AB[v], color=VAR_COL[v], va="center", fontsize=px(17), fontweight="bold")
    ax.axhline(50, color=GRID, lw=1, ls=(0, (2, 2)), zorder=0)
    ax.text(0.93, 54, "50 回/分", fontsize=px(15), color=INK2, va="bottom")
    ax.axvline(1.0, color=NEUTRAL, lw=1.2, ls=(0, (4, 3)))
    ax.text(1.04, 150, "行動で較正\nしたモデル\n(g = 1)", fontsize=px(16), color=INK2, va="center", linespacing=1.25)
    ax.set_xlim(0.9, 2.55)
    ax.set_xticks([1, 1.5, 2, 2.5], ["1", "1.5", "2", "2.5"])
    ax.set_ylim(-5, 245)
    ax.set_xlabel("シナプスのゲイン g（較正値に対する倍率）")
    ax.set_ylabel("自発バースト (回/分)")
    save(fig, "fly_gain")


def brain_ignition():
    m = fly_meta()
    ig = pd.read_csv(RF / "ignition_fixAL_g2.50.csv")
    xs, ys, grp = m["x"], m["y"], m["names"][m["codes"]]
    e = ig.set_index("group")
    per = (e.excess_spikes_per_burst / e.n).reindex(grp).to_numpy()
    pv = e.p.reindex(grp).to_numpy()
    sig = (pv < 1e-6) & (per > 0.02) & np.isfinite(per)
    fig = fig_px(580, 350)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.scatter(xs[~sig], ys[~sig], s=0.05, c="#DDDDDD", lw=0, rasterized=True)
    o = np.argsort(per[sig])
    ax.scatter(xs[sig][o], ys[sig][o], s=5, c=np.log10(per[sig][o]), cmap=HEAT, vmin=-1.7, vmax=0, lw=0,
               rasterized=True)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    j = np.flatnonzero(np.isin(grp, ["lLN1_bc|L", "lLN1_bc|R"]))
    mid = np.nanmedian(xs)
    for side_sel, lab in [(xs[j] < mid, "lLN1_bc"), (xs[j] >= mid, "lLN1_bc")]:
        jj = j[side_sel]
        if len(jj):
            ax.annotate(lab, (np.median(xs[jj]), np.median(ys[jj])), xytext=(0, 80), textcoords="offset points",
                        ha="center", fontsize=px(17), fontweight="bold", color=HUB,
                        arrowprops=dict(arrowstyle="-", lw=1.2, color=HUB))
    save(fig, "brain_ignition")


def title_brain():
    m = fly_meta()
    xs, ys, grp = m["x"], m["y"], m["names"][m["codes"]]
    eln = m["eln"]
    lln = np.isin(grp, ["lLN1_bc|L", "lLN1_bc|R"])
    fig = fig_px(520, 330)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.scatter(xs, ys, s=0.05, c="#C8C8C8", lw=0, rasterized=True)
    ax.scatter(xs[eln & ~lln], ys[eln & ~lln], s=4, c="#FF8D44", lw=0, rasterized=True)
    ax.scatter(xs[lln], ys[lln], s=9, c=HUB, lw=0, rasterized=True)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "title_brain")


def crossing():
    ig = pd.read_csv(RF / "ignition_fixAL_g2.50.csv")
    cand = ig[(ig.frac_bursts_crossed >= 0.5)].sort_values("cross_ms_median").head(10).iloc[::-1]
    fig = fig_px(520, 420)
    ax = fig.add_axes([0.3, 0.17, 0.66, 0.79])
    yy = np.arange(len(cand))
    hub = cand.group.str.startswith(("lLN1_bc", "lLN2X"))
    ax.barh(yy, cand.cross_ms_median, color=np.where(hub, HUB, LIGHT), height=0.62)
    ax.axvline(0, color=INK, lw=1.2)
    lab = [g.replace("|L", "（左）").replace("|R", "（右）") for g in cand.group]
    ax.set_yticks(yy, lab)
    for t, h in zip(ax.get_yticklabels(), hub):
        if h:
            t.set_color(HUB)
            t.set_fontweight("bold")
    for y, v, h in zip(yy, cand.cross_ms_median, hub):
        if h:
            ax.text(v + 1, y, f"{abs(v):.0f} ms 前", ha="left", va="center", fontsize=px(15), color="white",
                    fontweight="bold")
    ax.set_xlim(-47, 8)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("脳全体のバースト開始に対する時刻 (ms)")
    save(fig, "crossing")


def perron_share():
    cls = classes()
    vs = VARS
    order = [("ALLN_chol", "触角葉・アセチルコリン性局所", HUB), ("ALLN_other", "触角葉・ほかの局所", HUB_LIGHT),
             ("ALPN", "触角葉・投射", "#FF8D44"), ("KC", "キノコ体・ケニヨン細胞", BLUE_LIGHT), ("other", "その他", FAINT)]
    shares = {}
    for v in vs:
        d = pd.read_parquet(RF / f"structure_neurons_{v}.parquet")
        p = d.perron.to_numpy()
        shares[v] = [p[cls == k].sum() / p.sum() for k, _, _ in order]
    print({v: [round(x, 3) for x in s] for v, s in shares.items()})
    fig = fig_px(640, 380)
    ax = fig.add_axes([0.29, 0.16, 0.68, 0.64])
    for j, v in enumerate(vs[::-1]):
        left = 0
        for k, (key, lab, col) in enumerate(order):
            w = shares[v][k]
            ax.barh(j, w, left=left, color=col, height=0.62, edgecolor="white", lw=1)
            if key == "ALLN_chol":
                ax.text(left + w / 2, j, f"{w * 100:.0f}%", ha="center", va="center", color="white",
                        fontsize=px(16), fontweight="bold")
            left += w
    short = {"fixAL": "最小補正 MC", "fixALlit": "文献整合 LC", "typevote": "多数決 CV", "nomod": "モノアミンなし NM"}
    ax.set_yticks(range(len(vs)), [short[v] for v in vs[::-1]])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}"))
    ax.set_xlabel("ペロン固有ベクトルの重みの割合 (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in order]
    fig.legend(handles, [l for _, l, _ in order], loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=3,
               fontsize=px(15), handlelength=1.0, columnspacing=1.0, handletextpad=0.4)
    save(fig, "perron_share")


def _silencing_rel():
    sc = pd.read_csv(RF / "silencing_confirm.csv")
    sc["set"] = sc.silenced.str.replace(r"_#\d", "", regex=True)
    intact = sc[sc.set == "intact"].groupby(["variant", "gain"]).bursts_per_min.mean()
    sc["rel"] = sc.apply(lambda r: r.bursts_per_min / intact.loc[(r.variant, r.gain)], axis=1)
    return sc


def silencing():
    mt = pd.read_csv(RF / "silencing_matched.csv")
    mt["set"] = mt.silenced.str.replace(r"_#\d", "", regex=True)
    it = mt[mt.set == "intact"].groupby(["variant", "gain"]).bursts_per_min.mean()
    mt["rel"] = [b / it[(v, g)] for b, v, g in zip(mt.bursts_per_min, mt.variant, mt.gain)]
    sc = _silencing_rel()
    rows = [("AL_cholinergic_LNs", "アセチルコリン性\n局所ニューロン 111 個", mt),
            ("degree_matched_111", "出力シナプス数を\nそろえた 111 個", mt),
            ("high_drive_111", "駆動の最も大きい\n111 個（局所以外）", mt),
            ("lLN1_bc", "lLN1_bc 30 個", mt),
            ("rand_other", "ランダム・他の局所\n111 個（3 仮定）", sc)]
    fig = fig_px(660, 420)
    ax = fig.add_axes([0.33, 0.17, 0.63, 0.8])
    rng = np.random.default_rng(3)
    allv = {}
    for k, (key, lab, src) in enumerate(rows):
        y = len(rows) - 1 - k
        for j, v in enumerate(VARS):
            if key == "rand_other":
                s_ = src[src.set.isin(["random_111", "other_ALLN_111"]) & (src.variant == v)]
            else:
                s_ = src[(src.set == key) & (src.variant == v)]
            if not len(s_):
                continue
            vals = s_.groupby(["silenced", "gain"]).rel.mean().to_numpy()
            allv.setdefault(key, []).extend(vals)
            many = len(vals) > 2
            ax.scatter(vals, y + (j - 1.5) * 0.15 + rng.uniform(-0.03, 0.03, len(vals)), s=24 if many else 44,
                       color=VAR_COL[v], lw=0, alpha=0.65 if many else 0.95, zorder=3)
    for key, vals in allv.items():
        print(key, f"{min(vals):.2f}-{max(vals):.2f} mean {np.mean(vals):.2f}")
    ax.axvline(1, color=INK, lw=1.2)
    ax.set_xscale("log")
    ax.set_xticks([0.03, 0.1, 0.3, 1, 2], ["0.03", "0.1", "0.3", "1", "2"])
    ax.minorticks_off()
    ax.set_xlim(0.02, 2.4)
    ax.set_yticks(range(len(rows)), [l for _, l, _ in rows[::-1]], fontsize=px(16))
    for t, (key, _, _) in zip(ax.get_yticklabels(), rows[::-1]):
        if key == "AL_cholinergic_LNs":
            t.set_color(HUB)
            t.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("自発バーストの頻度（無処置 = 1、対数）")
    save(fig, "silencing")


def lln_rates():
    er = pd.read_csv(RF / "eln_rates.csv")
    fig = fig_px(500, 390)
    ax = fig.add_axes([0.18, 0.19, 0.78, 0.76])
    for v in VARS:
        s = er[er.variant == v]
        ax.plot(s.gain, s.lLN1_bc, color=VAR_COL[v], marker="o", ms=6, lw=2.5, label=VAR_AB[v])
    ax.plot(er[er.variant == "fixAL"].gain, er[er.variant == "fixAL"].brain_median, color=NEUTRAL, lw=2,
            ls=(0, (2, 2)))
    ax.text(2.5, 0.27, "脳全体の中央値", color=INK2, va="bottom", ha="right", fontsize=px(16))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xticks([1, 1.5, 2, 2.5], ["1", "1.5", "2", "2.5"])
    ax.set_xlim(0.9, 2.6)
    ax.set_ylim(0.1, 300)
    ax.legend(loc="center right", fontsize=px(16), ncol=2, handlelength=1.2, columnspacing=0.8,
              bbox_to_anchor=(1.0, 0.42))
    ax.set_xlabel("シナプスのゲイン g")
    ax.set_ylabel("lLN1_bc の自発発火率 (Hz)")
    save(fig, "lln_rates")


def crossing_base():
    cn = pd.read_csv(RF / "crossing_null.csv")
    b = cn[cn.file == "spikes_fixAL_g2.50"].set_index("group")
    fig = fig_px(470, 380)
    ax = fig.add_axes([0.18, 0.2, 0.78, 0.66])
    for k, (g, lab) in enumerate([("lLN1_bc|L", "左"), ("lLN1_bc|R", "右")]):
        a, r = b.loc[g, "frac_before_burst"] * 100, b.loc[g, "frac_before_random"] * 100
        ax.bar(k - 0.19, a, width=0.36, color=HUB)
        ax.bar(k + 0.19, r, width=0.36, color=LIGHT)
        ax.text(k - 0.19, a + 1.5, f"{a:.0f}%", ha="center", va="bottom", fontsize=px(17), fontweight="bold", color=HUB)
        ax.text(k + 0.19, r + 1.5, f"{r:.0f}%", ha="center", va="bottom", fontsize=px(17), color=INK2)
    ax.set_xticks([0, 1], ["lLN1_bc（左）", "lLN1_bc（右）"])
    ax.set_ylim(0, 85)
    ax.set_ylabel("閾値を越えた割合 (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in [HUB, LIGHT]]
    fig.legend(handles, ["バースト開始前", "ランダムな時刻"], loc="upper center", ncol=2, bbox_to_anchor=(0.57, 1.0),
               fontsize=px(16), handlelength=1, handletextpad=0.4)
    save(fig, "crossing_base")


def crossing_null_all():
    cn = pd.read_csv(RF / "crossing_null.csv")
    tags = [("spikes_fixAL_g2.50", "基準"), ("spikes_fixAL_g2.50-h0.15", "入力\n0.5 倍"), ("spikes_fixAL_g2.50-h0.6", "入力\n2 倍"),
            ("spikes_fixAL_g2.50-U0.1tau1000", "抑圧を\n強く"), ("spikes_fixAL_g2.50-seed6", "シード\n2"),
            ("spikes_fixAL_g2.50-seed7", "シード\n3")]
    fig = fig_px(640, 400)
    ax = fig.add_axes([0.13, 0.22, 0.85, 0.66])
    for k, (t, lab) in enumerate(tags):
        b = cn[cn.file == t].set_index("group")
        for j, (g, c) in enumerate([("lLN1_bc|L", MC), ("lLN1_bc|R", CV), ("lLN2X12|L", HUB)]):
            x = k + (j - 1) * 0.25
            a, r, pv = b.loc[g, "frac_before_burst"] * 100, b.loc[g, "frac_before_random"] * 100, b.loc[g, "p_fisher"]
            ax.bar(x, a, width=0.22, color=c, alpha=1.0 if pv < 0.05 else 0.35)
            ax.plot([x - 0.11, x + 0.11], [r, r], color=INK, lw=2.2)
        ax.text(k, 80, f"{int(b.n_bursts.iloc[0])}", ha="center", fontsize=px(15), color=INK2)
    ax.set_xticks(range(len(tags)), [l for _, l in tags], fontsize=px(15))
    ax.set_xlim(-0.6, 5.5)
    ax.set_ylim(0, 88)
    ax.set_ylabel("開始前に閾値を越えた割合 (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in [MC, CV, HUB]] + [plt.Line2D([], [], color=INK, lw=2.2)]
    fig.legend(handles, ["lLN1_bc 左", "lLN1_bc 右", "lLN2X12 左", "ランダムな時刻"], loc="upper center", ncol=4,
               bbox_to_anchor=(0.55, 1.0), fontsize=px(15), handlelength=1, handletextpad=0.4, columnspacing=0.9)
    save(fig, "crossing_null_all")


def two_modes():
    t = pd.read_csv(RF / "perron_two_modes.csv")
    fig = fig_px(620, 400)
    ax = fig.add_axes([0.3, 0.16, 0.66, 0.72])
    rows = []
    for v in VARS:
        d = t[(t.variant == v) & (t["mode"] <= 2)].sort_values("mode")
        lam1 = d.eigenvalue.iloc[0]
        for _, r in d.iterrows():
            rows.append((f"{VAR_AB[v]}　第 {int(r['mode'])} モード（{r.eigenvalue / lam1:.2f}）", r.left, r.right))
    for i, (lab, L, R) in enumerate(rows[::-1]):
        ax.barh(i, L * 100, color=MC, height=0.66)
        ax.barh(i, R * 100, left=L * 100, color=CV, height=0.66)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], fontsize=px(15))
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 100)
    ax.set_xlabel("モードの重みの割合 (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in [MC, CV]]
    fig.legend(handles, ["左半球", "右半球"], loc="upper center", ncol=2, bbox_to_anchor=(0.63, 1.0),
               fontsize=px(16), handlelength=1, handletextpad=0.4)
    save(fig, "two_modes")


def core_gain():
    lad = pd.read_csv(RF / "ladder.csv")
    fig = fig_px(560, 400)
    ax = fig.add_axes([0.22, 0.19, 0.74, 0.76])
    labs = {"shiu": ("公開の符号", NEUTRAL), "fixAL": ("MC", MC), "fixALlit": ("LC", MCL), "typevote": ("CV", CV),
            "nomod": ("NM", NM)}
    for v, (lab, c) in labs.items():
        if v not in set(lad.variant):
            continue
        s = lad[lad.variant == v].groupby("gain").n_core.mean()
        ax.plot(s.index, s.values + 1, color=c, marker="o", ms=5, lw=2.2, label=lab)
    ax.legend(loc="lower right", fontsize=px(16), handlelength=1.4)
    ax.axvline(1, color=NEUTRAL, lw=1, ls=(0, (4, 3)))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlim(0.45, 2.05)
    ax.set_xlabel("シナプスのゲイン g（短期抑圧なし）")
    ax.set_ylabel("自力で発火し続ける\nニューロン数 + 1")
    save(fig, "core_gain")


def robust_cross():
    tags = [("fixAL_g2.50", "基準"), ("fixAL_g2.50-h0.15", "入力\n0.5 倍"), ("fixAL_g2.50-h0.6", "入力\n2 倍"),
            ("fixAL_g2.50-U0.1tau1000", "抑圧を\n強く"), ("fixAL_g2.50-seed6", "シード\n2"), ("fixAL_g2.50-seed7", "シード\n3")]
    fig = fig_px(560, 400)
    ax = fig.add_axes([0.17, 0.24, 0.8, 0.66])
    for k, (t, lab) in enumerate(tags):
        ig = pd.read_csv(RF / f"ignition_{t}.csv").set_index("group")
        for side, c, dx in [("L", MC, -0.14), ("R", CV, 0.14)]:
            g = f"lLN1_bc|{side}"
            if g in ig.index and np.isfinite(ig.loc[g, "cross_ms_median"]):
                ax.scatter(k + dx, ig.loc[g, "cross_ms_median"], s=30 + 160 * ig.loc[g, "frac_bursts_crossed"],
                           color=c, lw=0, zorder=3)
        ax.text(k, 13, f"{int(ig.n_bursts.iloc[0])}", ha="center", fontsize=px(16), color=INK2)
    ax.text(-0.6, 13, "バースト数", ha="right", fontsize=px(15), color=INK2)
    ax.axhline(0, color=INK, lw=1.2)
    ax.set_xticks(range(len(tags)), [l for _, l in tags], fontsize=px(16))
    ax.set_xlim(-0.6, 5.5)
    ax.set_ylim(-52, 18)
    ax.set_ylabel("閾値を越えた時刻 (ms)\n（バースト開始 = 0）")
    ax.scatter([], [], s=80, color=MC, label="左")
    ax.scatter([], [], s=80, color=CV, label="右")
    ax.legend(loc="lower right", ncol=2, fontsize=px(16), handletextpad=0.2)
    save(fig, "robust_cross")


def eln_pn():
    c = pd.read_csv(RF / "eln_pn_control.csv")
    fig = fig_px(470, 360)
    ax = fig.add_axes([0.18, 0.18, 0.78, 0.66])
    for j, v in enumerate(["fixAL", "nomod"]):
        s = c[c.variant == v].groupby("test").value.mean()
        for k, (nm, col) in enumerate([("intact", INK), ("lLN1_bc", HUB), ("random30", LIGHT)]):
            val = s.get(f"bursts_per_min_{nm}", np.nan)
            ax.bar(j + (k - 1) * 0.26, val, width=0.24, color=col)
            ax.text(j + (k - 1) * 0.26, val + 2, f"{val:.1f}", ha="center", va="bottom", fontsize=px(15))
    ax.set_xticks([0, 1], ["最小補正 (MC)", "モノアミンなし (NM)"])
    ax.set_ylabel("バースト (回/分)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in [INK, HUB, LIGHT]]
    fig.legend(handles, ["無処置", "lLN1_bc を黙らせる", "ランダム 30 個"], loc="upper center", ncol=3,
               bbox_to_anchor=(0.55, 1.0), fontsize=px(15), handlelength=1, handletextpad=0.4, columnspacing=0.8)
    save(fig, "eln_pn")


def predict_compare():
    eig = pd.read_csv(RC / "c3_eigen_dandi.csv").dropna(subset=["rho_bin_eig", "rho_bin_ibrate_same"])
    fe = pd.read_csv(RF / "functional_eig_summary.csv")
    fe = fe[fe.tag.str.startswith("fixAL_g2.50")]
    items = [("培養：発火率", eig.rho_bin_ibrate_same, LIGHT, "c"),
             ("培養：活動の主モード", eig.rho_bin_eig, BLUE, "c"),
             ("ハエ：発火率", fe.rho_rate_ib, LIGHT, "f"),
             ("ハエ：活動の主モード", fe.rho_eig, BLUE_LIGHT, "f"),
             ("ハエ：配線の主モード", fe.rho_perron_per_neuron, HUB, "f")]
    fig = fig_px(600, 380)
    ax = fig.add_axes([0.36, 0.19, 0.6, 0.78])
    for k, (lab, vals, col, kind) in enumerate(items):
        y = len(items) - 1 - k
        if kind == "c":
            q1, q2, q3 = np.percentile(vals, [25, 50, 75])
            ax.barh(y, q2, color=col, height=0.6)
            ax.plot([q1, q3], [y, y], color=INK, lw=1.5)
        else:
            ax.barh(y, np.median(vals), color=col, height=0.6, alpha=0.75)
            ax.scatter(vals, np.full(len(vals), y), s=22, color=INK, lw=0, zorder=3)
    ax.set_yticks(range(len(items))[::-1], [i[0] for i in items])
    ax.tick_params(axis="y", length=0)
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("点火との順位相関 ρ")
    save(fig, "predict_compare")


def perron_lr():
    fig = fig_px(420, 340)
    ax = fig.add_axes([0.2, 0.18, 0.76, 0.7])
    for j, v in enumerate(["fixAL", "typevote", "nomod"]):
        dd = pd.read_parquet(RF / f"structure_neurons_{v}.parquet")
        L = dd.perron[dd.group.str.endswith("|L")].sum() / dd.perron.sum()
        R = dd.perron[dd.group.str.endswith("|R")].sum() / dd.perron.sum()
        ax.bar(j - 0.18, L * 100, width=0.34, color=MC)
        ax.bar(j + 0.18, R * 100, width=0.34, color=CV)
        print(v, round(L, 3), round(R, 3))
    ax.set_xticks(range(3), ["MC", "CV", "NM"])
    ax.set_ylabel("ペロンモードの重み (%)")
    ax.set_ylim(0, 100)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in [MC, CV]]
    fig.legend(handles, ["左半球", "右半球"], loc="upper center", ncol=2, bbox_to_anchor=(0.58, 1.0),
               fontsize=px(16), handlelength=1, handletextpad=0.4)
    save(fig, "perron_lr")


def common_scale():
    ses = pd.read_csv(RC / "sessions_dandi.csv")
    ses = ses[~ses.excluded_empty.astype(bool)]
    vm = pd.read_csv(RF / "virtual_mea.csv")
    fig = fig_px(540, 400)
    ax = fig.add_axes([0.18, 0.19, 0.78, 0.76])
    ax.scatter(ses.frac_out_of_burst, ses.m_all, s=22, color=BLUE_LIGHT, lw=0, label="培養（HD-MEA の各セッション）")
    s = vm[vm.variant == "fixAL"].groupby("gain").mean(numeric_only=True)
    ax.plot(s.frac_spikes_outside_bursts, s.m, color=HUB, marker="D", ms=7, lw=2, label="ハエのモデル（仮想 MEA、MC）")
    for gg, r in s.iterrows():
        if gg in (1.0, 2.5):
            ax.annotate(f"g = {gg:g}", (r.frac_spikes_outside_bursts, r.m), xytext=(-62, -4) if gg == 1.0 else (-20, -24),
                        textcoords="offset points", fontsize=px(16), color=HUB)
    ax.set_xlabel("バースト外のスパイクの割合")
    ax.set_ylabel("分岐比 m")
    ax.set_ylim(0.94, 1.01)
    ax.legend(loc="lower left", fontsize=px(15))
    save(fig, "common_scale")


# ================================================================== REAL FLIES
def turner_rank():
    s = pd.read_csv(RF / "turner_region_summary.csv").sort_values("ac1", ascending=False).reset_index(drop=True)
    fig = fig_px(640, 380)
    ax = fig.add_axes([0.13, 0.17, 0.85, 0.78])
    al = s.region.str.startswith("AL_")
    ax.bar(np.arange(len(s)), s.ac1, color=np.where(al, HUB, LIGHT), width=0.8)
    for i, r in s[al].iterrows():
        lab = "右の触角葉（1 位）" if r.region == "AL_R" else "左の触角葉（4 位）"
        ty = 1.06 if r.region == "AL_R" else 0.97
        ax.annotate(lab, (i, r.ac1 + 0.01), xytext=(14, ty), textcoords="data",
                    fontsize=px(17), color=HUB, fontweight="bold", va="center",
                    arrowprops=dict(arrowstyle="-", lw=1.2, color=HUB, shrinkA=2, shrinkB=0))
    ax.set_xlim(-1, len(s))
    ax.set_xticks([0, 19, 39, 59, 73], ["1", "20", "40", "60", "74"])
    ax.set_xlabel("脳領域（ゆらぎの遅い順）")
    ax.set_ylabel("ラグ 1 の自己相関")
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    save(fig, "turner_rank")


def turner_perfly():
    x = pd.read_csv(RF / "turner_AL_vs_others.csv")
    fig = fig_px(540, 380)
    ax = fig.add_axes([0.2, 0.15, 0.76, 0.72])
    items = [("ac1_AL", "ac1_oth", "自己相関"), ("sd_AL", "sd_oth", "ゆらぎの\n大きさ"), ("rg_AL", "rg_oth", "全体との\n相関")]
    rng = np.random.default_rng(1)
    for k, (a, b, lab) in enumerate(items):
        rr = np.log2(x[a] / x[b])
        up = rr > 0
        ax.scatter(k + rng.uniform(-0.13, 0.13, len(rr)), rr, s=34, color=np.where(up, HUB, NEUTRAL), lw=0,
                   alpha=0.85, zorder=3)
        ax.plot([k - 0.25, k + 0.25], [rr.median()] * 2, color=INK, lw=2.5, zorder=4)
        ax.text(k, 1.75, f"{up.sum()}/{len(rr)} 匹", ha="center", fontsize=px(17), fontweight="bold",
                color=HUB if up.sum() > len(rr) / 2 else INK2)
    ax.axhline(0, color=INK, lw=1)
    ax.set_ylim(-1.2, 2.0)
    ax.set_yticks([-1, 0, 1], ["0.5", "1", "2"])
    ax.set_xticks(range(3), [i[2] for i in items])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("触角葉 ÷ 他の領域（倍）")
    save(fig, "turner_perfly")


def sleep():
    sr = pd.read_csv(RF / "sleep_recruitment_norm.csv")
    t = sr.groupby(["circuit", "variant"]).pct.median().unstack()
    ja = {"l-LNv clock (wake)": "時計 l-LNv（覚醒）", "dFB FB6/FB7 (sleep-promoting)": "背側扇状体 FB6/7（睡眠）",
          "PPL1 DAN (wake)": "PPL1 ドーパミン（覚醒）", "MBON": "キノコ体出力 MBON",
          "Kenyon cells": "ケニヨン細胞", "ExR1 helicon": "ヘリコン細胞 ExR1", "ER5 (sleep drive)": "ER5（睡眠圧）",
          "DN1a/DN1p clock": "時計 DN1"}
    order = [c for c in t.index if c in ja]
    fig = fig_px(600, 430)
    ax = fig.add_axes([0.4, 0.15, 0.56, 0.72])
    yy = np.arange(len(order))
    vs = [v for v in VARS if v in t.columns]
    for j, v in enumerate(vs):
        ax.scatter(t.loc[order, v], yy + (j - (len(vs) - 1) / 2) * 0.17, s=44, color=VAR_COL[v], lw=0, label=VAR_AB[v], zorder=3)
    ax.axvline(0.5, color=NEUTRAL, lw=1.2, ls=(0, (4, 3)))
    ax.set_yticks(yy, [ja[o] for o in order], fontsize=px(16))
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("ランダムな細胞タイプ中のパーセンタイル")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4, fontsize=px(16), handletextpad=0.2)
    save(fig, "sleep")


ALL = ["crossing_base", "crossing_null_all", "two_modes", "culture_raster", "ign_maps", "ign_stability", "predictors", "m_days", "dev3", "fly_gain", "crossing",
       "lln_rates", "core_gain", "eln_pn", "predict_compare", "common_scale", "turner_rank",
       "turner_perfly", "sleep", "silencing", "perron_share", "brain_ignition", "title_brain"]

if __name__ == "__main__":
    for name in (sys.argv[1:] or ALL):
        globals()[name]()
