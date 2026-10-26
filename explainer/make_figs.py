"""Simplified, Japanese-labelled figures for the explainer (explainer/img/*.png).

Run from the project root:  .venv/bin/python explainer/make_figs.py
Data come from results/ (the same files the paper figures use); video frames from video/build/.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RC, RF = ROOT / "results/culture", ROOT / "results/fly"
OUT = ROOT / "explainer/img"
OUT.mkdir(parents=True, exist_ok=True)
MM = 1 / 25.4

INK, INK2, MUTED, GRID = "#1a1a1a", "#4d4d4d", "#767676", "#cccccc"
KEY = "#0017c1"
HUB = "#c74700"
MC, CV, NM = "#2a78d6", "#eb6834", "#1baf7a"
HEAT = mcolors.LinearSegmentedColormap.from_list(
    "heat", ["#ffffff", "#fde0c5", "#f6a46b", "#e2572b", "#a3220f", "#4a0b05"])

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK JP", "Noto Sans JP", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.labelcolor": INK,
    "savefig.dpi": 260,
    "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print("saved", name)


def culture_raster():
    ex = np.load(RC / "example_dandi_sub-23312_ses-23312-240504.npz")
    t = ex["spike_frame_20kHz"].astype(np.float64) / 20000.0
    ch = ex["spike_channel_index"].astype(np.int64)
    x, act = ex["channel_x_um"], ex["channel_active"]
    t0, t1 = 870.0, 900.0
    m = (t >= t0) & (t < t1) & act[ch]
    order = np.argsort(np.argsort(x))
    show = np.random.default_rng(0).random(len(x)) < 0.5
    mm = m & show[ch]
    fig, ax = plt.subplots(figsize=(170 * MM, 44 * MM))
    for a, b in zip(ex["burst_start_s"], ex["burst_end_s"]):
        if b > t0 and a < t1:
            ax.axvspan(max(a, t0) - t0, min(b, t1) - t0, color="#d9e6ff", lw=0, zorder=0)
    ax.scatter(t[mm] - t0, order[ch[mm]], s=0.08, c=INK, lw=0, rasterized=True)
    ax.set_xlim(0, t1 - t0)
    ax.set_ylim(order[act].min() - 5, order.max() + 5)
    ax.set_xlabel("時間（秒）")
    ax.set_ylabel("電極\n（左から右の順）")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    save(fig, "culture_raster.png")


def culture_maps():
    maps = np.load(RC / "ignition_maps_dandi.npz")
    sel = np.flatnonzero(maps["culture"] == "23312")
    days = maps["day"][sel]
    pick = [d for d in [9, 11, 13, 16, 18, 20] if d in set(days)]
    fig, axs = plt.subplots(1, len(pick), figsize=(170 * MM, 30 * MM))
    for ax, d in zip(axs, pick):
        i = sel[np.flatnonzero(days == d)[0]]
        pm = maps["ign_pmap"][i].T.copy()
        sup = maps["support"][i].T
        pm[~sup] = np.nan
        ax.imshow(pm / np.nanmax(pm), origin="lower", cmap=HEAT, vmin=0, vmax=1,
                  extent=[0, 3.9, 0, 2.1], interpolation="nearest")
        ax.set_facecolor("#ececec")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.6)
            s.set_color(MUTED)
        ax.set_title(f"{d} 日目", fontsize=9, pad=3)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "culture_maps.png")


def gain_curve():
    d = pd.read_csv(RF / "burst_stats.csv")
    # literature-consistent minimal correction (assumption B) is logged separately
    import ast
    rows = [ast.literal_eval(l) for l in (RF / "bursts_fixALlit_log.txt").read_text().splitlines()
            if l.startswith("{'variant': 'fixALlit'")]
    lit = pd.DataFrame(rows).rename(columns={"burst_rate_per_min": "bursts_per_min"})
    d = pd.concat([d[["variant", "gain", "seed", "bursts_per_min"]], lit[["variant", "gain", "seed", "bursts_per_min"]]])
    g = d.groupby(["variant", "gain"]).bursts_per_min.mean().reset_index()
    print(g[g.gain >= 2.5])
    fig, ax = plt.subplots(figsize=(84 * MM, 64 * MM))
    names = {"fixAL": ("仮定 A", "#9a9a9a"), "fixALlit": ("仮定 B", MC), "typevote": ("仮定 C", CV),
             "nomod": ("仮定 D", NM)}
    for v, (lab, col) in names.items():
        s = g[g.variant == v]
        ax.plot(s.gain, s.bursts_per_min, marker="o", color=col, ms=4, lw=1.6, label=lab,
                ls="--" if v == "fixAL" else "-")
    ax.axvline(1.0, color=MUTED, ls="--", lw=0.9)
    ax.text(1.04, 32, "元の設定\n（ゲイン 1）", fontsize=8.5, color=INK2, va="bottom")
    ax.set_xlabel("ゲイン（元の設定 = 1）")
    ax.set_ylabel("自発バースト（回/分）")
    ax.set_xticks([1.0, 1.25, 1.5, 1.75, 2.0, 2.5])
    ax.set_xticklabels(["1", "1.25", "1.5", "1.75", "2", "2.5"])
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_ylim(-5, 245)
    save(fig, "gain_curve.png")


def silencing():
    """Matched-control silencing: bursts relative to the intact model, 4 assumptions x 2 gains."""
    d = pd.read_csv(RF / "silencing_matched.csv")
    m = d.groupby(["variant", "gain", "silenced"]).bursts_per_min.mean().reset_index()
    intact = m[m.silenced == "intact"].set_index(["variant", "gain"]).bursts_per_min
    m["ratio"] = [r.bursts_per_min / intact[(r.variant, r.gain)] for r in m.itertuples()]
    m["grp"] = m.silenced.str.replace(r"_#\d+", "", regex=True)
    rows = [
        ("AL_cholinergic_LNs", "触角葉のアセチルコリン性\n局所ニューロン 111 個", HUB),
        ("degree_matched_111", "つなぎ目の数をそろえた\n別のアクセル役 111 個", "#9a9a9a"),
        ("high_drive_111", "駆動が最も強いアクセル役\n111 個（主に投射ニューロン）", "#7a1f00"),
    ]
    fig, ax = plt.subplots(figsize=(140 * MM, 62 * MM))
    rng = np.random.default_rng(1)
    for k, (key, lab, col) in enumerate(rows):
        y = 4 - k
        v = m[m.grp == key].ratio.to_numpy()
        ax.scatter(v, y + rng.uniform(-0.13, 0.13, len(v)), s=22, color=col, zorder=3, lw=0.4, edgecolor="white")
        ax.plot([v.mean()] * 2, [y - 0.28, y + 0.28], color=INK, lw=1.6, zorder=4)
        print(key, round(v.min(), 2), round(v.max(), 2), "mean", round(v.mean(), 2))
    # preliminary random / other-LN controls (range only)
    ax.add_patch(mpl.patches.Rectangle((0.74, 0.72), 1.74 - 0.74, 0.56, color="#d9d9d9", lw=0, zorder=1))
    ax.text(1.24, 1.0, "0.74〜1.74", ha="center", va="center", fontsize=9, color=INK2)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["同じ数のランダムな細胞など\n（予備実験、範囲のみ）"] + [r[1] for r in rows][::-1], fontsize=9.5)
    ax.axvline(1.0, color=INK2, lw=0.9, ls=":")
    ax.text(1.0, 4.62, "何もしないとき", ha="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0, 1.8)
    ax.set_ylim(0.5, 4.85)
    ax.set_xlabel("黙らせたあとの自発バースト（何もしないとき = 1）")
    save(fig, "silencing.png")


def fly_regions():
    s = pd.read_csv(RF / "turner_region_summary.csv").sort_values("ac1", ascending=False).reset_index(drop=True)
    rank = {r: i + 1 for i, r in enumerate(s.region)}
    print("AL ranks", rank["AL_R"], rank["AL_L"], "n", len(s))
    fig, ax = plt.subplots(figsize=(170 * MM, 52 * MM))
    cols = [HUB if r.startswith("AL_") else "#b3b3b3" for r in s.region]
    ax.bar(np.arange(len(s)), s.ac1, color=cols, width=0.82)
    ax.set_xlim(-1, len(s))
    ax.set_xticks([])
    ax.set_ylabel("ゆらぎの遅さ\n（ラグ 1 の自己相関）")
    ax.set_xlabel(f"脳の {len(s)} 領域（ゆっくりゆらぐ順）")
    for r, lab, dy in [("AL_R", "触角葉（右）1 位", 0.0), ("AL_L", "触角葉（左）4 位", 0.0)]:
        i = rank[r] - 1
        ax.annotate(lab, xy=(i, s.ac1[i]), xytext=(i + 8 + (6 if r == "AL_L" else 0), s.ac1[i] + 0.03 - (0.1 if r == "AL_L" else 0)),
                    fontsize=9, color=HUB, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=HUB, lw=0.8))
    ax.set_ylim(0, 1.0)
    save(fig, "fly_regions.png")


def crops():
    vb = ROOT / "video/build"
    a = Image.open(vb / "check_03_chip.png").convert("RGB").crop((186, 96, 1094, 620))
    a.save(OUT / "chip_ignition.png")
    b = Image.open(vb / "check_10_slowmo2.png").convert("RGB").crop((20, 150, 900, 590))
    b.save(OUT / "fly_ignition.png")
    print("saved crops", a.size, b.size)


if __name__ == "__main__":
    culture_raster()
    culture_maps()
    gain_curve()
    silencing()
    fly_regions()
    crops()
