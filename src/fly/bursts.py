"""Spontaneous network bursts in the FlyWire whole-brain model and where they ignite.

Each neuron receives weak Poisson 'spontaneous spikes' (rate h). Synapses have
short-term depression (Tsodyks-Markram, U, tau_rec), as in models of
dissociated cultures. The synaptic gain g multiplies all weights (g = 1 is the
behaviourally calibrated Shiu et al. model). For each run we record:

* whole-brain population activity (1 ms bins),
* activity of every cell type x hemisphere group (2 ms bins), from which
  burst-triggered averages give each group's pre-onset excess and
  recruitment latency (= the ignition analysis),
* spike events of a random 'virtual MEA' of 1,024 neurons with their
  positions, for the same metrics as the cultured networks.

Usage: python bursts.py <variant> [gains...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from spont import SpontSim, load_connectome  # noqa: E402
from variants import load_annotations, make_variant  # noqa: E402

ROOT = HERE.parents[1]
OUT = ROOT / "results" / "fly" / "bursts"

H = 0.3          # Hz, spontaneous drive per neuron
U, TAU = 0.05, 2000.0
T_RUN = 15000.0  # ms
GBIN = 2.0       # ms, group bins
BURST_SMOOTH = 20  # ms
PRE = 50         # ms window before onset for the pre-onset excess


def groups_of(ann: pd.DataFrame):
    side = ann["side"].fillna("na").replace({"center": "c", "left": "L", "right": "R", "na": "c"})
    typ = ann["cell_type"].fillna(ann["cell_class"]).fillna(ann["super_class"]).fillna("untyped")
    key = typ.astype(str) + "|" + side.astype(str)
    codes, uniq = pd.factorize(key)
    return codes.astype(np.int64), np.asarray(uniq)


def detect_bursts(pop: np.ndarray, base: float, smooth: int = BURST_SMOOTH):
    """Bursts = runs where smoothed population rate exceeds 1.5x baseline + 30 spikes/ms."""
    x = np.convolve(pop, np.ones(smooth) / smooth, mode="same")
    thr = 1.5 * base + 30
    above = (x > thr).astype(np.int8)
    d = np.diff(np.r_[0, above, 0])
    on, off = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    keep = (off - on) >= 10
    return on[keep], off[keep], thr


def run(variant: str, gains, seeds=(0, 1), t_run: float = T_RUN, std_U: float = U,
        std_tau: float = TAU, h: float = H, tag: str = ""):
    """All gains x seeds in one GPU batch."""
    con0 = load_connectome("783")
    con, ann = make_variant(variant, con0)
    codes, names = groups_of(ann)
    G = len(names)
    sizes = np.bincount(codes, minlength=G)
    rng = np.random.default_rng(1234)
    mea = np.sort(rng.choice(con.n, 1024, replace=False))
    mask = np.zeros(con.n, bool)
    mask[mea] = True
    sim = SpontSim(con)
    OUT.mkdir(parents=True, exist_ok=True)
    combos = [(g, s) for g in gains for s in seeds]
    out = sim.run([g for g, _ in combos], h=h, t_run=t_run, seed=hash_seed(variant, tag), bin_ms=1.0,
                  std_U=std_U, std_tau=std_tau, groups=codes, n_groups=G, record_events=mask,
                  grp_bin_ms=GBIN)
    gb = int(GBIN)
    rows = []
    for b, (g, seed) in enumerate(combos):
        pop = out["pop"][b]
        grp = out["grp"][b].astype(np.float32)
        nb = grp.shape[1]
        base = con.n * h * 1e-3
        on, off, thr = detect_bursts(pop[300:], base)
        on, off = on + 300, off + 300
        inb = np.zeros(len(pop), bool)
        for a, e in zip(on, off):
            inb[max(0, a - 200): e + 200] = True
        inb_g = inb[: nb * gb].reshape(nb, gb).any(1)
        base_g = grp[:, ~inb_g].mean(1) if (~inb_g).sum() > 50 else grp.mean(1)
        pre_ex, lat, peak = [], [], []
        for a, e in zip(on, off):
            a2, b2 = a // gb, e // gb
            p0 = max(0, a2 - PRE // gb)
            if p0 == a2 or b2 <= a2:
                continue
            pre = grp[:, p0:a2].mean(1)
            seg = grp[:, a2:b2]
            pre_ex.append(pre - base_g)
            peak.append(seg.max(1))
            half = seg >= 0.5 * seg.max(1, keepdims=True)
            first = np.where(seg.max(1) > base_g + 1e-6, half.argmax(1), -1)
            lat.append(first * GBIN)
        sel = out["ev_b"] == b
        np.savez_compressed(
            OUT / f"{variant}{tag}_g{g:.2f}_s{seed}.npz",
            pop=pop.astype(np.int32), on=on, off=off, thr=thr, names=names, sizes=sizes,
            base_g=base_g, pre_ex=np.array(pre_ex, np.float32).reshape(-1, G),
            lat=np.array(lat, np.float32).reshape(-1, G), peak=np.array(peak, np.float32).reshape(-1, G),
            rate=out["counts"][b].astype(np.float32) / (t_run * 1e-3), mea_idx=mea,
            ev_t=out["ev_t"][sel], ev_n=out["ev_n"][sel],
            gain=g, h=h, U=std_U, tau=std_tau, t_run=t_run)
        rows.append(dict(variant=variant, gain=g, seed=seed, n_bursts=len(on),
                         burst_rate_per_min=len(on) / (t_run / 60000),
                         frac_time_in_burst=float(np.sum(off - on) / len(pop)),
                         mean_rate=float(out["counts"][b].sum() / con.n / (t_run * 1e-3))))
        print(rows[-1], flush=True)
    return rows


def hash_seed(*parts) -> int:
    import zlib
    return zlib.crc32("|".join(map(str, parts)).encode()) % 100000


if __name__ == "__main__":
    variant = sys.argv[1]
    gains = [float(x) for x in sys.argv[2:]] or [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
    run(variant, gains)
