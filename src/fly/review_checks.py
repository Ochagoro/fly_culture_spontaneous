"""Checks added after internal review.

1. False-alarm null for ignition crossings: the same crossing test applied at
   random times far from bursts (>= 300 ms from any burst).
2. Left/right structure of the two leading modes of the excitatory wiring.
3. Real-fly antennal-lobe result controlled for signal brightness (SNR proxy):
   per fly, lag-1 autocorrelation regressed on log mean raw F across regions.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from spont import load_connectome  # noqa: E402
from variants import load_annotations, make_variant  # noqa: E402
from bursts import groups_of  # noqa: E402
from ignition import onsets_exact  # noqa: E402

ROOT = HERE.parents[1]
RF = ROOT / "results" / "fly"


def crossing_null(con, ann, codes, names):
    G = len(names)
    targets = ["lLN1_bc|L", "lLN1_bc|R", "lLN2X12|L"]
    tid = [int(np.flatnonzero(names == t)[0]) for t in targets]
    rows = []
    rng = np.random.default_rng(0)
    for f in sorted((RF / "video").glob("spikes_fixAL*_g2.50*.npz")) + sorted((RF / "video").glob("spikes_typevote_g2.00.npz")) + sorted((RF / "video").glob("spikes_nomod_g2.50.npz")):
        d = np.load(f)
        t_ms = d["t"].astype(np.int64) // 10
        n = d["n"].astype(np.int64)
        T = len(d["pop"])
        on = onsets_exact(d["pop"], d["on"], float(np.median(d["pop"])))
        g = codes[n]
        sel = np.isin(g, tid)
        M = np.zeros((len(tid), T), np.float32)
        lut = {t: k for k, t in enumerate(tid)}
        np.add.at(M, (np.vectorize(lut.get)(g[sel]), t_ms[sel]), 1)
        far = np.ones(T, bool)
        for a, b in zip(d["on"], d["off"]):
            far[max(0, a - 300): b + 300] = False
        far[:400] = False
        far[-100:] = False
        lam = M[:, far].mean(1)
        W = 5
        cs = np.cumsum(np.c_[np.zeros(len(tid)), M], 1)
        thr = np.maximum(lam * W + 5 * np.sqrt(lam * W), 3)

        def crossed_before(a):
            lo, hi = max(W, a - 50), a
            win = cs[:, lo:hi] - cs[:, lo - W: hi - W]
            return (win > thr[:, None]).any(1)
        rb = np.array([crossed_before(a) for a in on if a > 60])
        cand = np.flatnonzero(far)
        rt = np.array([crossed_before(a) for a in rng.choice(cand, 500, replace=False)])
        for k, tname in enumerate(targets):
            rows.append(dict(file=f.stem, group=tname, n_bursts=len(rb), frac_before_burst=rb[:, k].mean(),
                             frac_before_random=rt[:, k].mean(),
                             p_fisher=stats.fisher_exact([[rb[:, k].sum(), len(rb) - rb[:, k].sum()],
                                                          [rt[:, k].sum(), len(rt) - rt[:, k].sum()]],
                                                         alternative="greater")[1]))
    df = pd.DataFrame(rows)
    df.to_csv(RF / "crossing_null.csv", index=False)
    print(df.round(3).to_string())


def two_modes(con0, ann):
    rows = []
    lln = (ann["cell_type"] == "lLN1_bc").to_numpy()
    side = ann["side"].to_numpy()
    for v in ["fixAL", "fixALlit", "typevote", "nomod"]:
        con, _ = make_variant(v, con0, ann)
        w = con.count.astype(np.float64) * np.clip(con.sign[con.pre], 0, None)
        Wp = sp.csr_matrix((w, (con.post, con.pre)), shape=(con.n, con.n))
        vals, vecs = eigs(Wp, k=3, which="LR", tol=1e-6, maxiter=10000)
        o = np.argsort(-vals.real)
        for r, i in enumerate(o):
            v_ = np.abs(vecs[:, i]); v_ /= v_.sum()
            rows.append(dict(variant=v, mode=r + 1, eigenvalue=vals[i].real, left=v_[side == "left"].sum(),
                             right=v_[side == "right"].sum(), lLN1_bc_L=v_[lln & (side == "left")].sum(),
                             lLN1_bc_R=v_[lln & (side == "right")].sum(),
                             AL_LN=v_[(ann.cell_class == "ALLN").to_numpy()].sum(),
                             KC=v_[(ann.cell_class == "Kenyon_Cell").to_numpy()].sum()))
    df = pd.DataFrame(rows)
    df.to_csv(RF / "perron_two_modes.csv", index=False)
    print(df.round(3).to_string())


def turner_snr():
    from turner_load import load
    from turner_metrics import TD, dff, region_names
    names = region_names()
    rows = []
    for fn in sorted((TD / "ito_responses").glob("ito_*.pkl")):
        df = load(fn)
        df = df.loc[[i for i in df.index if isinstance(names.get(i), str)]]
        F = df.to_numpy(np.float64)
        X = dff(F)
        ac1 = np.array([np.corrcoef(x[:-1], x[1:])[0, 1] for x in X])
        lf = np.log(F.mean(1))
        b = np.polyfit(lf, ac1, 1)
        res = ac1 - np.polyval(b, lf)
        reg = [names[i] for i in df.index]
        rk = pd.Series(res, index=reg).rank(ascending=False)
        rows.append(dict(fly=fn.stem, slope=b[0], r_ac1_logF=np.corrcoef(lf, ac1)[0, 1],
                         AL_R_resid_rank=rk.get("AL_R"), AL_L_resid_rank=rk.get("AL_L"), n=len(reg),
                         AL_resid=np.mean([res[reg.index("AL_R")], res[reg.index("AL_L")]]),
                         others_resid_median=np.median([r for r, n_ in zip(res, reg) if not n_.startswith("AL_")])))
    d = pd.DataFrame(rows)
    d.to_csv(RF / "turner_snr_control.csv", index=False)
    print(d.round(3).to_string())
    print("AL residual > others in", (d.AL_resid > d.others_resid_median).sum(), "of", len(d),
          "Wilcoxon p", stats.wilcoxon(d.AL_resid, d.others_resid_median).pvalue,
          "median r(ac1, logF)", d.r_ac1_logF.median())


if __name__ == "__main__":
    con0 = load_connectome("783")
    ann = load_annotations(con0)
    codes, names = groups_of(ann)
    which = sys.argv[1:] or ["null", "modes", "snr"]
    if "null" in which:
        crossing_null(con0, ann, codes, names)
    if "modes" in which:
        two_modes(con0, ann)
    if "snr" in which:
        turner_snr()
