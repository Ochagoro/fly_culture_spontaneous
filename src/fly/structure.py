"""Structural predictors of ignition in the FlyWire connectome.

* Perron mode of the excitatory subgraph W+ (count of excitatory synapses
  j -> i): the pattern that grows first when excitation is scaled up and
  inhibition is ignored.
* Leading mode of the full signed matrix W (eigenvalue with the largest real
  part): the first linear instability of the signed network.
* Reciprocal excitation per neuron: sum over partners of min(w_ij, w_ji) for
  excitatory pairs (2-cycles), a local measure of recurrent excitation.
Results are aggregated per cell type x hemisphere.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from spont import load_connectome  # noqa: E402
from variants import make_variant, load_annotations  # noqa: E402
from bursts import groups_of  # noqa: E402

ROOT = HERE.parents[1]


def signed_matrix(con):
    w = con.count.astype(np.float64) * con.sign[con.pre]
    return sp.csr_matrix((w, (con.post, con.pre)), shape=(con.n, con.n))  # row = target


def run(variant: str):
    con0 = load_connectome("783")
    con, ann = make_variant(variant, con0)
    W = signed_matrix(con)
    Wp = W.maximum(0).tocsr()
    t = time.time()
    lp, vp = eigs(Wp, k=1, which="LR", tol=1e-6, maxiter=5000)
    print(variant, "Perron W+", lp, f"{time.time()-t:.0f}s", flush=True)
    t = time.time()
    lr, vr = eigs(W, k=6, which="LR", tol=1e-6, maxiter=20000)
    print(variant, "W top-LR eigenvalues", np.round(lr, 2), f"{time.time()-t:.0f}s", flush=True)
    vp = np.abs(vp[:, 0]); vp /= vp.sum()
    i0 = np.argmax(lr.real)
    v0 = np.abs(vr[:, i0]); v0 /= v0.sum()
    # reciprocal excitation
    A = Wp.tocoo()
    R = Wp.minimum(Wp.T.tocsr())
    recip = np.asarray(R.sum(1)).ravel()
    codes, names = groups_of(ann)
    df = pd.DataFrame(dict(group=names[codes], cell_class=ann["cell_class"].to_numpy(),
                           super_class=ann["super_class"].to_numpy(), perron=vp, lead_mode=v0,
                           recip=recip, exc_in=np.asarray(Wp.sum(1)).ravel()))
    out = ROOT / "results" / "fly"
    df.to_parquet(out / f"structure_neurons_{variant}.parquet")
    g = df.groupby("group").agg(n=("perron", "size"), perron=("perron", "sum"), lead_mode=("lead_mode", "sum"),
                                recip=("recip", "mean"), exc_in=("exc_in", "mean"))
    g.to_csv(out / f"structure_groups_{variant}.csv")
    pd.DataFrame(dict(re=lr.real, im=lr.imag)).to_csv(out / f"eig_LR_{variant}.csv", index=False)
    for col in ["perron", "lead_mode", "recip"]:
        print(f"  top groups by {col}:", g[col].sort_values(ascending=False).head(12).round(4).to_dict(), flush=True)


if __name__ == "__main__":
    for v in sys.argv[1:] or ["fixAL", "typevote", "nomod"]:
        run(v)
