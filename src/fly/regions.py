"""Neuron-to-neuropil weights from the FlyWire v783 per-neuron synapse counts.

A neuron's weight in a neuropil is the fraction of its synapses (pre + post)
that lie in that neuropil, so a region's population signal is the
synapse-weighted sum of neuron activity, which is roughly what a pan-neuronal
calcium signal in that neuropil measures.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
FW = ROOT / "data" / "flywire"


def neuropil_matrix(ids: np.ndarray, kind: str = "both"):
    """Return (sparse N x R matrix of synapse fractions, list of region names)."""
    parts = []
    if kind in ("pre", "both"):
        d = pd.read_feather(FW / "per_neuron_neuropil_count_pre_783.feather")
        parts.append(d.rename(columns={"pre_pt_root_id": "root_id"}))
    if kind in ("post", "both"):
        d = pd.read_feather(FW / "per_neuron_neuropil_count_post_783.feather")
        parts.append(d.rename(columns={"post_pt_root_id": "root_id"}))
    d = pd.concat(parts)
    d = d[d.neuropil != "None"].groupby(["root_id", "neuropil"], as_index=False)["count"].sum()
    names = sorted(d.neuropil.unique())
    lut = pd.Series(np.arange(len(ids)), index=ids)
    d = d[d.root_id.isin(lut.index)]
    rows = lut.loc[d.root_id].to_numpy()
    cols = pd.Series(np.arange(len(names)), index=names).loc[d.neuropil].to_numpy()
    m = sp.csr_matrix((d["count"].to_numpy(np.float64), (rows, cols)), shape=(len(ids), len(names)))
    tot = np.asarray(m.sum(1)).ravel()
    tot[tot == 0] = 1
    m = sp.diags(1 / tot) @ m
    return m.tocsr(), names


def main_neuropil(ids: np.ndarray) -> pd.Series:
    m, names = neuropil_matrix(ids)
    best = np.asarray(m.argmax(1)).ravel()
    has = np.asarray(m.sum(1)).ravel() > 0
    out = np.where(has, np.array(names)[best], "none")
    return pd.Series(out, index=ids)


if __name__ == "__main__":
    # builds data/flywire/neuron_neuropil_frac.npz and neuropil_names.csv (used by analyze_bursts.py)
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from sim import load_connectome
    con = load_connectome("783")
    m, names = neuropil_matrix(con.ids)
    sp.save_npz(FW / "neuron_neuropil_frac.npz", m)
    pd.Series(names).to_csv(FW / "neuropil_names.csv", index=False)
    print(m.shape, len(names))
