"""Causal test of the predicted ignition sites: silence them and count spontaneous bursts.

For each transmitter variant, at a gain where the intact model bursts
regularly, the outputs of a neuron set are removed and spontaneous network
bursts are counted (same drive and short-term depression as bursts.py).
Sets: the predicted ignition core (cholinergic AL local neurons), Kenyon
cells, and random size-matched controls (all neurons, and AL local neurons of
other transmitters).
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from spont import SpontSim, load_connectome  # noqa: E402
from variants import load_annotations, make_variant, transmitter_table  # noqa: E402
from bursts import detect_bursts, H, U, TAU  # noqa: E402

ROOT = HERE.parents[1]
GAINS = {"fixAL": [1.5, 1.75, 2.0, 2.25, 2.5], "typevote": [1.0, 1.25, 1.5, 1.75, 2.0],
         "nomod": [1.5, 1.75, 2.0, 2.25, 2.5]}
T = 8000.0
SEEDS = [21, 22]


def neuron_sets(ann: pd.DataFrame, rng):
    nt = transmitter_table(ann)
    alln = (ann["cell_class"] == "ALLN").to_numpy()
    eln = alln & (nt == "acetylcholine").to_numpy()
    lln1bc = (ann["cell_type"] == "lLN1_bc").to_numpy()
    kc = (ann["cell_class"] == "Kenyon_Cell").to_numpy()
    other_ln = np.flatnonzero(alln & ~eln)
    sets = {"intact": np.zeros(0, np.int64),
            "lLN1_bc": np.flatnonzero(lln1bc),
            "AL_cholinergic_LNs": np.flatnonzero(eln),
            "Kenyon_cells": np.flatnonzero(kc)}
    n1, n = len(sets["lLN1_bc"]), len(sets["AL_cholinergic_LNs"])
    for k in range(3):
        sets[f"random_{n1}_#{k}"] = rng.choice(len(ann), n1, replace=False)
        sets[f"random_{n}_#{k}"] = rng.choice(len(ann), n, replace=False)
        sets[f"other_ALLN_{n}_#{k}"] = rng.choice(other_ln, min(n, len(other_ln)), replace=False)
    sets[f"random_{len(sets['Kenyon_cells'])}"] = rng.choice(len(ann), len(sets["Kenyon_cells"]), replace=False)
    return sets


def main(variants):
    con0 = load_connectome("783")
    ann = load_annotations(con0)
    rows = []
    for v in variants:
        con, _ = make_variant(v, con0, ann)
        rng = np.random.default_rng(7)
        sets = neuron_sets(ann, rng)
        combos = [(name, g, s) for name in sets for g in GAINS[v] for s in SEEDS]
        sim = SpontSim(con)
        out = sim.run([g for _, g, _ in combos], h=H, t_run=T, seed=zlib.crc32(v.encode()) % 10000,
                      std_U=U, std_tau=TAU, silence=[sets[name] for name, _, _ in combos])
        base = con.n * H * 1e-3
        for b, (name, g, s) in enumerate(combos):
            pop = out["pop"][b]
            on, off, _ = detect_bursts(pop[300:], base)
            rows.append(dict(variant=v, gain=g, silenced=name, n_silenced=len(sets[name]), seed=s,
                             bursts_per_min=len(on) / ((T - 300) / 60000),
                             mean_rate=float(pop.sum() / con.n / (T * 1e-3))))
        print(v, "done", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "fly"
    df.to_csv(out / "silencing.csv", index=False)
    print(df.groupby(["variant", "silenced", "gain"])["bursts_per_min"].mean().round(1).unstack().to_string())


if __name__ == "__main__":
    main(sys.argv[1:] or ["fixAL", "typevote", "nomod"])
