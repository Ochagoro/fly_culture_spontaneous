"""Confirmation of the silencing result with longer runs, more seeds and more
size-matched controls, at two gains per variant (just above burst onset and
well above it)."""
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
GAINS = {"fixAL": [2.0, 2.5], "typevote": [1.75, 2.0], "nomod": [2.25, 2.5]}
T = 20000.0
SEEDS = [31, 32, 33]
NCTRL = 5


def sets_for(ann, rng):
    nt = transmitter_table(ann)
    alln = (ann["cell_class"] == "ALLN").to_numpy()
    eln = alln & (nt == "acetylcholine").to_numpy()
    s = {"intact": np.zeros(0, np.int64), "lLN1_bc": np.flatnonzero((ann["cell_type"] == "lLN1_bc").to_numpy()),
         "AL_cholinergic_LNs": np.flatnonzero(eln)}
    other = np.flatnonzero(alln & ~eln)
    for k in range(NCTRL):
        s[f"random_30_#{k}"] = rng.choice(len(ann), len(s["lLN1_bc"]), replace=False)
        s[f"random_{eln.sum()}_#{k}"] = rng.choice(len(ann), eln.sum(), replace=False)
        s[f"other_ALLN_{eln.sum()}_#{k}"] = rng.choice(other, eln.sum(), replace=False)
    return s


def main(variants):
    con0 = load_connectome("783")
    ann = load_annotations(con0)
    rows = []
    for v in variants:
        con, _ = make_variant(v, con0, ann)
        sets = sets_for(ann, np.random.default_rng(11))
        combos = [(n, g, s) for n in sets for g in GAINS[v] for s in SEEDS]
        sim = SpontSim(con)
        out = sim.run([g for _, g, _ in combos], h=H, t_run=T, seed=zlib.crc32(("c" + v).encode()) % 10000,
                      std_U=U, std_tau=TAU, silence=[sets[n] for n, _, _ in combos])
        base = con.n * H * 1e-3
        for b, (n, g, s) in enumerate(combos):
            on, off, _ = detect_bursts(out["pop"][b][300:], base)
            rows.append(dict(variant=v, gain=g, silenced=n, n_silenced=len(sets[n]), seed=s,
                             bursts_per_min=len(on) / ((T - 300) / 60000)))
        print(v, "done", flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "results" / "fly" / "silencing_confirm.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:] or ["fixAL", "typevote", "nomod"])
