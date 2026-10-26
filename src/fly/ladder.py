"""Which circuit becomes self-sustaining first as synaptic gain rises?

No short-term depression. All neurons get 0.5-Hz spontaneous drive for 500 ms,
then the drive stops; neurons still firing >= 10 Hz 1-2 s later form the
self-sustained core. Repeated for each transmitter variant, gain and 3 seeds.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from spont import SpontSim, load_connectome  # noqa: E402
from variants import load_annotations, make_variant  # noqa: E402

ROOT = HERE.parents[1]
GAINS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0]
SEEDS = [1, 2, 3]


def main():
    con0 = load_connectome("783")
    ann = load_annotations(con0)
    cls = ann["cell_class"].fillna(ann["super_class"]).fillna("other").to_numpy()
    typ = ann["cell_type"].fillna("").to_numpy()
    side = ann["side"].fillna("na").to_numpy()
    allidx = np.arange(con0.n)
    rows, members = [], []
    for v in ["shiu", "fixAL", "typevote", "nomod"]:
        con, _ = make_variant(v, con0, ann)
        combos = [(g, s) for g in GAINS for s in SEEDS]
        out = SpontSim(con).run([g for g, _ in combos], h=0.0, t_run=2500, seed=11,
                                stim=[[(allidx, 0, 500, 0.5)]] * len(combos), count_from_ms=1500)
        for b, (g, s) in enumerate(combos):
            core = out["counts"][b] >= 10
            r = dict(variant=v, gain=g, seed=s, n_core=int(core.sum()),
                     n_left=int((core & (side == "left")).sum()), n_right=int((core & (side == "right")).sum()),
                     n_AL_LN=int((core & (cls == "ALLN")).sum()), n_KC=int((core & (cls == "Kenyon_Cell")).sum()),
                     n_lLN1_bc=int((core & (typ == "lLN1_bc")).sum()))
            rows.append(r)
            if core.any():
                vc = pd.Series(cls[core]).value_counts()
                for c, n in vc.items():
                    members.append(dict(variant=v, gain=g, seed=s, cell_class=c, n=int(n)))
        print(v, "done", flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "fly" / "ladder.csv", index=False)
    pd.DataFrame(members).to_csv(ROOT / "results" / "fly" / "ladder_classes.csv", index=False)
    d = pd.DataFrame(rows)
    print(d.groupby(["variant", "gain"])[["n_core", "n_AL_LN", "n_lLN1_bc", "n_KC", "n_left", "n_right"]].mean().round(1).to_string())


if __name__ == "__main__":
    main()
