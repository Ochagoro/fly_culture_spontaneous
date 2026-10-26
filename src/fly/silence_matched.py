"""Silencing with controls matched for output synapses and activity (added after review).

The 111 cholinergic AL local neurons have unusually many output synapses
(lLN1_bc: ~3,700 each) and high spontaneous rates, so a fair control must be
matched on these properties, not drawn at random. Two control types, both
restricted to excitatory neurons outside the AL local-neuron class:

* high-drive: the N neurons with the largest (output synapses x spontaneous
  rate), deterministic -- the strongest possible 'generic' drivers;
* degree-matched: for every target neuron, a random excitatory neuron with the
  nearest output-synapse count (three draws).

Run for each transmitter variant at two gains, 3 seeds x 20 s.
"""
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
RF = ROOT / "results" / "fly"
GAINS = {"fixAL": [2.0, 2.5], "fixALlit": [2.0, 2.5], "typevote": [1.75, 2.0], "nomod": [2.25, 2.5]}
T = 20000.0
SEEDS = [41, 42, 43]


def matched(target, pool, outsyn, rng):
    order = pool[np.argsort(outsyn[pool])]
    vals = outsyn[order]
    used, out = set(), []
    for j in target:
        k = int(np.searchsorted(vals, outsyn[j]))
        cand = [c for c in range(max(0, k - 20), min(len(order), k + 20)) if order[c] not in used]
        c = rng.choice(cand)
        used.add(order[c])
        out.append(order[c])
    return np.array(out)


def main(variants):
    con0 = load_connectome("783")
    ann = load_annotations(con0)
    nt = transmitter_table(ann)
    alln = (ann["cell_class"] == "ALLN").to_numpy()
    eln = np.flatnonzero(alln & (nt == "acetylcholine").to_numpy())
    lln = np.flatnonzero((ann["cell_type"] == "lLN1_bc").to_numpy())
    outsyn = np.bincount(con0.pre, weights=con0.count, minlength=con0.n)
    rows = []
    for v in variants:
        con, _ = make_variant(v, con0, ann)
        rate = np.load(RF / "bursts" / f"{v}_g2.00_s0.npz", allow_pickle=True)["rate"]
        pool = np.flatnonzero((con.sign > 0) & ~alln & (outsyn > 0))
        drive = outsyn[pool] * rate[pool]
        hd = pool[np.argsort(drive)[::-1]]
        rng = np.random.default_rng(zlib.crc32(v.encode()))
        sets = {"intact": np.zeros(0, np.int64), "lLN1_bc": lln, "AL_cholinergic_LNs": eln,
                f"high_drive_{len(lln)}": hd[: len(lln)], f"high_drive_{len(eln)}": hd[: len(eln)]}
        for k in range(3):
            sets[f"degree_matched_{len(lln)}_#{k}"] = matched(lln, pool, outsyn, rng)
            sets[f"degree_matched_{len(eln)}_#{k}"] = matched(eln, pool, outsyn, rng)
        info = {n: dict(out_syn=float(outsyn[s].mean()) if len(s) else 0, rate=float(rate[s].mean()) if len(s) else 0)
                for n, s in sets.items()}
        print(v, {n: (round(d["out_syn"]), round(d["rate"], 1)) for n, d in info.items()}, flush=True)
        combos = [(n, g, s) for n in sets for g in GAINS[v] for s in SEEDS]
        out = SpontSim(con).run([g for _, g, _ in combos], h=H, t_run=T,
                                seed=zlib.crc32(("m" + v).encode()) % 10000, std_U=U, std_tau=TAU,
                                silence=[sets[n] for n, _, _ in combos])
        base = con.n * H * 1e-3
        for b, (n, g, s) in enumerate(combos):
            on, _, _ = detect_bursts(out["pop"][b][300:], base)
            rows.append(dict(variant=v, gain=g, silenced=n, seed=s, n_silenced=len(sets[n]),
                             mean_out_syn=info[n]["out_syn"], mean_rate=info[n]["rate"],
                             n_bursts=len(on), bursts_per_min=len(on) / ((T - 300) / 60000),
                             brain_rate=float(out["pop"][b].sum() / con.n / (T * 1e-3))))
        pd.DataFrame(rows).to_csv(RF / "silencing_matched.csv", index=False)
        print(v, "done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["fixAL", "fixALlit", "typevote", "nomod"])
