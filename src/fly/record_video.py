"""Record every spike of the whole fly brain during spontaneous bursting, for the video.

One trial at a gain just above burst onset (fixAL variant, short-term
depression as in bursts.py). Saves spike times (0.1 ms steps) and neuron
indices for all 138,639 neurons, plus the population rate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from spont import SpontSim, load_connectome  # noqa: E402
from variants import make_variant  # noqa: E402
from bursts import H, TAU, U, detect_bursts  # noqa: E402

ROOT = HERE.parents[1]


def main(variant="fixAL", gain=2.0, t_run=10000.0, seed=5):
    con, ann = make_variant(variant, load_connectome("783"))
    sim = SpontSim(con)
    out = sim.run([gain], h=H, t_run=t_run, seed=seed, std_U=U, std_tau=TAU,
                  record_events=np.ones(con.n, bool))
    on, off, _ = detect_bursts(out["pop"][0][300:], con.n * H * 1e-3)
    dst = ROOT / "results" / "fly" / "video"
    dst.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst / f"spikes_{variant}_g{gain:.2f}.npz", t=out["ev_t"], n=out["ev_n"].astype(np.int32),
                        pop=out["pop"][0], on=on + 300, off=off + 300, gain=gain)
    print(len(out["ev_t"]), "spikes;", len(on), "bursts at", on + 300)


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["fixAL"]), *(float(x) for x in sys.argv[2:3]))
