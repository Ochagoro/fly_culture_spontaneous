"""Robustness of the ignition site: whole-brain spike recordings at fixAL g = 2.5
with a different spontaneous drive, different short-term depression and other
seeds. Output files follow record_video.py naming with a suffix."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from spont import SpontSim, load_connectome  # noqa: E402
from variants import make_variant  # noqa: E402
from bursts import H, TAU, U, detect_bursts  # noqa: E402

ROOT = HERE.parents[1]
CONDS = {  # tag: (h, U, tau, seed, gain)
    "h0.15": (0.15, U, TAU, 5, 2.5),
    "h0.6": (0.6, U, TAU, 5, 2.5),
    "U0.1tau1000": (H, 0.1, 1000.0, 5, 2.5),
    "seed6": (H, U, TAU, 6, 2.5),
    "seed7": (H, U, TAU, 7, 2.5),
}


def main(variant="fixAL"):
    con, _ = make_variant(variant, load_connectome("783"))
    sim = SpontSim(con)
    dst = ROOT / "results" / "fly" / "video"
    for tag, (h, u, tau, seed, g) in CONDS.items():
        out = sim.run([g], h=h, t_run=10000.0, seed=seed, std_U=u, std_tau=tau,
                      record_events=np.ones(con.n, bool))
        on, off, _ = detect_bursts(out["pop"][0][300:], con.n * h * 1e-3)
        np.savez_compressed(dst / f"spikes_{variant}_g{g:.2f}-{tag}.npz", t=out["ev_t"],
                            n=out["ev_n"].astype(np.int32), pop=out["pop"][0], on=on + 300, off=off + 300,
                            gain=g)
        print(tag, len(on), "bursts", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:2])
