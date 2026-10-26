"""Population traces for the video/figures: intact vs lLN1_bc silenced vs 30 random
neurons silenced, at gains with regular bursting (fixAL 2.5, nomod 2.5)."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from spont import SpontSim, load_connectome  # noqa: E402
from variants import load_annotations, make_variant  # noqa: E402
from bursts import H, TAU, U, detect_bursts  # noqa: E402

ROOT = HERE.parents[1]


def main():
    con0 = load_connectome("783")
    ann = load_annotations(con0)
    lln = np.flatnonzero((ann["cell_type"] == "lLN1_bc").to_numpy())
    rnd = np.random.default_rng(3).choice(con0.n, len(lln), replace=False)
    out = {}
    for v in ["fixAL", "nomod"]:
        con, _ = make_variant(v, con0, ann)
        r = SpontSim(con).run([2.5] * 3, h=H, t_run=15000.0, seed=77, std_U=U, std_tau=TAU,
                              silence=[[], lln, rnd])
        for k, name in enumerate(["intact", "lLN1_bc_silenced", "random30_silenced"]):
            out[f"{v}_{name}"] = r["pop"][k]
            on, _, _ = detect_bursts(r["pop"][k][300:], con.n * H * 1e-3)
            print(v, name, len(on), "bursts", flush=True)
    np.savez_compressed(ROOT / "results" / "fly" / "video" / "silence_traces.npz", **out)


if __name__ == "__main__":
    main()
