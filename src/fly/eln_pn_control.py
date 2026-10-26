"""Control for the disputed eLN -> PN chemical connection.

In vivo the excitatory (cholinergic) AL local neurons couple to projection
neurons mainly through gap junctions (Yaksi & Wilson 2010), which the connectome
does not contain, while the model turns the EM-detected chemical synapses into
excitation. Here all chemical synapses from AL cholinergic LNs onto AL
projection neurons are removed (eLN -> LN synapses are kept) and we ask whether
the cholinergic LNs remain (i) the leading mode of the excitatory wiring,
(ii) the first self-sustaining circuit, and (iii) a hub whose silencing lowers
brain-wide spontaneous activity and bursting.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from spont import SpontSim, load_connectome  # noqa: E402
from variants import load_annotations, make_variant, transmitter_table  # noqa: E402
from bursts import H, TAU, U, detect_bursts, groups_of  # noqa: E402

ROOT = HERE.parents[1]


def main():
    con0 = load_connectome("783")
    ann = load_annotations(con0)
    nt = transmitter_table(ann)
    eln = ((ann["cell_class"] == "ALLN") & (nt == "acetylcholine")).to_numpy()
    pn = (ann["cell_class"] == "ALPN").to_numpy()
    lln = np.flatnonzero((ann["cell_type"] == "lLN1_bc").to_numpy())
    rnd = np.random.default_rng(5).choice(con0.n, len(lln), replace=False)
    codes, names = groups_of(ann)
    rows = []
    for v in ["fixAL", "nomod"]:
        con, _ = make_variant(v, con0, ann)
        cut = eln[con.pre] & pn[con.post]
        c2 = con.copy(count=np.where(cut, 0, con.count))
        print(v, "removed eLN->PN synapses:", int(con.count[cut].sum()), flush=True)
        # (i) Perron mode
        w = c2.count.astype(np.float64) * np.clip(c2.sign[c2.pre], 0, None)
        Wp = sp.csr_matrix((w, (c2.post, c2.pre)), shape=(c2.n, c2.n))
        _, vec = eigs(Wp, k=1, which="LR", tol=1e-6, maxiter=5000)
        vec = np.abs(vec[:, 0]); vec /= vec.sum()
        g = pd.Series(vec).groupby(names[codes]).sum().sort_values(ascending=False)
        print("  Perron top:", g.head(8).round(3).to_dict(), flush=True)
        rows.append(dict(variant=v, test="perron_share_AL_cholinergic_LNs", value=float(vec[eln].sum())))
        rows.append(dict(variant=v, test="perron_share_KC", value=float(vec[(ann.cell_class == "Kenyon_Cell").to_numpy()].sum())))
        # (ii) ladder (no STD)
        sim = SpontSim(c2)
        gains = [0.8, 1.0, 1.2, 1.5, 1.75, 2.0]
        allidx = np.arange(c2.n)
        o = sim.run(gains, h=0.0, t_run=2500, seed=11, stim=[[(allidx, 0, 500, 0.5)]] * len(gains), count_from_ms=1500)
        for k, gg in enumerate(gains):
            core = o["counts"][k] >= 10
            rows.append(dict(variant=v, test=f"ladder_core_g{gg}", value=int(core.sum())))
            rows.append(dict(variant=v, test=f"ladder_core_ALLN_g{gg}", value=int((core & (ann.cell_class == 'ALLN').to_numpy()).sum())))
        # (iii) silencing at g = 2.5 with STD
        combos = [("intact", []), ("lLN1_bc", lln), ("random30", rnd)]
        o = sim.run([2.5] * 6, h=H, t_run=15000.0, seed=21, std_U=U, std_tau=TAU,
                    silence=[c[1] for c in combos for _ in range(2)])
        for k in range(6):
            name = combos[k // 2][0]
            on, _, _ = detect_bursts(o["pop"][k][300:], c2.n * H * 1e-3)
            rows.append(dict(variant=v, test=f"bursts_per_min_{name}", value=len(on) / (14.7 / 60)))
            rows.append(dict(variant=v, test=f"mean_rate_{name}", value=float(o["pop"][k].sum() / c2.n / 15.0)))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "fly" / "eln_pn_control.csv", index=False)
    print(df.groupby(["variant", "test"]).value.mean().round(3).to_string())


if __name__ == "__main__":
    main()
