"""Do the left and right copies of a region share fluctuations that the rest of
the brain does not? For each bilateral region pair: r(L, R) of dF/F versus the
mean r of each side with all other regions, and the partial correlation of L and
R after regressing out the global mean signal."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from turner_load import load  # noqa: E402
from turner_metrics import TD, dff, region_names  # noqa: E402

ROOT = HERE.parents[1]


def main():
    names = region_names()
    rows = []
    for fn in sorted((TD / "ito_responses").glob("ito_*.pkl")):
        df = load(fn)
        df = df.loc[[i for i in df.index if isinstance(names.get(i), str)]]
        reg = [names[i] for i in df.index]
        X = dff(df.to_numpy(np.float64))
        Z = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
        C = np.corrcoef(Z)
        g = Z.mean(0)
        resid = Z - np.outer(Z @ g / (g @ g), g)
        Cp = np.corrcoef(resid)
        idx = {r: k for k, r in enumerate(reg)}
        for r in reg:
            if not r.endswith("_L"):
                continue
            rr = r[:-2] + "_R"
            if rr not in idx:
                continue
            i, j = idx[r], idx[rr]
            others = [k for k in range(len(reg)) if k not in (i, j)]
            rows.append(dict(fly=fn.stem, region=r[:-2], r_LR=C[i, j],
                             r_others=(C[i, others].mean() + C[j, others].mean()) / 2,
                             partial_LR=Cp[i, j]))
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "results" / "fly" / "turner_homolog.csv", index=False)
    s = d.groupby("region")[["r_LR", "r_others", "partial_LR"]].median()
    s["excess"] = s.r_LR - s.r_others
    print(s.sort_values("partial_LR", ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
