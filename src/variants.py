"""Transmitter-sign assumptions for the FlyWire LIF model.

The connectome gives wiring but not synaptic sign; signs come from the
machine-learning transmitter predictions (Eckstein et al. 2024), several of
which are known to be wrong (e.g. 40 antennal-lobe local neurons predicted to
be serotonergic, which the model treats as excitatory). Following the lesson
of the mirror-test study, low-confidence predictions are treated as
assumptions: every result is recomputed under several sign variants.

shiu      : signs exactly as in Shiu et al. (2024).
fixAL     : as shiu, but the 40 antennal-lobe local neurons predicted to be
            serotonergic are made inhibitory (AL local neurons are GABA/Glu).
typevote  : one transmitter per cell type (confidence-weighted vote of the
            per-neuron predictions), overridden by literature-known transmitters
            ('known_nt'); histamine is inhibitory.
nomod     : as typevote, but dopamine/serotonin/octopamine neurons have no fast
            synaptic effect (sign 0), i.e. monoamines act only as modulators.
fixALlit  : as fixAL, but those of the 40 neurons whose cell type has a
            literature-known transmitter of acetylcholine (FlyWire 'known_nt',
            e.g. lLN1_bc; Shang et al. 2007) stay excitatory (added after review:
            fixAL made 7 of the 30 lLN1_bc inhibitory).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sim import DATA, Connectome, load_connectome

SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1,
        "dopamine": 1, "serotonin": 1, "octopamine": 1, "tyramine": 1}
FAST = ["acetylcholine", "gaba", "glutamate", "histamine"]
MOD = ["dopamine", "serotonin", "octopamine", "tyramine"]
VARIANTS = ["shiu", "fixAL", "fixALlit", "typevote", "nomod"]


def load_annotations(con: Connectome) -> pd.DataFrame:
    ann = pd.read_csv(DATA / "neuron_annotations.tsv", sep="\t", low_memory=False)
    ann = ann.drop_duplicates("root_id").set_index("root_id").reindex(con.ids)
    return ann


def _known(s) -> str | None:
    if not isinstance(s, str):
        return None
    names = set()
    for part in s.replace(";", ",").split(","):
        part = part.strip()
        if not part or part.endswith("-negative"):
            continue
        if part in SIGN:
            names.add(part)
    names_fast = names & set(FAST)
    if len(names_fast) == 1:
        return names_fast.pop()
    if not names_fast and len(names) == 1:
        return names.pop()
    return None


def transmitter_table(ann: pd.DataFrame) -> pd.Series:
    """One transmitter per neuron: cell-type vote, then literature override."""
    nt = ann["top_nt"].copy()
    conf = ann["top_nt_conf"].fillna(0.5)
    df = pd.DataFrame({"type": ann["cell_type"], "nt": nt, "conf": conf})
    vote = (df.dropna(subset=["type", "nt"]).groupby(["type", "nt"])["conf"].sum()
            .reset_index().sort_values("conf", ascending=False).drop_duplicates("type")
            .set_index("type")["nt"])
    has_type = df["type"].notna() & df["type"].isin(vote.index)
    nt = nt.where(~has_type, df["type"].map(vote))
    known = ann["known_nt"].map(_known)
    # literature transmitter, voted per type as well so that a type is consistent
    kdf = pd.DataFrame({"type": ann["cell_type"], "k": known}).dropna()
    kvote = kdf.groupby("type")["k"].agg(lambda x: x.value_counts().index[0])
    k_type = ann["cell_type"].map(kvote)
    nt = nt.where(k_type.isna(), k_type)
    nt = nt.where(known.isna() | ann["cell_type"].notna(), known)
    return nt


def make_variant(name: str, con: Connectome | None = None, ann: pd.DataFrame | None = None):
    con = con or load_connectome("783")
    ann = ann if ann is not None else load_annotations(con)
    sign = con.sign.copy()
    if name == "shiu":
        pass
    elif name in ("fixAL", "fixALlit"):
        al5ht = ((ann["cell_class"] == "ALLN") & (ann["top_nt"] == "serotonin")).to_numpy()
        if name == "fixALlit":
            known = ann["known_nt"].map(_known)
            kdf = pd.DataFrame({"type": ann["cell_type"], "k": known}).dropna()
            kvote = kdf.groupby("type")["k"].agg(lambda x: x.value_counts().index[0])
            lit_ach = (ann["cell_type"].map(kvote) == "acetylcholine").to_numpy()
            al5ht = al5ht & ~lit_ach
        sign[al5ht] = -1
    elif name in ("typevote", "nomod"):
        nt = transmitter_table(ann)
        s = nt.map(SIGN)
        sign = np.where(s.notna(), s.fillna(0).to_numpy(), con.sign).astype(np.int64)
        if name == "nomod":
            sign[nt.isin(MOD).to_numpy()] = 0
    else:
        raise ValueError(name)
    return con.copy(sign=sign, meta=dict(con.meta, variant=name)), ann
