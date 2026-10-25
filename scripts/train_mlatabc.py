"""MLatticeABC-style benchmark on the identical G_lattice splits.

Mirrors Li et al., ACS Omega 2021 (github.com/usccolumbia/MLatticeABC):
  descriptors = Magpie ElementProperty + Stoichiometry + ValenceOrbital(frac)
                + IonProperty(fast) + BandCenter + ElementFraction
                + mlmd atom-count stats (num_atoms, max, min, mean, var)
                + NComp, with the SpaceGroupNumber columns dropped (as in
                their predict.py).
  model = RandomForestRegressor(n_estimators=100, max_features=70,
          max_depth=32, min_samples_split=8), squared_error criterion
          ('mse' in their code = squared_error in current sklearn).

ONE documented augmentation: their descriptors are composition-only, which
cannot distinguish the same composition in different prototypes (a core axis
of our task). We append prototype one-hot + conventional-cell atom count +
x_B so the benchmark can represent the task. Composition features themselves
are exactly theirs.

Trained per (split, target) on the manifest train tags (RF needs no val set;
val tags are unused here), tested on manifest test tags.
Output: glattice_mlatabc_preds.json {disjoint,random}_{abs,corr} -> tag->pred.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from pymatgen.core.composition import Composition
from sklearn.ensemble import RandomForestRegressor

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf

sp = json.load(open(os.path.join(BASE, "glattice_splits.json")))
META, SPLITS = sp["meta"], sp["splits"]

FEATS = MultipleFeaturizer([
    cf.ElementProperty.from_preset(preset_name="magpie"),
    cf.Stoichiometry(),
    cf.ValenceOrbital(props=["frac"]),
    cf.IonProperty(fast=True),
    cf.BandCenter(),
    cf.ElementFraction(),
])

DROP = ["MagpieData minimum SpaceGroupNumber", "MagpieData maximum SpaceGroupNumber",
        "MagpieData range SpaceGroupNumber", "MagpieData mean SpaceGroupNumber",
        "MagpieData avg_dev SpaceGroupNumber", "MagpieData mode SpaceGroupNumber"]

PROTOS = ["CsCl", "Rocksalt", "Zincblende", "Fluorite"]
NCELL = {"CsCl": 2, "Rocksalt": 8, "Zincblende": 8, "Fluorite": 12}


def mlmd_row(formula):
    comp = Composition(formula)
    data = np.array(list(comp.as_dict().values()), dtype=float)
    return {"mlmd_nat": comp.num_atoms, "mlmd_max": data.max(),
            "mlmd_min": data.min(), "mlmd_mean": data.mean(),
            "mlmd_var": data.var()}


def fstr(m):
    import math
    if m["A"] == m["B"]:
        return m["A"]
    xB = m["x_B"] if m["x_B"] is not None else 0.5
    na, nb = round((1.0 - xB) * 12), round(xB * 12)
    g = math.gcd(na, nb) or 1
    na, nb = na // g, nb // g
    s = m["A"] + ("" if na == 1 else str(na)) + m["B"] + ("" if nb == 1 else str(nb))
    return s


def build_frame(tags):
    # NB: StrToComposition hangs on the full 412-row frame in this env
    # (isolated 20-row subsets run in ~1 s; root cause not pursued). Bypass:
    # assign exact fractional Composition objects directly.
    comps = []
    for t in tags:
        m = META[t]
        xB = m["x_B"] if m["x_B"] is not None else 0.5
        comps.append(Composition({m["A"]: 1.0 - xB, m["B"]: xB}))
    df = pd.DataFrame({"composition": comps})
    df = FEATS.featurize_dataframe(df, col_id="composition")
    df["NComp"] = df["composition"].apply(len)
    mm = pd.DataFrame([mlmd_row(c.formula.replace(" ", "")) for c in comps])
    for c in mm.columns:
        df[c] = mm[c].values
    for m, t in zip((META[t] for t in tags), tags):
        pass
    df["proto"] = [META[t]["proto"] for t in tags]
    df["x_B"] = [META[t]["x_B"] if META[t]["x_B"] is not None else 0.5 for t in tags]
    df["ncell"] = [NCELL[META[t]["proto"]] for t in tags]
    for p in PROTOS:
        df["is_%s" % p] = (df["proto"] == p).astype(float)
    feat_cols = [c for c in df.columns
                 if c not in ("formula", "composition", "proto") and c not in DROP]
    return df[feat_cols].fillna(0).values, feat_cols


def target(tags, which):
    return np.array([META[t]["a_mp"] if which == "abs" else META[t]["corr_true"]
                     for t in tags])


def main():
    print("featurizing 412 rows (matminer, one pass)...", flush=True)
    ALL = sorted(META)
    X_all, cols = build_frame(ALL)
    print("descriptor dim = %d" % len(cols), flush=True)
    X = {t: v for t, v in zip(ALL, X_all)}

    out = {}
    for split in ["disjoint", "random"]:
        S = SPLITS[split]
        Xtr = np.array([X[t] for t in S["train"]])
        for which in ["abs", "corr"]:
            ytr = target(S["train"], which)
            Xte = np.array([X[t] for t in S["test"]])
            yte = target(S["test"], which)
            rf = RandomForestRegressor(
                n_estimators=100, max_features=min(70, Xtr.shape[1]),
                max_depth=32, min_samples_split=8, n_jobs=8, random_state=42)
            rf.fit(Xtr, ytr)
            p = rf.predict(Xte)
            key = "%s_%s" % (split, which)
            out[key] = {t: float(v) for t, v in zip(S["test"], p)}
            print("%-14s test_target_MAE=%.5f" % (
                key, float(np.mean(np.abs(p - yte)))), flush=True)

    json.dump(out, open(os.path.join(BASE, "glattice_mlatabc_preds.json"), "w"))
    print("wrote glattice_mlatabc_preds.json")


# NB: matminer's featurize_dataframe uses multiprocessing-spawn, which
# re-imports this file in workers; the __main__ guard is REQUIRED (without it
# workers recurse into training and the run hangs/dies).
if __name__ == "__main__":
    main()
