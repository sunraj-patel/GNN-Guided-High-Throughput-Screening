"""XGBoost baseline for G_lattice (diagnostic only).

Features (no target leakage, no a_guess as a feature):
  rA, rB (atomic_radii.json), r_sum, r_diff, r_prod, x_B, prototype one-hot.
The radii prior enters only label-side (corr target) and at recovery
(a0 = a_guess * c_pred), exactly paralleling the ALIGNN corr setup.

Trains XGBRegressor per (split, target) on the same train/val/test tags as
the ALIGNN runs. Test predictions -> glattice_xgb_preds.json.
"""
import json
import os
import sys

import numpy as np
import xgboost as xgb

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

RADII = json.load(open(os.path.join(BASE, "atomic_radii.json")))
PROTOS = ["CsCl", "Rocksalt", "Zincblende", "Fluorite"]

sp = json.load(open(os.path.join(BASE, "glattice_splits.json")))
META, SPLITS = sp["meta"], sp["splits"]


def feats(m):
    rA, rB = RADII[m["A"]], RADII[m["B"]]
    x = [rA, rB, rA + rB, abs(rA - rB), rA * rB,
         m["x_B"] if m["x_B"] is not None else 0.5]
    x += [1.0 if m["proto"] == p else 0.0 for p in PROTOS]
    return x


def target(m, which):
    return m["a_mp"] if which == "abs" else m["corr_true"]


out = {}
for split in ["disjoint", "random"]:
    for which in ["abs", "corr"]:
        S = SPLITS[split]
        Xtr = np.array([feats(META[t]) for t in S["train"]])
        ytr = np.array([target(META[t], which) for t in S["train"]])
        Xva = np.array([feats(META[t]) for t in S["val"]])
        yva = np.array([target(META[t], which) for t in S["val"]])
        Xte = np.array([feats(META[t]) for t in S["test"]])
        reg = xgb.XGBRegressor(
            n_estimators=3000, learning_rate=0.02, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
            random_state=42, n_jobs=8, early_stopping_rounds=100,
        )
        reg.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        p = reg.predict(Xte)
        key = "%s_%s" % (split, which)
        out[key] = {t: float(v) for t, v in zip(S["test"], p)}
        mae = float(np.mean(np.abs(p - np.array(
            [target(META[t], which) for t in S["test"]]))))
        print("%-14s best_iter=%4d test_target_MAE=%.5f"
              % (key, reg.best_iteration, mae), flush=True)

json.dump(out, open(os.path.join(BASE, "glattice_xgb_preds.json"), "w"))
print("wrote glattice_xgb_preds.json")
