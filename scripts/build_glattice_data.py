"""Build leak-free G_lattice datasets from the strict v2 retrospective sets.

Input representation (NO target leakage):
  Every input structure is the framework prototype builder evaluated at a
  FIXED reference lattice a_ref = 5.0 A for all rows. Graphs therefore differ
  only by composition (species) + prototype topology/orientation. The MP
  equilibrium lattice constant appears NOWHERE in the inputs (not in the
  lattice matrix, not in bond lengths). a_guess is used only as a label-side
  prior for the correction target and at prediction-recovery time, never as
  an input feature.

Targets:
  abs  : a0_MP (A, conventional cubic lattice constant)
  corr : c = a0_MP / a_guess   (recover a0_pred = a_guess * c_pred)

Rows: retrospective_v2_exact.csv (A-B, 325) + retrospective_v2_aa.csv (A-A, 87).
Fluorite AB2/A2B orientations are preserved as distinct rows via
resolve_builder_and_xB (same builder the framework uses).

Splits (recorded in glattice_splits.json):
  disjoint : composition-disjoint test/val. A-B grouped by unordered pair,
             A-A grouped by element; stratified by (subset, prototype) so each
             prototype is represented in test. No chemistry shared across
             train/val/test.
  random   : standard random 76/12/12 split, stratified by (subset, prototype).

Output dirs (one id_prop.csv each, row order = train, val, test for
keep_data_order training):
  glattice_abs_disjoint/ glattice_corr_disjoint/
  glattice_abs_random/   glattice_corr_random/
CIFs live in glattice_structures/ and are symlinked into each run dir.
"""
import csv
import json
import os
import random
import sys
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import screen_one as S

A_REF = 5.0
STRUCT_DIR = os.path.join(BASE, "glattice_structures")
os.makedirs(STRUCT_DIR, exist_ok=True)

# ---------------- load rows ----------------
ab = list(csv.DictReader(open(os.path.join(BASE, "retrospective_v2_exact.csv"))))
aa = list(csv.DictReader(open(os.path.join(BASE, "retrospective_v2_aa.csv"))))
guess_map = {}
for r in csv.DictReader(open(os.path.join(BASE, "framework_run_dedup.csv"))):
    guess_map[r["pool_id"]] = float(r["a_guess"])

rows = []
for r in ab:
    rows.append(("AB", r))
for r in aa:
    rows.append(("AA", r))
print("loaded rows: AB=%d AA=%d" % (len(ab), len(aa)), flush=True)

# ---------------- build structures + labels ----------------
meta = {}  # tag -> metadata
for subset, r in rows:
    A, B, proto, formula = r["A"], r["B"], r["proto"], r["formula"]
    a_mp = float(r["a_mp"])
    a_guess = guess_map[r["pool_id"]]
    builder, xB = S.resolve_builder_and_xB(proto, formula, A, B)
    atoms = builder(A_REF)
    tag = "%s_%s" % (subset, r["pool_id"])
    # POSCAR (not CIF): jarvis from_cif primitivizes fcc prototypes, which
    # would silently change the representation per prototype. from_poscar
    # round-trips the conventional cell exactly (verified).
    vasp = os.path.join(STRUCT_DIR, tag + ".vasp")
    if not os.path.exists(vasp):
        atoms.write_poscar(vasp)
    meta[tag] = {
        "tag": tag, "subset": subset, "pool_id": r["pool_id"],
        "A": A, "B": B, "proto": proto, "formula": formula,
        "x_B": float(r["x_B"]) if r.get("x_B") not in ("", None) else None,
        "a_mp": a_mp, "a_guess": a_guess,
        "a_energy": float(r["a_pred"]),  # current G_energy(a) result
        "quality": r.get("quality", ""),
        "corr_true": a_mp / a_guess,
    }

# leak check: every input lattice must be exactly A_REF cubic conventional,
# parsed through the same from_poscar path the trainer uses
import numpy as np
from jarvis.core.atoms import Atoms as JAtoms
ref = None
for tag in meta:
    a = JAtoms.from_poscar(os.path.join(STRUCT_DIR, tag + ".vasp"))
    m = np.array(a.lattice_mat)
    if ref is None:
        ref = m
    assert np.allclose(m, ref), (tag, m)
assert np.allclose(ref, A_REF * np.eye(3)), ref
print("leak check OK: all %d input lattices are exactly %.1f A cubic"
      % (len(meta), A_REF), flush=True)

# ---------------- splits ----------------
def chemkey(m):
    if m["subset"] == "AA":
        return ("AA", m["A"])
    return ("AB", tuple(sorted((m["A"], m["B"]))))

def strat_key(m):
    return (m["subset"], m["proto"])

def split_disjoint(tags, seed=42):
    rng = random.Random(seed)
    groups = defaultdict(list)   # chemkey -> [tags]
    for t in tags:
        groups[chemkey(meta[t])].append(t)
    # stratify: within each (subset, proto) cell, split its chemistries
    cells = defaultdict(list)    # strat -> [chemkeys]
    for ck in groups:
        cells[strat_key(meta[groups[ck][0]])].append(ck)
    tr, va, te = [], [], []
    for cell, cks in sorted(cells.items()):
        rng.shuffle(cks)
        n = len(cks)
        n_te = max(1, round(0.12 * n)) if n >= 3 else (1 if n == 2 else 0)
        n_va = max(1, round(0.12 * n)) if n - n_te >= 3 else (1 if n - n_te == 2 else 0)
        for ck in cks[:n_te]:
            te += groups[ck]
        for ck in cks[n_te:n_te + n_va]:
            va += groups[ck]
        for ck in cks[n_te + n_va:]:
            tr += groups[ck]
    # verify disjointness
    def chems(ts):
        return {chemkey(meta[t]) for t in ts}
    assert not (chems(tr) & chems(va) or chems(tr) & chems(te) or chems(va) & chems(te))
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return tr, va, te

def split_random(tags, seed=42):
    rng = random.Random(seed)
    cells = defaultdict(list)
    for t in tags:
        cells[strat_key(meta[t])].append(t)
    tr, va, te = [], [], []
    for cell, ts in sorted(cells.items()):
        rng.shuffle(ts)
        n = len(ts)
        n_te = max(1, round(0.12 * n)) if n >= 3 else (1 if n == 2 else 0)
        n_va = max(1, round(0.12 * n)) if n - n_te >= 3 else (1 if n - n_te == 2 else 0)
        te += ts[:n_te]; va += ts[n_te:n_te + n_va]; tr += ts[n_te + n_va:]
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return tr, va, te

tags = sorted(meta)
splits = {}
for name, fn in [("disjoint", split_disjoint), ("random", split_random)]:
    tr, va, te = fn(tags)
    splits[name] = {"train": tr, "val": va, "test": te}
    print("%-8s train=%d val=%d test=%d" % (name, len(tr), len(va), len(te)), flush=True)
    for cell in sorted({strat_key(meta[t]) for t in tags}):
        c = lambda ts: sum(1 for t in ts if strat_key(meta[t]) == cell)
        print("   %-16s train=%3d val=%2d test=%2d" % (str(cell), c(tr), c(va), c(te)), flush=True)

json.dump({"meta": meta, "splits": splits},
          open(os.path.join(BASE, "glattice_splits.json"), "w"))

# ---------------- run dirs ----------------
def write_rundir(dname, order, target):
    d = os.path.join(BASE, dname)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "id_prop.csv"), "w") as f:
        for t in order:
            m = meta[t]
            y = m["a_mp"] if target == "abs" else m["corr_true"]
            f.write("%s.vasp,%.8f\n" % (t, y))
            dst = os.path.join(d, t + ".vasp")
            if not os.path.lexists(dst):
                os.symlink(os.path.join(STRUCT_DIR, t + ".vasp"), dst)
    print("wrote %s (%s, n=%d)" % (dname, target, len(order)), flush=True)

for split in ["disjoint", "random"]:
    order = splits[split]["train"] + splits[split]["val"] + splits[split]["test"]
    write_rundir("glattice_abs_%s" % split, order, "abs")
    write_rundir("glattice_corr_%s" % split, order, "corr")

print("DONE")
