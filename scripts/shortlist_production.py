"""DFT shortlist: dedupe, novelty-check vs local MP, write shortlist.

Dedup: unordered (A,B) + proto + orientation for fluorite (x_B distinguishes
AB2/A2B). Novelty per candidate from local MP tables:
  known_same_structure: MP entry, same composition+stoichiometry, SG == proto SG
  known_chemistry:      same composition+stoichiometry in MP, other structure
  novel_chemistry:      composition+stoichiometry absent from MP tables
Selection: top by B within e_hull<=0.05, balanced across prototype families
(CsCl / Rocksalt / Zincblende / Fluorite-AB2 / Fluorite-A2B), then overall top
to fill 25 rows.
"""
import csv
import json
import os
import re
import statistics
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
ELEMS = sorted(["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na",
                "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti",
                "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
                "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo",
                "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",
                "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm",
                "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
                "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb",
                "Bi"], key=len, reverse=True)
ESET = set(ELEMS)
PROTO_SG = {"CsCl": 221, "Rocksalt": 225, "Zincblende": 216, "Fluorite": 225}


def parse_formula(f):
    d = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", f):
        if el and el in ESET:
            d[el] = d.get(el, 0) + (int(n) if n else 1)
    return d


# MP index: (lo, hi, xB) -> list of (mp_id, sg)
sym = {}
for r in csv.DictReader(open(os.path.join(BASE, "mp_binary_symmetry.csv"))):
    try:
        sym[r["material_id"]] = int(r["space_group_number"])
    except (ValueError, KeyError):
        pass
mp_index = defaultdict(list)
for r in csv.DictReader(open(os.path.join(BASE, "mp_binary_stability.csv"))):
    d = parse_formula(r["formula"])
    if len(d) != 2:
        continue
    (A, na), (B, nb) = sorted(d.items())
    x = round(nb / (na + nb), 3)
    mp_index[(A, B, x)].append((r["material_id"], sym.get(r["material_id"])))
    if x != 0.5:
        mp_index[(A, B, round(1.0 - x, 3))].append((r["material_id"], sym.get(r["material_id"])))
print("MP chemistry entries indexed", flush=True)

rows = list(csv.DictReader(open(os.path.join(BASE, "production_ranked.csv"))))

# dedupe unordered chemistry (+fluorite orientation via x_B)
seen, uniq = set(), []
for r in sorted(rows, key=lambda r: -float(r["bulk_GPa"])):
    A, B = r["A"], r["B"]
    key = (tuple(sorted((A, B))), r["proto"],
           round(float(r["x_B"]), 3) if r["proto"] == "Fluorite" else 0.5)
    if key in seen:
        continue
    seen.add(key)
    uniq.append(r)
print("deduped unique configs: %d" % len(uniq), flush=True)


def novelty(r):
    A, B = r["A"], r["B"]
    if A == B:
        return "elemental (MP polymorph check separate)"
    lo, hi = sorted((A, B))
    xB = round(float(r["x_B"]), 3)
    cands = mp_index.get((lo, hi, xB), []) or mp_index.get((lo, hi, round(1.0 - xB, 3)), [])
    if not cands:
        return "novel_chemistry"
    want = PROTO_SG[r["proto"]]
    same = [m for m, sg in cands if sg == want]
    if same:
        return "known_same_structure:%s" % ("|".join(same[:3]))
    sgs = sorted({sg for _, sg in cands if sg})
    return "known_chemistry_other_structure(MP SGs %s)" % sgs


pool = [r for r in uniq if float(r["e_hull"]) <= 0.05]
pool = sorted(pool, key=lambda r: -float(r["bulk_GPa"]))

# balanced pick: 5 per family, then fill by B to 25
fams = ["CsCl", "Rocksalt", "Zincblende", "Fluorite"]
short, used = [], set()
for fam in fams:
    fam_rows = [r for r in pool if r["proto"] == fam][:5]
    for r in fam_rows:
        if r["pool_id"] not in used:
            short.append(r); used.add(r["pool_id"])
for r in pool:
    if len(short) >= 25:
        break
    if r["pool_id"] not in used:
        short.append(r); used.add(r["pool_id"])
short = sorted(short, key=lambda r: -float(r["bulk_GPa"]))

cols = ["pool_id", "A", "B", "proto", "formula", "x_B", "a_guess", "c_pred",
        "a0", "bulk_GPa", "e_hull", "min_nn_A", "novelty"]
with open(os.path.join(BASE, "production_shortlist.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in short:
        nv = novelty(r)
        w.writerow({c: r.get(c) for c in cols if c != "novelty"} | {"novelty": nv})
        print("  B=%7.1f eh=%+.4f %-3s-%-3s %-11s %-8s a0=%.3f %s" % (
            float(r["bulk_GPa"]), float(r["e_hull"]), r["A"], r["B"],
            r["proto"], r["formula"], float(r["a0"]), nv), flush=True)
print("wrote production_shortlist.csv (%d rows)" % len(short), flush=True)
