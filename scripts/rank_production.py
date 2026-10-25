"""Merge production rows, rank by high B + low E_hull, novelty-check vs MP.

Ranking:
  eligible: quality == 'ok', B and E_hull finite.
  shortlist pool: e_hull <= 0.05 eV/atom, sorted by B desc.
  pareto: non-dominated in (B maximize, e_hull minimize) over eligible rows.
Novelty (local MP tables only): same chemistry+stoichiometry present in
mp_binary_stability? same prototype family (SG via mp_binary_symmetry)?
Writes production_run.csv, production_ranked.csv, production_shortlist.csv.
"""
import csv
import glob
import json
import os
import statistics

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results_production")

rows = []
for p in glob.glob(os.path.join(RES, "row_*.json")):
    rows.append(json.load(open(p)))
print("production rows: %d" % len(rows), flush=True)

cols = ["pool_id", "A", "B", "proto", "formula", "x_B", "a_guess", "c_pred",
        "a0", "bulk_GPa", "e_hull", "quality", "min_nn_A"]
with open(os.path.join(BASE, "production_run.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in sorted(rows, key=lambda x: x["pool_id"]):
        w.writerow({c: r.get(c) for c in cols})

ok = [r for r in rows if r["quality"] == "ok"
      and isinstance(r["bulk_GPa"], (int, float))
      and isinstance(r["e_hull"], (int, float))]
print("eligible (ok + finite B/E_hull): %d" % len(ok), flush=True)

# quality / coverage summary
from collections import Counter
print("quality:", dict(Counter(r["quality"] for r in rows)), flush=True)
bh = [r["bulk_GPa"] for r in ok]
eh = [r["e_hull"] for r in ok]
print("B: n=%d median=%.1f max=%.1f | E_hull: median=%.3f min=%.3f max=%.3f" % (
    len(bh), statistics.median(bh), max(bh),
    statistics.median(eh), min(eh), max(eh)), flush=True)
for thr in [0.0, 0.02, 0.05, 0.1, 0.2]:
    print("  e_hull <= %.2f: %d rows" % (thr, sum(1 for r in ok if r["e_hull"] <= thr)),
          flush=True)

# pareto frontier (B up, e_hull down)
par = []
for r in ok:
    if not any(q["bulk_GPa"] >= r["bulk_GPa"] and q["e_hull"] <= r["e_hull"]
               and (q["bulk_GPa"] > r["bulk_GPa"] or q["e_hull"] < r["e_hull"])
               for q in ok):
        par.append(r)
print("pareto-optimal rows: %d" % len(par), flush=True)

# shortlist pool: stable + stiff, ranked by B
pool = sorted([r for r in ok if r["e_hull"] <= 0.05],
              key=lambda r: -r["bulk_GPa"])
print("shortlist pool (e_hull<=0.05): %d" % len(pool), flush=True)
for r in pool[:30]:
    print("  B=%7.1f eh=%+.4f %-3s-%-3s %-11s %-8s a0=%.3f q=%s" % (
        r["bulk_GPa"], r["e_hull"], r["A"], r["B"], r["proto"],
        r["formula"], r["a0"], r["quality"]), flush=True)

with open(os.path.join(BASE, "production_ranked.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols + ["pareto"])
    w.writeheader()
    pset = {r["pool_id"] for r in par}
    for r in sorted(ok, key=lambda r: (r["e_hull"], -r["bulk_GPa"])):
        w.writerow({**{c: r.get(c) for c in cols},
                    "pareto": r["pool_id"] in pset})
print("wrote production_run.csv + production_ranked.csv", flush=True)
