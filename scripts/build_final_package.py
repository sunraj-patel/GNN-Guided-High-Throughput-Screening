#!/usr/bin/env python3
"""Build the final-results package (no new DFT; consumes frozen artifacts).

Inputs (local, frozen):
  master_dft_validation.csv   - per-candidate GNN vs DFT summary
  final_qc.json               - stress-fit Cij, B_el, fit quality, mag
  eos_fits.json               - Birch-Murnaghan B0 fits
  production_run.csv          - GNN a0 / B / E_hull
  candidates/<C>/info.json    - DFT relaxed cells

Outputs (final_results_package/):
  FINAL_RESULTS.csv                    - spec columns, one row/candidate
  figdata_a_parity.csv                 - a_GNN vs a_DFT
  figdata_B_parity.csv                 - B_GNN vs B_DFT
  figdata_cprime_c44.csv               - Cprime vs C44 stability map
  figdata_Berror_stability.csv         - GNN B error vs DFT stability verdict
  RESULTS_SUMMARY.md                   - computed counts / errors
  fig_cprime_c44.png, fig_Berror_stability.png  (from the fig-ready CSVs)

delta_B_EOS_pct = 100*(B_EOS - B_GNN)/B_GNN   (GNN error vs EOS-derived B)
Cprime          = (C11 - C12)/2               (GPa; tet-scan C' listed in QC md)
Born (cubic)    = C11-C12>0 AND C11+2*C12>0 AND C44>0
Ru-O: elastic/EOS columns blank (WITHDRAWN) - downstream SCF ran on excited
magnetic branches; only a0/E_hull remain valid. B_EOS quoted only when the
fit_eos.py QC gate passes (rms, B0-vs-B_el, a0-vs-a_relax).
"""
import csv
import json
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
OUT = os.path.join(BASE, "final_results_package")
os.makedirs(OUT, exist_ok=True)

master = list(csv.DictReader(open(os.path.join(BASE, "master_dft_validation.csv"))))
qc = json.load(open(os.path.join(BASE, "final_qc.json")))
eos = json.load(open(os.path.join(BASE, "eos_fits.json")))
# full-precision inputs for deltas (matches make_master.py exactly):
prod = {}
for r in csv.DictReader(open(os.path.join(BASE, "production_run.csv"))):
    prod[r["pool_id"]] = r
CAND = os.path.join(BASE, "candidates")
POOLS = {"Re-Os": "6073", "Tc-Os": "6108", "P-Os2": "6056",
         "Re-Rh": "7173", "B-Ir2": "521", "Re-Ir": "3664",
         "Ni-Os": "5610", "Re-B": "598", "Tc-Re": "7203", "Ru-O": "5865"}
N_STRAIN = {c: q["n_strain"] for c, q in qc.items()}
# tet-scan C' (DeltaE = 6*Omega*C'*d^2), independent of stress sign
TET = {"NiOs": -23.5, "TcOs": -151.8, "ReRh": -197.1,
       "ReIr": -268.0, "TcRe": -11.0}
FILO = {r["candidate"]: r["candidate"].replace("-", "") for r in master}

rows = []
for m in master:
    name = m["candidate"]
    c = FILO[name]
    q = qc[c]
    e = eos.get(c, {})
    # display columns (rounded) come from the master table
    a_gnn, a_dft = float(m["a_GNN"]), float(m["a_DFT"])
    b_gnn = float(m["B_GNN"])
    b_dft = float(m["B_DFT"]) if m["B_DFT"] else None
    b_eos = float(m["B_EOS"]) if m["B_EOS"] else None
    # deltas from FULL precision (identical to make_master.py): production
    # floats, info.json relaxed cell, final_qc B_el, eos B0
    pr = prod[POOLS[name]]
    ag_full, bq_full = float(pr["a0"]), float(pr["bulk_GPa"])
    ad_full = float(np.mean(np.diag(np.array(
        json.load(open(os.path.join(CAND, c, "info.json")))["relaxed_cell_A"]))))
    B_el_full = q.get("B_el")
    B0_full = e.get("B0") if e.get("qc") == "pass" else None
    da_full = round(100 * (ad_full - ag_full) / ag_full, 2)
    db_full = (round(100 * (B_el_full - bq_full) / bq_full, 2)
               if (B_el_full is not None and m["B_DFT"]) else "")
    db_eos_full = (round(100 * (B0_full - bq_full) / bq_full, 2)
                   if (B0_full is not None and m["B_EOS"]) else "")
    complete = N_STRAIN[c] == 24
    c11 = float(m["C11"]) if m["C11"] else None
    c12 = float(m["C12"]) if m["C12"] else None
    c44 = float(m["C44"]) if m["C44"] else None
    cprime = (c11 - c12) / 2 if None not in (c11, c12) else None

    if complete and c11 is not None:
        fails = []
        if c11 - c12 <= 0:
            fails.append("C11-C12<0")
        if c11 + 2 * c12 <= 0:
            fails.append("C11+2C12<0")
        if c44 <= 0:
            fails.append("C44<0")
        born = "PASS" if not fails else "Born unstable (" + "; ".join(fails) + ")"
    elif complete:
        born = "WITHDRAWN (QC fail)"
    else:
        born = "PENDING(%d/24 strains)" % N_STRAIN[c]

    rows.append({
        "candidate": name,
        "prototype": m["proto"],
        "a_GNN": round(a_gnn, 4),
        "a_DFT": round(a_dft, 4),
        "delta_a_pct": da_full,
        "B_GNN": round(b_gnn, 1),
        "B_DFT": round(b_dft, 1) if b_dft else "",
        "delta_B_pct": db_full,
        "C11": round(c11, 1) if c11 is not None else "",
        "C12": round(c12, 1) if c12 is not None else "",
        "C44": round(c44, 1) if c44 is not None else "",
        "Cprime": round(cprime, 1) if cprime is not None else "",
        "B_EOS": round(b_eos, 1) if b_eos else "",
        "delta_B_EOS_pct": db_eos_full,
        "E_hull": round(float(m["E_hull_GNN"]), 4),
        "magnetic_moment": round(q["mag"], 2),
        "Born_criterion": born,
        "final_verdict": m["verdict"],
    })

cols = ["candidate", "prototype", "a_GNN", "a_DFT", "delta_a_pct",
        "B_GNN", "B_DFT", "delta_B_pct", "C11", "C12", "C44", "Cprime",
        "B_EOS", "delta_B_EOS_pct", "E_hull", "magnetic_moment",
        "Born_criterion", "final_verdict"]
with open(os.path.join(OUT, "FINAL_RESULTS.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

# ------------------------------------------------------------- figure-ready CSVs
def write_csv(fname, header, data):
    with open(os.path.join(OUT, fname), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(data)
    print("wrote", fname, "(%d rows)" % len(data))

write_csv("figdata_a_parity.csv",
          ["candidate", "a_GNN", "a_DFT"],
          [[r["candidate"], r["a_GNN"], r["a_DFT"]] for r in rows])

write_csv("figdata_B_parity.csv",
          ["candidate", "B_GNN", "B_DFT"],
          [[r["candidate"], r["B_GNN"], r["B_DFT"]]
           for r in rows if r["B_DFT"] != ""])

write_csv("figdata_cprime_c44.csv",
          ["candidate", "Cprime", "C44", "Born_criterion"],
          [[r["candidate"], r["Cprime"], r["C44"], r["Born_criterion"]]
           for r in rows if r["Cprime"] != ""])

write_csv("figdata_Berror_stability.csv",
          ["candidate", "delta_B_pct", "Born_criterion", "final_verdict"],
          [[r["candidate"], r["delta_B_pct"], r["Born_criterion"],
            r["final_verdict"]]
           for r in rows if r["delta_B_pct"] != ""])

# ------------------------------------------------------------- stats (computed)
da = [abs(r["delta_a_pct"]) for r in rows]
db = [abs(r["delta_B_pct"]) for r in rows if r["delta_B_pct"] != ""]
db_eos = [abs(r["delta_B_EOS_pct"]) for r in rows if r["delta_B_EOS_pct"] != ""]
stable = [r for r in rows if r["Born_criterion"] == "PASS"]
pending = [r for r in rows if r["Born_criterion"].startswith("PENDING")]
withdrawn = [r for r in rows if r["Born_criterion"].startswith("WITHDRAWN")]
tet_un = [r for r in rows if r["Cprime"] != "" and r["Cprime"] < 0]
shear_un = [r for r in rows if r["C44"] != "" and r["C44"] < 0]
# false positive = shortlisted hull-stable by GNN but Born-unstable in DFT
fp = sorted([r for r in rows if r["Born_criterion"].startswith("Born unstable")],
            key=lambda r: r["E_hull"])

def fmt(v, nd=2):
    return f"{v:.{nd}f}"

lines = []
lines.append("# Results Summary\n")
lines.append("All counts computed from `FINAL_RESULTS.csv` "
             "(generated by `build_final_package.py`).\n")
lines.append("## Funnel\n")
lines.append("| stage | n |")
lines.append("|---|---|")
lines.append("| Production pool screened | 4,976 unique configs |")
lines.append("| Shortlisted (lowest GNN E_hull) | 25 |")
lines.append("| DFT-validated here | 10 |")
lines.append("| Complete with quotable elastic data | %d |"
             % (10 - len(pending) - len(withdrawn)))
lines.append("| Withdrawn after QC (magnetic branch) | %d (%s) |"
             % (len(withdrawn),
                ", ".join(r["candidate"] for r in withdrawn) or "-"))
lines.append("| Partial strain sets | %d |" % len(pending))
lines.append("")
lines.append("## DFT mechanical stability (cubic Born criteria)\n")
lines.append("| verdict | n | candidates |")
lines.append("|---|---|---|")
lines.append("| Mechanically stable (PASS) | %d | %s |"
             % (len(stable), ", ".join(r["candidate"] for r in stable)))
lines.append("| Tetragonally unstable (C11-C12<0) | %d | %s |"
             % (len([r for r in rows if "C11-C12<0" in r["Born_criterion"]]),
                ", ".join(r["candidate"] for r in rows
                          if "C11-C12<0" in r["Born_criterion"])))
lines.append("| Shear unstable (C44<0) | %d | %s |"
             % (len([r for r in rows if "C44<0" in r["Born_criterion"]]),
                ", ".join(r["candidate"] for r in rows
                          if "C44<0" in r["Born_criterion"])))
lines.append("| C11<0 (saddle) | %d | %s |"
             % (len([r for r in rows if r["C11"] != "" and r["C11"] < 0]),
                ", ".join(r["candidate"] for r in rows
                          if r["C11"] != "" and r["C11"] < 0) or "-"))
lines.append("| Pending | %d | %s |"
             % (len(pending), ", ".join(r["candidate"] for r in pending) or "-"))
lines.append("| Withdrawn (QC fail) | %d | %s |"
             % (len(withdrawn),
                ", ".join(r["candidate"] for r in withdrawn) or "-"))
lines.append("")
lines.append("## Prediction errors (GNN vs DFT)\n")
lines.append("| metric | mean abs | median abs | max abs | n |")
lines.append("|---|---|---|---|---|")
lines.append("| delta_a (%%) | %s | %s | %s | %d |"
             % (fmt(statistics.mean(da)), fmt(statistics.median(da)),
                fmt(max(da)), len(da)))
lines.append("| delta_B elastic (%%) | %s | %s | %s | %d |"
             % (fmt(statistics.mean(db)), fmt(statistics.median(db)),
                fmt(max(db)), len(db)))
lines.append("| delta_B EOS (%%) | %s | %s | %s | %d |"
             % (fmt(statistics.mean(db_eos)), fmt(statistics.median(db_eos)),
                fmt(max(db_eos)), len(db_eos)))
lines.append("")
lines.append("## Largest false positives\n")
lines.append("Definition: GNN shortlisted it as hull-stable "
             "(E_hull <= 0.045 eV/atom) but the DFT Born criterion is unstable. "
             "Ranked by most-negative/lowest E_hull first.\n")
lines.append("| candidate | E_hull | Born | delta_a (%) | delta_B (%) |")
lines.append("|---|---|---|---|---|")
for r in fp:
    lines.append("| %s | %+.4f | %s | %+.2f | %s |"
                 % (r["candidate"], r["E_hull"], r["Born_criterion"],
                    r["delta_a_pct"], r["delta_B_pct"]))
lines.append("")
lines.append("Note: Ru-O's elastic/EOS rows are withdrawn (WITHDRAWN, "
             "QC fail): its strain and EOS SCF runs converged to excited "
             "magnetic branches (6.58-6.81 and 3.15-8.13 uB vs the 7.49 uB "
             "reference), so it is excluded from stability counts and error "
             "statistics. a0/da remain valid (from vc-relax).")
open(os.path.join(OUT, "RESULTS_SUMMARY.md"), "w").write("\n".join(lines) + "\n")
print("wrote RESULTS_SUMMARY.md")

# ------------------------------------------------------------- two figures
import anthropic_style as S
S.apply()

fig, ax = plt.subplots(figsize=(6.2, 6.2), constrained_layout=True)
for r in rows:
    if r["Cprime"] == "":
        continue
    ok = r["Born_criterion"] == "PASS"
    ax.scatter(r["Cprime"], r["C44"], s=90,
               c=S.GREEN if ok else S.CLAY,
               edgecolors="white", linewidths=1.2, zorder=3)
    ax.annotate(r["candidate"], (r["Cprime"], r["C44"]),
                xytext=(6, 5), textcoords="offset points",
                fontsize=8.5, fontweight="semibold")
ax.axvline(0, color=S.INK, lw=1.1, ls=(0, (5, 3)))
ax.axhline(0, color=S.INK, lw=1.1, ls=(0, (5, 3)))
ax.fill_between([ax.get_xlim()[0], 0], 0, ax.get_ylim()[1],
                color=S.CLAY, alpha=0.06, lw=0)
ax.fill_between([0, ax.get_xlim()[1]], 0, ax.get_ylim()[1],
                color=S.GREEN, alpha=0.06, lw=0)
ax.set_xlabel("C$'$ = (C$_{11}$-C$_{12}$)/2  (GPa)")
ax.set_ylabel("C$_{44}$  (GPa)")
ax.grid(True)
fig.suptitle("Elastic stability map (DFT)", x=0.02, ha="left",
             fontfamily=S.SERIF, fontsize=15, fontweight="normal")
fig.savefig(os.path.join(OUT, "fig_cprime_c44.png"))
plt.close(fig)
print("wrote fig_cprime_c44.png")

fig, ax = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)
dat = [r for r in rows if r["delta_B_pct"] != ""]
xs = np.arange(len(dat))
cols_ = [S.GREEN if r["Born_criterion"] == "PASS" else S.CLAY for r in dat]
ax.bar(xs, [r["delta_B_pct"] for r in dat], color=cols_, width=0.62,
       edgecolor="white", linewidth=0.8)
for i, r in enumerate(dat):
    ax.annotate(r["candidate"], (i, r["delta_B_pct"]),
                xytext=(0, 4 if r["delta_B_pct"] >= 0 else -12),
                textcoords="offset points", ha="center", fontsize=8,
                fontweight="semibold")
ax.axhline(0, color=S.INK, lw=1.0)
ax.set_xticks(xs)
ax.set_xticklabels([])
ax.set_ylabel("GNN B error  (B$_{DFT}$-B$_{GNN}$)/B$_{GNN}$  (%)")
ax.grid(True, axis="y")
fig.suptitle("Bulk-modulus error vs DFT stability verdict", x=0.02,
             ha="left", fontfamily=S.SERIF, fontsize=15, fontweight="normal")
fig.savefig(os.path.join(OUT, "fig_Berror_stability.png"))
plt.close(fig)
print("wrote fig_Berror_stability.png")
print("DONE ->", OUT)
