#!/usr/bin/env python3
"""Final master table + parity plots (GNN vs DFT).

Reads: production_run.csv (GNN), candidates/*/info.json (DFT a0 + ref),
final_qc.json (Cij/B fits), eos_fits.json (B0), tet C' values.
Writes: master_dft_validation.csv/.xlsx, fig_parity_a0.png, fig_parity_B.png.
Incomplete compounds show BEST AVAILABLE with status flags; rerun after
closeout to finalize.
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAND = os.path.join(BASE, "candidates")

prod = {}
for r in csv.DictReader(open(os.path.join(BASE, "production_run.csv"))):
    prod[r["pool_id"]] = r
qc = json.load(open(os.path.join(BASE, "final_qc.json")))
# only complete (24/24) strain sets give quotable Cij/B; partial fits are
# shown for tracking but flagged preliminary
QC_N = {c: q.get("n_strain", 0) for c, q in qc.items()}
try:
    eos = json.load(open(os.path.join(BASE, "eos_fits.json")))
except FileNotFoundError:
    eos = {}
TET = {"NiOs": -23.5, "TcOs": -151.8, "ReRh": -197.1,
       "ReIr": -268.0, "TcRe": -11.0}
POOLS = {"Re-Os": "6073", "Tc-Os": "6108", "P-Os2": "6056",
         "Re-Rh": "7173", "B-Ir2": "521", "Re-Ir": "3664",
         "Ni-Os": "5610", "Re-B": "598", "Tc-Re": "7203", "Ru-O": "5865"}
FILO = {"Re-Os": "ReOs", "Tc-Os": "TcOs", "P-Os2": "POs2",
        "Re-Rh": "ReRh", "B-Ir2": "BIr2", "Re-Ir": "ReIr",
        "Ni-Os": "NiOs", "Re-B": "ReB", "Tc-Re": "TcRe", "Ru-O": "RuO"}
VERDICT = {"Re-Os": "tetragonally unstable (C11-C12<0)",
           "Tc-Os": "tetragonally unstable (C11-C12<0)",
           "P-Os2": "mechanically stable",
           "Re-Rh": "tetragonally unstable (C11-C12<0)",
           "B-Ir2": "mechanically stable",
           "Re-Ir": "tetragonally unstable (C11<0 saddle)",
           "Ni-Os": "tetragonally unstable (tet-proven)",
           "Re-B": "mechanically stable",
           "Tc-Re": "marginally unstable (C11-C12<0)"}


def born_verdict(c11, c12, c44):
    if None in (c11, c12, c44):
        return "pending"
    if c11 - c12 <= 0:
        return "tetragonally unstable (C11-C12<0)"
    if c44 <= 0:
        return "unstable (C44<0)"
    return "mechanically stable"


# Withdrawal is DATA-DRIVEN: a candidate is withdrawn iff its downstream SCFs
# left the reference spin branch (mag_branch_ok is False in final_qc.json).
# (Ru-O previously ran on excited branches; a consistent-branch rerun clears it.)
WITHDRAWN = {name for name, c in FILO.items()
             if qc.get(c, {}).get("mag_branch_ok") is False}

rows = []
for name, pid in POOLS.items():
    r = prod[pid]
    c = FILO[name]
    d = json.load(open(os.path.join(CAND, c, "info.json")))
    ag, bq = float(r["a0"]), float(r["bulk_GPa"])
    ad = float(np.mean(np.diag(np.array(d["relaxed_cell_A"]))))
    q = qc.get(c, {})
    e = eos.get(c, {})
    B_el = q.get("B_el")
    complete = QC_N.get(c, 0) == 24
    if not complete or name in WITHDRAWN:
        B_el = None
    c11 = c12 = c44 = None
    if complete and name not in WITHDRAWN:
        c11, c12, c44 = q.get("C11"), q.get("C12"), q.get("C44")
    b_eos = e.get("B0") if e.get("qc") == "pass" else None
    rows.append({
        "candidate": name, "proto": r["proto"], "formula": r["formula"],
        "a_GNN": round(ag, 4), "a_DFT": round(ad, 4),
        "dA_pct": round(100 * (ad - ag) / ag, 2),
        "B_GNN": round(bq, 1),
        "B_DFT": round(B_el, 1) if B_el is not None else "",
        "dB_pct": (round(100 * (B_el - bq) / bq, 2)
                   if B_el is not None else ""),
        "C11": (c11 and round(c11, 1)) if c11 is not None else "",
        "C12": (c12 and round(c12, 1)) if c12 is not None else "",
        "C44": (c44 and round(c44, 1)) if c44 is not None else "",
        "Cprime_tet": TET.get(c, ""),
        "B_EOS": b_eos and round(b_eos, 1),
        "E_hull_GNN": round(float(r["e_hull"]), 4),
        "verdict": ("QC fail: magnetic branch inconsistent; withdrawn"
                    if name in WITHDRAWN
                    else VERDICT.get(name, born_verdict(c11, c12, c44))),
    })

cols = ["candidate", "proto", "formula", "a_GNN", "a_DFT", "dA_pct",
        "B_GNN", "B_DFT", "dB_pct", "C11", "C12", "C44", "Cprime_tet",
        "B_EOS", "E_hull_GNN", "verdict"]
with open(os.path.join(BASE, "master_dft_validation.csv"), "w",
          newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    wb = Workbook()
    ws = wb.active
    ws.title = "DFT validation"
    hdr = ["Candidate", "Prototype", "Formula", "a_GNN (A)", "a_DFT (A)",
           "Δa (%)", "B_GNN (GPa)", "B_DFT (GPa)", "ΔB (%)", "C11", "C12",
           "C44", "C'_tet", "B_EOS", "E_hull GNN", "Verdict"]
    ws.append(hdr)
    for c in ws[1]:
        c.font = Font(bold=True)
    for r in rows:
        ws.append([r["candidate"], r["proto"], r["formula"], r["a_GNN"],
                   r["a_DFT"], r["dA_pct"], r["B_GNN"], r["B_DFT"],
                   r["dB_pct"], r["C11"], r["C12"], r["C44"],
                   r["Cprime_tet"], r["B_EOS"], r["E_hull_GNN"],
                   r["verdict"]])
    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width = 14
    ws.column_dimensions["A"].width = 11
    ws.column_dimensions["P"].width = 34
    wb.save(os.path.join(BASE, "master_dft_validation.xlsx"))
    print("wrote xlsx")
except ImportError:
    print("openpyxl missing: csv only")
print("wrote master_dft_validation.csv")


def parity(x, y, labels, xlabel, ylabel, fname, ann=True):
    x = np.array(x, dtype=float)
    y = np.array([v if v != "" else np.nan for v in y], dtype=float)
    m = ~np.isnan(y)
    fig, a = plt.subplots(figsize=(7.5, 7))
    lo = min(x.min(), np.nanmin(y)) * 0.97
    hi = max(x.max(), np.nanmax(y)) * 1.03
    a.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="y = x")
    a.scatter(x[m], y[m], s=60, c="C0", zorder=3)
    if ann:
        for xi, yi, lb in zip(x[m], y[m], np.array(labels)[m]):
            e = 100 * (yi - xi) / xi
            a.annotate("%s\n%+.1f%%" % (lb, e), (xi, yi),
                       fontsize=7, xytext=(5, 5), textcoords="offset points")
    a.set_xlabel(xlabel)
    a.set_ylabel(ylabel)
    a.legend(fontsize=9)
    a.grid(alpha=0.3)
    a.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, fname), dpi=170)
    print("wrote", fname)


labs = [r["candidate"] for r in rows]
parity([r["a_GNN"] for r in rows], [r["a_DFT"] for r in rows], labs,
       "GNN a0 (A)", "DFT a0 (A)", "fig_parity_a0.png")
parity([r["B_GNN"] for r in rows], [r["B_DFT"] for r in rows], labs,
       "GNN B (GPa)", "DFT B (GPa)", "fig_parity_B.png")
print("DONE")
