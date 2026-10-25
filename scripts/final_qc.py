#!/usr/bin/env python3
"""FINAL DFT QC assembly from local candidates/ data only (no new calcs).

Reads: candidates/<C>/info.json, strain_calc/*.out, eos via eos_fits.json,
tet data (remote values embedded after verification).
Writes: final_qc.json + printed point-by-point tables.
"""
import glob
import json
import os
import re

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAND = os.path.join(BASE, "candidates")
RYGPA = 14710.5
BOHR = 1.889726125

COMPOUNDS = ["ReOs", "TcOs", "POs2", "ReRh", "BIr2",
             "ReIr", "NiOs", "ReB", "TcRe", "RuO"]
# verified tet C' (quadratic fits, ΔE = 6ΩC'd^2); NiOs from local tet_test
TET = {"NiOs": -23.5, "TcOs": -151.8, "ReRh": -197.1,
       "ReIr": -268.0, "TcRe": -11.0}


def load_strain(comp):
    d = os.path.join(CAND, comp, "strain_calc")
    data = {}
    for f in glob.glob(os.path.join(d, "*.out")):
        t = os.path.basename(f).replace(".out", "")
        txt = open(f, encoding="utf-8", errors="replace").read()
        if "JOB DONE" not in txt:
            continue
        m = re.search(r"total\s+stress.*?\n((?:.*\n){3})", txt)
        s = np.array([[float(x) for x in ln.split()[3:6]]
                      for ln in m.group(1).strip().split("\n")])
        e = float(re.search(r"!\s+total energy\s+=\s+([-\d.]+)\s+Ry",
                            txt).group(1))
        mm = re.findall(r"total magnetization\s*=\s*([-\d.]+)", txt)
        mag = float(mm[-1]) if mm else None
        data[t] = (s / 10.0, e, mag)
    return data


def fit(x, y):
    x = np.array(x)
    y = np.array(y)
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    yh = m * x + b
    r = yh - y
    ss = (r ** 2).sum()
    st = ((y - y.mean()) ** 2).sum()
    return m, b, 1 - ss / st if st > 0 else float("nan"), float(np.abs(r).max())


rep = {}
for comp in COMPOUNDS:
    info = json.load(open(os.path.join(CAND, comp, "info.json")))
    a0 = float(np.mean(np.diag(np.array(info["relaxed_cell_A"]))))
    data = load_strain(comp)
    n = len(data)
    r = {"n_strain": n, "a_relax": a0,
         "mag": info["reference_scf"].get("mag_tot"),
         "P_scf": info["reference_scf"].get("P_kbar")}
    if n == 0:
        rep[comp] = r
        continue
    # per-axis normal fits + transverse
    c11s, c12s, r2a, rmsa, maxa = [], [], [], [], []
    for ax, nm in [(0, "xx"), (1, "yy"), (2, "zz")]:
        ts = sorted([t for t in data if t.split("_")[0] == nm],
                    key=lambda t: float(t.split("_")[1]))
        if len(ts) < 2:
            continue
        xs = [float(t.split("_")[1]) for t in ts]
        yd = [data[t][0][ax, ax] for t in ts]
        yo = [np.mean([data[t][0][o, o] for o in range(3) if o != ax])
              for t in ts]
        m1, b1, r1, mx1 = fit(xs, yd)
        m2, b2, r2, mx2 = fit(xs, yo)
        c11s.append(-m1)
        c12s.append(-m2)
        r2a += [r1, r2]
        rmsa.append(float(np.sqrt(((np.array(yd) - (m1 * np.array(xs) + b1)) ** 2).mean())))
        maxa.append(mx1)
    sh = []
    for a, (i, j) in {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}.items():
        ts = sorted([t for t in data if t.split("_")[0] == a],
                    key=lambda t: float(t.split("_")[1]))
        if len(ts) < 2:
            continue
        xs = [float(t.split("_")[1]) for t in ts]
        ys = [data[t][0][i, j] for t in ts]
        m, b, r2, mx = fit(xs, ys)
        sh.append(-2 * m)
        r2a.append(r2)
        rmsa.append(float(np.sqrt(((np.array(ys) - (m * np.array(xs) + b)) ** 2).mean())))
        maxa.append(mx)
    r["C11"] = float(np.mean(c11s)) if c11s else None
    r["C12"] = float(np.mean(c12s)) if c12s else None
    r["C44"] = float(np.mean(sh)) if sh else None
    r["R2min"] = float(min(r2a)) if r2a else None
    r["RMSmax"] = float(max(rmsa)) if rmsa else None
    r["MAXRES"] = float(max(maxa)) if maxa else None
    if r["C11"] is not None and r["C12"] is not None:
        r["B_el"] = (r["C11"] + 2 * r["C12"]) / 3
    # magnetic-branch consistency: every strain SCF must stay on the same spin
    # branch as the reference scf, else the stress slopes mix magnetic states.
    mags = [v[2] for v in data.values() if v[2] is not None]
    ref = info["reference_scf"].get("mag_tot")
    r["mag_ref"] = ref
    r["mag_strain_min"] = float(min(mags)) if mags else None
    r["mag_strain_max"] = float(max(mags)) if mags else None
    if ref is not None and mags:
        dev = max(abs(m - ref) for m in mags)
        r["mag_branch_dev"] = float(dev)
        r["mag_branch_ok"] = bool(dev <= 0.5)
    else:
        r["mag_branch_dev"] = None
        r["mag_branch_ok"] = None
    rep[comp] = r

json.dump(rep, open(os.path.join(BASE, "final_qc.json"), "w"), indent=1)
for comp in COMPOUNDS:
    r = rep[comp]
    print("%-7s n=%-3d C11=%8s C12=%8s C44=%8s B_el=%8s R2min=%s RMSmax=%s P=%s mag=%s" % (
        comp, r["n_strain"],
        ("%.1f" % r["C11"]) if r.get("C11") is not None else "n/a",
        ("%.1f" % r["C12"]) if r.get("C12") is not None else "n/a",
        ("%.1f" % r["C44"]) if r.get("C44") is not None else "n/a",
        ("%.1f" % r["B_el"]) if r.get("B_el") is not None else "n/a",
        ("%.4f" % r["R2min"]) if r.get("R2min") is not None else "n/a",
        ("%.2f" % r["RMSmax"]) if r.get("RMSmax") is not None else "n/a",
        r["P_scf"], r["mag"]))
print("wrote final_qc.json")
