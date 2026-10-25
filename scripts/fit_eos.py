#!/usr/bin/env python3
"""Birch-Murnaghan EOS fits from candidates/<C>/eos/*.out.

E(V) 3rd-order Birch-Murnaghan:
  E = E0 + (9*V0*B0/16) * ( (eta^2-1)^3 * Bp + (eta^2-1)^2 * (6-4*eta^2) )
  with eta = (V0/V)^(1/3). V in bohr^3, E in Ry -> B0 in GPa via RYGPA.
Volumes from known isotropic scales s in [0.97..1.03] times the relaxed a0
(from info.json). Requires >=5 complete points; else reports insufficiency.
QC gate (a fit is quotable only if ALL pass):
  rms_meV <= 0.2,  |B0 - B_el|/B_el <= 0.02,  |a0_EOS/a_relax - 1| <= 0.005
where B_el comes from final_qc.json. Failing fits keep their numbers in the
JSON for forensics but carry "qc": "fail" and are excluded downstream.
Output: per-compound JSON + comparison table vs elastic B and relax cell.
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy.optimize import curve_fit

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAND = os.path.join(BASE, "candidates")
RYGPA = 14710.5
BOHR = 1.889726125
S = [0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03]

COMPOUNDS = ["ReOs", "TcOs", "POs2", "ReRh", "BIr2",
             "ReIr", "NiOs", "ReB", "TcRe", "RuO"]

QC = json.load(open(os.path.join(BASE, "final_qc.json")))


def bm(V, E0, V0, B0_GPa, Bp):
    B = B0_GPa / RYGPA
    eta = (V0 / V) ** (1.0 / 3.0)
    t = eta * eta - 1.0
    return E0 + (9.0 * V0 * B / 16.0) * (t ** 3 * Bp + t * t * (6.0 - 4.0 * eta * eta))


def load(comp):
    info = json.load(open(os.path.join(CAND, comp, "info.json")))
    a = float(np.mean(np.diag(np.array(info["relaxed_cell_A"]))))
    pts = []
    for i in range(7):
        f = os.path.join(CAND, comp, "eos", "eos_%d.out" % i)
        if not os.path.exists(f):
            continue
        txt = open(f, encoding="utf-8", errors="replace").read()
        if "JOB DONE" not in txt:
            continue
        e = float(re.search(r"!\s+total energy\s+=\s+([-\d.]+)\s+Ry", txt).group(1))
        V = (a * S[i] * BOHR) ** 3
        pm = re.findall(r"P=\s+([-\d.]+)", txt)
        mm = re.findall(r"total magnetization\s*=\s*([-\d.]+)", txt)
        pts.append((V, e, float(pm[-1]) if pm else None,
                    float(mm[-1]) if mm else None))
    return a, pts


print("%-7s %4s %9s %9s %8s %6s %9s  %-8s %s" % (
    "comp", "n", "a0_EOS", "V0", "B0", "Bp", "rms_meV", "P@closest", "qc"))
out = {}
for comp in COMPOUNDS:
    a, pts = load(comp)
    if len(pts) < 5:
        print("%-7s %4d INSUFFICIENT DATA" % (comp, len(pts)))
        out[comp] = {"n": len(pts), "fit": False}
        continue
    V = np.array([p[0] for p in pts])
    E = np.array([p[1] for p in pts])
    p0 = [E.min(), V[np.argmin(E)], 300.0, 4.0]
    try:
        po, _ = curve_fit(bm, V, E, p0=p0, maxfev=20000)
    except Exception as ex:
        print("%-7s %4d FIT FAILED: %s" % (comp, len(pts), str(ex)[:80]))
        out[comp] = {"n": len(pts), "fit": False}
        continue
    E0, V0, B0, Bp = [float(v) for v in po]
    pred = bm(V, *po)
    rms = float(np.sqrt(((E - pred) ** 2).mean()) * 13605.7)  # Ry->meV
    a0 = (V0) ** (1.0 / 3.0) / BOHR
    i = int(np.argmin(abs(V - V0)))
    reasons = []
    if rms > 0.2:
        reasons.append("rms=%.3f meV>0.2" % rms)
    B_el = QC.get(comp, {}).get("B_el")
    if B_el is not None and abs(B0 - B_el) / B_el > 0.02:
        reasons.append("|B0-B_el|/B_el=%.1f%%>2%%" %
                       (100 * abs(B0 - B_el) / B_el))
    if abs(a0 / a - 1) > 0.005:
        reasons.append("|a0/a_relax-1|=%.2f%%>0.5%%" %
                       (100 * abs(a0 / a - 1)))
    # magnetic-branch consistency vs the reference scf
    ref = QC.get(comp, {}).get("mag_ref")
    mags = [p[3] for p in pts if p[3] is not None]
    mag_dev = max(abs(m - ref) for m in mags) if (ref is not None and mags) else None
    if mag_dev is not None and mag_dev > 0.5:
        reasons.append("mag_branch dev=%.2f uB>0.5" % mag_dev)
    out[comp] = {"n": len(pts), "fit": True, "E0": E0, "V0": V0, "B0": B0,
                 "Bp": Bp, "rms_meV": rms, "a0_EOS": a0,
                 "P_at_closest": pts[i][2], "a_relax": a,
                 "mag_dev": mag_dev,
                 "qc": "fail" if reasons else "pass",
                 "qc_reasons": reasons}
    print("%-7s %4d %9.4f %9.2f %8.1f %6.2f %9.3f  %-8s %s" % (
        comp, len(pts), a0, V0, B0, Bp, rms, pts[i][2],
        "QC-FAIL " + ";".join(reasons) if reasons else "qc-pass"))
json.dump(out, open(os.path.join(BASE, "eos_fits.json"), "w"), indent=1)
print("wrote eos_fits.json")
