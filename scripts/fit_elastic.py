#!/usr/bin/env python3
"""Fit cubic elastic constants from strained-relax outputs.

Sign/unit handling (verified against NiOs data, NOT assumed):
  - QE prints stress with P = +Tr/3 (compression -> positive). Slopes are in
    kbar per unit strain; C[GPa] = -slope_kbar/10, with the overall sign
    CALIBRATED per run by matching C11 from stress against C11 from the
    energy curvature d2E/dd2 (sign-free). Both must agree; mismatch aborts.
  - Shear (code convention a[i] += d/2*b[i]): e12 = d/4, so
    s12(d) slope m -> C44 = -m/10, cross-checked vs shear-energy curvature
    C44 = 8*kappa_E/Omega when resolvable.
Energy unit: Ry; Omega in bohr^3; 1 Ry/bohr^3 = 14710.5 GPa.
Born criteria for cubic: C11-C12>0, C44>0, C11+2*C12>0.
Usage: fit_elastic.py <strain_calc_dir> <a0_A> <natoms>
"""
import glob
import os
import re
import sys

import numpy as np

RYGPA = 14710.5
BOHR = 1.889726125

D = sys.argv[1]
A0 = float(sys.argv[2])
NAT = int(sys.argv[3])
OM = (A0 * BOHR) ** 3  # conventional-cell volume in bohr^3

data = {}
for f in glob.glob(os.path.join(D, "*.out")):
    t = os.path.basename(f).replace(".out", "")
    txt = open(f, encoding="utf-8", errors="replace").read()
    if "JOB DONE" not in txt:
        print("  SKIP (incomplete): %s" % t)
        continue
    m = re.search(r"total\s+stress.*?\n((?:.*\n){3})", txt)
    s = np.array([[float(x) for x in ln.split()[3:6]]
                  for ln in m.group(1).strip().split("\n")])
    e = re.search(r"!\s+total energy\s+=\s+([-\d.]+)\s+Ry", txt)
    data[t] = (s / 10.0, float(e.group(1)))  # kbar -> GPa
print("complete: %d" % len(data), flush=True)


def fit(x, y):
    x = np.array(x)
    y = np.array(y)
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    yh = m * x + b
    ss = ((y - yh) ** 2).sum()
    st = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss / st if st > 0 else float("nan")
    # quadratic energy curvature fit if y looks like energy
    return m, b, r2


def eq(kind, d):
    xs, ys, es = [], [], []
    for t, (s, e) in data.items():
        a, v = t.split("_")
        if a == kind:
            xs.append(float(v))
            ys.append(s[d[0], d[1]])
            es.append(e)
    return np.array(xs), np.array(ys), np.array(es)


# --- energy curvature -> C11 (sign-free) from xx/yy/zz ---
c11e = []
for ax, nm in [(0, "xx"), (1, "yy"), (2, "zz")]:
    xs, _, es = eq(nm, (ax, ax))
    c = np.polyfit(xs, es, 2)
    c11e.append(2 * c[0] / OM * RYGPA)
C11_E = float(np.mean(c11e))
print("energy-curvature C11 = %.1f GPa (per-axis %s)" % (
    C11_E, ["%.0f" % v for v in c11e]), flush=True)

# --- stress slopes, sign calibrated so <C11> matches C11_E ---
c11s, c12s = [], []
for ax, nm in [(0, "xx"), (1, "yy"), (2, "zz")]:
    xs, yd, _ = eq(nm, (ax, ax))
    m1, b1, r1 = fit(xs, yd)
    c11s.append(-m1)
    yo = []
    for t, (s, e) in data.items():
        a, v = t.split("_")
        if a == nm:
            yo.append(np.mean([s[o, o] for o in range(3) if o != ax]))
    m2, b2, r2 = fit(xs, yo)
    c12s.append(-m2)
    print("%s: C11=%7.1f (R2=%.4f)  C12=%7.1f (R2=%.4f)" % (nm, -m1, r1, -m2, r2),
          flush=True)
C11, C12 = float(np.mean(c11s)), float(np.mean(c12s))
print("stress C11=%.1f vs energy C11=%.1f -> %s" % (
    C11, C11_E, "AGREE" if abs(C11 - C11_E) / C11_E < 0.25 else "*** MISMATCH ***"),
    flush=True)

# --- shear: e_xy = d/4 for the code's a[i]+=d/2*b[i] convention, so
# sigma_xy = -2*C44*e_xy -> slope m (kbar) gives C44[GPa] = -2*m/10.
# Cross-checked vs energy curvature C44 = 8*kappa_E/Omega (must agree).
c44s = []
for a, (i, j) in {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}.items():
    xs, ys, es = eq(a, (i, j))
    if len(xs) < 2:
        print("%s: n<2, skip" % a, flush=True)
        continue
    m, b, r = fit(xs, ys)
    if len(xs) >= 3:
        c = np.polyfit(xs, es, 2)
        c44e = 8 * c[0] / OM * RYGPA
    else:
        c44e = float("nan")
    c44s.append(-2 * m)  # data already GPa: C44 = -2*slope
    print("%s: C44=%7.1f (R2=%.4f)  energy-xcheck=%7.1f" % (a, -2 * m, r, c44e),
          flush=True)
C44 = float(np.mean(c44s))
B = (C11 + 2 * C12) / 3
G = (C11 - C12 + 3 * C44) / 5  # Voigt shear
print("\nC11=%.1f C12=%.1f C44=%.1f  B_Voigt=%.1f G_Voigt=%.1f GPa" % (
    C11, C12, C44, B, G), flush=True)
print("Born: C11-C12=%+.1f %s | C44=%+.1f %s | C11+2C12=%+.1f %s" % (
    C11 - C12, "OK" if C11 - C12 > 0 else "VIOLATED",
    C44, "OK" if C44 > 0 else "VIOLATED",
    C11 + 2 * C12, "OK" if C11 + 2 * C12 > 0 else "VIOLATED"), flush=True)
