#!/usr/bin/env python3
"""Fit + E-vs-strain plots for every candidate in candidates/.

Per compound <C> (using candidates/<C>/info.json for a0, strain_calc/*.out):
  - fits C11/C12/C44/B/G with the energy-calibrated sign convention
  - writes candidates/<C>/fig_elastic.png:
      top row: E(d) parabolas for xx/yy/zz (overlaid) and for shears
      bottom row: stress(d) slopes with linear fits
  - prints one-line summary table.
Run from scripts/gnn. Data snapshot: partial strain sets give preliminary fits.
"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAND = os.path.join(BASE, "candidates")
RYGPA = 14710.5
BOHR = 1.889726125

COMPOUNDS = ["ReOs", "TcOs", "POs2", "ReRh", "BIr2",
             "ReIr", "NiOs", "ReB", "TcRe", "RuO"]


def load(comp):
    d = os.path.join(CAND, comp, "strain_calc")
    info = json.load(open(os.path.join(CAND, comp, "info.json")))
    a0 = float(np.mean(np.diag(np.array(info["relaxed_cell_A"]))))
    data = {}
    for f in glob.glob(os.path.join(d, "*.out")):
        t = os.path.basename(f).replace(".out", "")
        txt = open(f, encoding="utf-8", errors="replace").read()
        if "JOB DONE" not in txt:
            continue
        m = re.search(r"total\s+stress.*?\n((?:.*\n){3})", txt)
        s = np.array([[float(x) for x in ln.split()[3:6]]
                      for ln in m.group(1).strip().split("\n")])
        e = re.search(r"!\s+total energy\s+=\s+([-\d.]+)\s+Ry", txt)
        data[t] = (s / 10.0, float(e.group(1)))
    return info, a0, data


def fit(x, y):
    x = np.array(x)
    y = np.array(y)
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    yh = m * x + b
    ss = ((y - yh) ** 2).sum()
    st = ((y - y.mean()) ** 2).sum()
    return m, b, 1 - ss / st if st > 0 else float("nan")


print("%-7s %3s %7s %7s %7s %8s %8s  %s" % (
    "comp", "n", "C11", "C12", "C44", "B", "G", "born"))
for comp in COMPOUNDS:
    info, a0, data = load(comp)
    n = len(data)
    om = (a0 * BOHR) ** 3
    # energy-curvature C11 (sign-free calibration)
    c11e = []
    for ax, nm in [(0, "xx"), (1, "yy"), (2, "zz")]:
        xs = [float(t.split("_")[1]) for t in data if t.split("_")[0] == nm]
        es = [data[t][1] for t in data if t.split("_")[0] == nm]
        if len(xs) >= 3:
            c = np.polyfit(xs, es, 2)
            c11e.append(2 * c[0] / om * RYGPA)
    C11E = float(np.mean(c11e)) if c11e else float("nan")
    # stress slopes
    c11s, c12s = [], []
    for ax, nm in [(0, "xx"), (1, "yy"), (2, "zz")]:
        xs = [float(t.split("_")[1]) for t in data if t.split("_")[0] == nm]
        yd = [data[t][0][ax, ax] for t in data if t.split("_")[0] == nm]
        yo = [np.mean([data[t][0][o, o] for o in range(3) if o != ax])
              for t in data if t.split("_")[0] == nm]
        if len(xs) >= 2:
            m1, _, _ = fit(xs, yd)
            m2, _, _ = fit(xs, yo)
            c11s.append(-m1)
            c12s.append(-m2)
    C11 = float(np.mean(c11s)) if c11s else float("nan")
    C12 = float(np.mean(c12s)) if c12s else float("nan")
    c44s = []
    for a, (i, j) in {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}.items():
        xs = [float(t.split("_")[1]) for t in data if t.split("_")[0] == a]
        ys = [data[t][0][i, j] for t in data if t.split("_")[0] == a]
        if len(xs) >= 2:
            m, _, _ = fit(xs, ys)
            c44s.append(-2 * m)  # data already GPa: C44 = -2*slope
    C44 = float(np.mean(c44s)) if c44s else float("nan")
    B = (C11 + 2 * C12) / 3
    G = (C11 - C12 + 3 * C44) / 5
    import math
    b1 = "OK" if C11 - C12 > 0 else ("n/a" if any(math.isnan(v) for v in (C11, C12)) else "VIOLATED")
    b2 = "OK" if C44 > 0 else ("n/a" if math.isnan(C44) else "VIOLATED")
    born = "C11-C12 %s, C44 %s" % (b1, b2)
    cal = ("E-cal %.0f" % C11E) if c11e else "E-cal n/a"
    print("%-7s %3d %7.1f %7.1f %7.1f %8.1f %8.1f  %s [%s]" % (
        comp, n, C11, C12, C44, B, G, born, cal))

    # ---- figure ----
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    cols = {"xx": "C0", "yy": "C1", "zz": "C2",
            "xy": "C3", "xz": "C4", "yz": "C5"}
    AX = {"xx": 0, "yy": 1, "zz": 2}
    SH = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    for nm in ["xx", "yy", "zz"]:
        pts = sorted(((float(t.split("_")[1]), data[t][1]) for t in data
                      if t.split("_")[0] == nm), key=lambda p: p[0])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        es = [p[1] for p in pts]
        ax[0][0].plot(xs, es, "o-", color=cols[nm], label=nm)
        ss = [data[t][0][AX[nm], AX[nm]] for t in
              sorted((t for t in data if t.split("_")[0] == nm),
                     key=lambda t: float(t.split("_")[1]))]
        m, b, _ = fit(xs, ss)
        xx = np.linspace(min(xs), max(xs), 50)
        ax[1][0].plot(xs, ss, "o", color=cols[nm])
        ax[1][0].plot(xx, m * xx + b, "-", color=cols[nm],
                      label="%s slope %.0f" % (nm, m))
    for nm, (i, j) in SH.items():
        pts = sorted(((float(t.split("_")[1]), data[t][1]) for t in data
                      if t.split("_")[0] == nm), key=lambda p: p[0])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        es = [p[1] for p in pts]
        ax[0][1].plot(xs, es, "s-", color=cols[nm], label=nm)
        ss = [data[t][0][i, j] for t in
              sorted((t for t in data if t.split("_")[0] == nm),
                     key=lambda t: float(t.split("_")[1]))]
        m, b, _ = fit(xs, ss)
        xx = np.linspace(min(xs), max(xs), 50)
        ax[1][1].plot(xs, ss, "s", color=cols[nm])
        ax[1][1].plot(xx, m * xx + b, "-", color=cols[nm],
                      label="%s slope %.0f" % (nm, m))
    ax[0][0].set_title("%s E vs normal strain (n=%d)" % (comp, n))
    ax[0][0].set_ylabel("E (Ry)")
    ax[0][0].legend(fontsize=7)
    ax[0][0].grid(alpha=0.3)
    ax[0][1].set_title("%s E vs shear strain" % comp)
    ax[0][1].set_ylabel("E (Ry)")
    ax[0][1].legend(fontsize=7)
    ax[0][1].grid(alpha=0.3)
    ax[1][0].set_title("diagonal stress vs normal strain (GPa)")
    ax[1][0].set_xlabel("strain")
    ax[1][0].set_ylabel("stress (GPa)")
    ax[1][0].legend(fontsize=7)
    ax[1][0].grid(alpha=0.3)
    ax[1][1].set_title("shear stress vs shear strain (GPa)")
    ax[1][1].set_xlabel("strain")
    ax[1][1].legend(fontsize=7)
    ax[1][1].grid(alpha=0.3)
    fig.suptitle("%s elastic response: a0=%.4f A (DFT relaxed)" % (comp, a0))
    fig.tight_layout()
    fig.savefig(os.path.join(CAND, comp, "fig_elastic.png"), dpi=150)
    print("  wrote candidates/%s/fig_elastic.png" % comp)
print("DONE")
