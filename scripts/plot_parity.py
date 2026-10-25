#!/usr/bin/env python3
"""Parity figures: GNN vs DFT lattice parameter and bulk modulus.

Reads master_dft_validation.csv (written by dft_step1/make_master.py) and
produces:
  fig_parity_a0.png  - DFT a0 vs GNN a0 with y=x and per-point % error
  fig_parity_B.png   - DFT B  vs GNN B  with y=x and per-point % error
Rows with missing DFT values are skipped (POs2/BIr2 until their tails land;
rerun this script after regenerating the master CSV).
Usage (from scripts/gnn):  python dft_step1/plot_parity.py
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
import anthropic_style as S  # noqa: E402
S.apply()


def load():
    with open(os.path.join(BASE, "master_dft_validation.csv")) as f:
        return list(csv.DictReader(f))


def parity(xkey, ykey, xlabel, ylabel, title, unit, fname, band=5.0):
    xs, ys, labs = [], [], []
    for r in load():
        if r[xkey] in ("", None) or r[ykey] in ("", None):
            continue
        xs.append(float(r[xkey]))
        ys.append(float(r[ykey]))
        labs.append(r["candidate"])
    xs, ys = np.array(xs), np.array(ys)
    err = 100.0 * (ys - xs) / xs
    mape = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean((ys - xs) ** 2))
    r2 = 1 - ((ys - xs) ** 2).sum() / ((xs - xs.mean()) ** 2).sum()

    fig, a = plt.subplots(figsize=(7.4, 7.6), constrained_layout=True)
    lo = min(xs.min(), ys.min())
    hi = max(xs.max(), ys.max())
    pad = 0.06 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    g = np.linspace(lo, hi, 100)

    a.fill_between(g, g * (1 - band / 100), g * (1 + band / 100),
                   color=S.CLAY, alpha=0.10, lw=0, label="±%.0f%% band" % band)
    a.plot(g, g, color=S.INK, lw=1.2, ls=(0, (5, 3)), label="y = x", zorder=2)
    a.scatter(xs, ys, s=70, c=S.CLAY, edgecolors="white", linewidths=1.2,
              zorder=3)
    for xi, yi, lb, e in zip(xs, ys, labs, err):
        a.annotate(lb, (xi, yi), xytext=(7, 7), textcoords="offset points",
                   fontsize=8.5, fontweight="semibold", color=S.INK)
        a.annotate("%+.1f%%" % e, (xi, yi), xytext=(7, -3),
                   textcoords="offset points", fontsize=7.5, color=S.MUTED)

    a.set_xlim(lo, hi)
    a.set_ylim(lo, hi)
    a.set_aspect("equal", adjustable="box")
    a.set_xlabel(xlabel)
    a.set_ylabel(ylabel)
    a.grid(True)
    a.tick_params(length=3, width=0.8, color=S.RULE)
    a.legend(loc="upper left", handlelength=1.8)
    a.text(0.97, 0.04,
           "n = %d\nMAPE = %.1f%%\nRMSE = %.2f %s\n$R^2$ = %.3f"
           % (len(xs), mape, rmse, unit, r2),
           transform=a.transAxes, ha="right", va="bottom", fontsize=8.5,
           color=S.INK, linespacing=1.5,
           bbox=dict(boxstyle="round,pad=0.5", fc=S.BG, ec=S.RULE, lw=0.8))
    a.set_title("")
    fig.suptitle(title, x=0.02, ha="left", fontfamily=S.SERIF, fontsize=15,
                 fontweight="normal")
    fig.savefig(os.path.join(BASE, fname))
    plt.close(fig)
    print("wrote %s (%d points)" % (fname, len(xs)), flush=True)


if __name__ == "__main__":
    parity("a_GNN", "a_DFT", "GNN $a_0$ (Å)", "DFT $a_0$ (Å)",
           "Lattice parameter: GNN vs DFT", "Å", "fig_parity_a0.png", band=2.0)
    parity("B_GNN", "B_DFT", "GNN $B$ (GPa)", "DFT $B$ (GPa)",
           "Bulk modulus: GNN vs DFT", "GPa", "fig_parity_B.png", band=10.0)
    print("DONE")
