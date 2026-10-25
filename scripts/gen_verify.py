#!/usr/bin/env python3
"""Generate tetragonal + EOS verification inputs from candidates/*/info.json.

Tetragonal scan (for C11-C12 check): volume-conserving tetragonal distortion
  a' = a(1+d), b' = a(1+d), c' = a/(1+d)^2   (V exact to all orders)
  d in {-0.02,-0.01,0,0.01,0.02}, scf tight 1e-8, same k/mags/pseudos.
  Energy: ΔE = 3*Omega*(C11-C12)*d^2 + O(d^3)  [cubic, exact V conservation]
  -> sign of curvature = sign of (C11-C12). No stress conventions involved.

EOS scan: isotropic cell scaling s in {0.97,0.98,...,1.03} mapped to
  volumes ~0.91-1.09; 7 scf points; Birch-Murnaghan fit -> B0, V0.
  B_elastic = (C11+2*C12)/3 must match B0 within fit/error bars.

Writes dft_verify/<Compound>/tet_<d>.in and eos_<i>.in from the RELAXED
cell/positions in info.json (NOT re-idealized: keeps any relaxed distortion).
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAND = os.path.join(BASE, "candidates")
OUT = os.path.join(BASE, "dft_verify")
os.makedirs(OUT, exist_ok=True)

TET_D = [-0.02, -0.01, 0.0, 0.01, 0.02]
EOS_S = [0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03]

HEAD = """&CONTROL
    calculation  = 'scf'
    pseudo_dir   = '/scratch/sunraj/dft_pseudo/'
    outdir       = '/scratch/sunraj/dft_verify/%s/tmp/'
    prefix       = '%s'
    tstress      = .true.
    tprnfor      = .true.
    disk_io      = 'low'
    wf_collect   = .false.
/
&SYSTEM
    ibrav        = 0
    nat          = %d
    ntyp         = %d
    ecutwfc      = 100
    ecutrho      = 1000
    occupations  = 'smearing'
    smearing     = 'mv'
    degauss      = 0.02
    nspin        = 2
%s
/
&ELECTRONS
    conv_thr         = 1.0D-8
    electron_maxstep = 300
    mixing_beta      = 0.2
    mixing_ndim      = 8
/
ATOMIC_SPECIES
%sATOMIC_POSITIONS (crystal)
%sK_POINTS (automatic)
%s %s %s 0 0 0
"""


def mag_block(info):
    lines = []
    for i, s in enumerate(info["species"], 1):
        m = info["starting_mag"].get(s["symbol"], 0.0)
        lines.append("    starting_magnetization(%d) = %s" % (i, m))
    return "\n".join(lines)


def species_block(info):
    return "".join("%s  %.2f  %s\n" % (s["symbol"], s["mass"], s["pp"])
                   for s in info["species"])


def pos_block(info):
    return "".join("%s %.10f %.10f %.10f\n" % tuple(p) for p in
                   info["relaxed_positions_crystal"])


def run(comp, kinds):
    info = json.load(open(os.path.join(CAND, comp, "info.json")))
    nat = len(info["relaxed_positions_crystal"])
    ntyp = len(info["species"])
    k = info["k_mesh"]
    d = os.path.join(OUT, comp)
    os.makedirs(d, exist_ok=True)
    a = float(__import__("numpy").mean(
        __import__("numpy").diag(__import__("numpy").array(
            info["relaxed_cell_A"]))))
    n = 0
    if "tet" in kinds:
        for dd in TET_D:
            tag = "tet_%+.3f" % dd
            cell = ("CELL_PARAMETERS (angstrom)\n"
                    "  %.10f 0 0\n  0 %.10f 0\n  0 0 %.10f\n" % (
                        a * (1 + dd), a * (1 + dd), a / (1 + dd) ** 2))
            open(os.path.join(d, tag + ".in"), "w").write(
                HEAD % (comp, comp.replace("-", ""), nat, ntyp,
                        mag_block(info), species_block(info),
                        pos_block(info), k, k, k) + cell)
            n += 1
    if "eos" in kinds:
        for i, s in enumerate(EOS_S):
            tag = "eos_%d" % i
            cell = ("CELL_PARAMETERS (angstrom)\n"
                    "  %.10f 0 0\n  0 %.10f 0\n  0 0 %.10f\n" % (
                        a * s, a * s, a * s))
            # EOS keeps fractional positions fixed at relaxed values
            open(os.path.join(d, tag + ".in"), "w").write(
                HEAD % (comp, comp.replace("-", ""), nat, ntyp,
                        mag_block(info), species_block(info),
                        pos_block(info), k, k, k) + cell)
            n += 1
    print("wrote %d inputs for %s (a=%.5f)" % (n, comp, a), flush=True)


if __name__ == "__main__":
    # usage: gen_verify.py [tet:E1,E2] [eos:E1,E2,...]  (default: all)
    tet = ["TcOs", "ReRh", "ReIr", "TcRe"]
    eos = ["ReOs", "TcOs", "POs2", "ReRh", "BIr2",
           "ReIr", "NiOs", "ReB", "TcRe", "RuO"]
    for c in tet:
        run(c, ["tet"])
    for c in eos:
        run(c, ["eos"])
    print("DONE")
