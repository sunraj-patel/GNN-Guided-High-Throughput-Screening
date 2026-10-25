#!/usr/bin/env python3
"""Generate elastic-strain inputs from relaxed structures (remote script).

Usage: gen_elastic.py <CompoundDir> <relax_output_name>
Reads relaxed cell + positions (LAST in file), species/mags/k/ecut from the
original vcrelax.in, writes strain_calc/<tag>.in for xx/yy/zz/xy/xz/yz x
+-0.005/+-0.01 (24 inputs, calculation='relax' at fixed strained cell,
conv_thr 1e-8, tstress/tprnfor). Mirrors the group's strain conventions.
NOTE: forc_conv_thr is rejected by this QE 7.3.1 build's parser (tested);
default force criterion applies (positions are symmetry-fixed anyway).
"""
import os
import re
import sys

D = sys.argv[1]
SRC = sys.argv[2] if len(sys.argv) > 2 else "vcrelax.out"
C = os.path.basename(D.rstrip("/"))
QE = "/apps/software/q-e-qe-7.3.1/bin/pw.x"

txt = open(os.path.join(D, SRC), encoding="utf-8", errors="replace").read()
cells = re.findall(r"CELL_PARAMETERS \(angstrom\)\s*\n((?:.*\n){3})", txt)
poss = re.findall(
    r"ATOMIC_POSITIONS \(crystal\)\s*\n((?:\s*\S+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\n)+)",
    txt)
assert cells and poss, "no geometry in %s" % SRC
cell = [[float(x) for x in row.split()] for row in cells[-1].strip().split("\n")]
pos = []
for ln in poss[-1].strip().split("\n"):
    p = ln.split()
    pos.append((p[0], float(p[1]), float(p[2]), float(p[3])))
nat = len(pos)
order = list(dict.fromkeys(p[0] for p in pos))
ntyp = len(order)

# read mag/species/k/ecut from the input that produced the relax source, so
# the magnetic branch matches the relaxed structure (e.g. vcrelax_mag.out ->
# vcrelax_mag.in). Falls back to vcrelax.in.
src_in = SRC.replace(".out", ".in")
if not os.path.exists(os.path.join(D, src_in)):
    src_in = "vcrelax.in"
old = open(os.path.join(D, src_in)).read()
k = re.search(r"K_POINTS \(automatic\)\s*\n(\d+)", old).group(1)
spm = re.search(r"ATOMIC_SPECIES\s*\n((?:.*\n)+?)ATOMIC_POSITIONS", old)
species = spm.group(1)
mag = "\n".join(l for l in old.split("\n") if "starting_magnetization" in l)
ecut = re.search(r"ecutwfc\s*=\s*(\d+)", old).group(1)
ecrho = re.search(r"ecutrho\s*=\s*(\d+)", old).group(1)

STRAINS = [-0.010, -0.005, 0.005, 0.010]
TYPES = ["xx", "yy", "zz", "xy", "xz", "yz"]


def strained_cell(t, d):
    a, b, c = [list(v) for v in cell]
    if t == "xx":
        a = [(1 + d) * x for x in a]
    elif t == "yy":
        b = [(1 + d) * x for x in b]
    elif t == "zz":
        c = [(1 + d) * x for x in c]
    elif t == "xy":
        a = [a[i] + d / 2 * b[i] for i in range(3)]
    elif t == "xz":
        a = [a[i] + d / 2 * c[i] for i in range(3)]
    elif t == "yz":
        b = [b[i] + d / 2 * c[i] for i in range(3)]
    return [a, b, c]


def cell_str(cc):
    return "\n".join("  %20.14f  %20.14f  %20.14f" % tuple(v) for v in cc)


def pos_str(pp):
    return "\n".join("%s   %.10f  %.10f  %.10f" % p for p in pp)


sdir = os.path.join(D, "strain_calc")
os.makedirs(sdir, exist_ok=True)
# outdir must be the ABSOLUTE remote path (jobs run on the cluster), not the
# local staging dir. Remote layout: <REMOTE_ROOT>/<Compound>/strain_calc/.
REMOTE_ROOT = os.environ.get("REMOTE_ROOT", "/scratch/sunraj/dft_prod")
remote_sdir = os.path.join(REMOTE_ROOT, C, "strain_calc")
tags = []
for t in TYPES:
    for d in STRAINS:
        tag = "%s_%s" % (t, d)
        tags.append(tag)
        cc = strained_cell(t, d)
        inp = """&CONTROL
    calculation  = 'relax'
    pseudo_dir   = '/scratch/sunraj/dft_pseudo/'
    outdir       = '%s/tmp_%s/'
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
    ecutwfc      = %s
    ecutrho      = %s
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
&IONS
    ion_dynamics = 'bfgs'
/
ATOMIC_SPECIES
%sATOMIC_POSITIONS (crystal)
%s
K_POINTS (automatic)
%s %s %s 0 0 0
CELL_PARAMETERS (angstrom)
%s
""" % (remote_sdir, tag, C.replace("-", ""), nat, ntyp, ecut, ecrho, mag,
       species, pos_str(pos), k, k, k, cell_str(cc))
        open(os.path.join(sdir, tag + ".in"), "w").write(inp)
print("wrote %d strain inputs for %s (nat=%d)" % (len(tags), C, nat), flush=True)
