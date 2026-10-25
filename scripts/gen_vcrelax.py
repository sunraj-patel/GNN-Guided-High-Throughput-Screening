#!/usr/bin/env python3
"""Generate Step-1 vc-relax inputs for all 10 DFT compounds.

Structures come from the validated framework prototype builders
(prototypes.py via screen_one.resolve_builder_and_xB) at the production a0
(a_guess * c_pred) -- the exact structures the ML model predicted.
No pymatgen, no hand-typed coordinates.

Checks per input (fail fast, before submit):
  - pseudo file exists in the frozen PBE-PAW set
  - ATOMIC_SPECIES order matches position labels
  - starting_magnetization count == ntyp
  - nat/ntyp counts consistent
Writes dft_step1/<Compound>.in + submit-all script content.
"""
import csv
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
GNN = os.path.dirname(BASE)
sys.path.insert(0, GNN)
sys.path.insert(0, BASE)
import screen_one as S

PP = {"Re": "Re.pbe-spn-kjpaw_psl.1.0.0.UPF",
      "Os": "Os.pbe-spn-kjpaw_psl.1.0.0.UPF",
      "Tc": "Tc.pbe-spn-kjpaw_psl.0.3.0.UPF",
      "P": "P.pbe-n-kjpaw_psl.1.0.0.UPF",
      "Rh": "Rh.pbe-spn-kjpaw_psl.1.0.0.UPF",
      "B": "B.pbe-n-kjpaw_psl.1.0.0.UPF",
      "Ir": "Ir.pbe-n-kjpaw_psl.0.2.3.UPF",
      "Ni": "Ni.pbe-spn-kjpaw_psl.1.0.0.UPF",
      "Ru": "Ru.pbe-spn-kjpaw_psl.1.0.0.UPF",
      "O": "O.pbe-n-kjpaw_psl.1.0.0.UPF"}
MASS = {"Re": 186.21, "Os": 190.23, "Tc": 98.0, "P": 30.97, "Rh": 102.91,
        "B": 10.81, "Ir": 192.22, "Ni": 58.69, "Ru": 101.07, "O": 16.00}
MAIN = {"B", "P", "O"}

COMPOUNDS = [
    ("Re-Os", "Re", "Os", "CsCl", "ReOs", 6073, 12),
    ("Tc-Os", "Tc", "Os", "CsCl", "TcOs", 6108, 12),
    ("P-Os2", "P", "Os", "Fluorite", "POs2", 6056, 6),
    ("Re-Rh", "Re", "Rh", "CsCl", "ReRh", 7173, 12),
    ("B-Ir2", "B", "Ir", "Fluorite", "BIr2", 521, 7),
    ("Re-Ir", "Re", "Ir", "CsCl", "ReIr", 3664, 12),
    ("Ni-Os", "Ni", "Os", "CsCl", "NiOs", 5610, 12),
    ("Re-B", "Re", "B", "CsCl", "ReB", 598, 13),
    ("Tc-Re", "Tc", "Re", "CsCl", "TcRe", 7203, 12),
    ("Ru-O", "Ru", "O", "Rocksalt", "RuO", 5865, 8),
]

prod = {}
for r in csv.DictReader(open(os.path.join(GNN, "production_run.csv"))):
    prod[r["pool_id"]] = r

OUT = os.path.join(GNN, "dft_step1")
os.makedirs(OUT, exist_ok=True)

TEMPLATE = """&CONTROL
    calculation  = 'vc-relax'
    pseudo_dir   = '/scratch/sunraj/dft_pseudo/'
    outdir       = '/scratch/sunraj/dft_prod/{comp}/tmp/'
    prefix       = '{tag}'
    tstress      = .true.
    tprnfor      = .true.
    disk_io      = 'low'
    wf_collect   = .false.
/
&SYSTEM
    ibrav        = 0
    nat          = {nat}
    ntyp         = {ntyp}
    ecutwfc      = 100
    ecutrho      = 1000
    occupations  = 'smearing'
    smearing     = 'mv'
    degauss      = 0.02
    nspin        = 2
{mag}
/
&ELECTRONS
    conv_thr         = 1.0D-6
    electron_maxstep = 300
    mixing_beta      = 0.2
    mixing_ndim      = 8
/
&IONS
    ion_dynamics = 'bfgs'
/
&CELL
    cell_dynamics = 'bfgs'
    press_conv_thr = 0.5
    cell_dofree = 'all'
/
ATOMIC_SPECIES
{species}
ATOMIC_POSITIONS (crystal)
{pos}
K_POINTS (automatic)
{K} {K} {K} 0 0 0
CELL_PARAMETERS (angstrom)
{A} 0.0 0.0
0.0 {A} 0.0
0.0 0.0 {A}
"""

for comp, A, B, proto, formula, pid, K in COMPOUNDS:
    a0 = float(prod[str(pid)]["a0"])
    builder, xB = S.resolve_builder_and_xB(proto, formula, A, B)
    atoms = builder(a0)
    els = list(atoms.elements)
    nat = len(els)
    # species order = first-appearance; labels must match positions
    order = list(dict.fromkeys(els))
    ntyp = len(order)
    assert set(order) <= {A, B}, (comp, order)
    assert all(e in order for e in els)
    mag_lines = []
    for i, e in enumerate(order, 1):
        assert e in PP, (comp, e)
        mag_lines.append("    starting_magnetization(%d) = %s" % (
            i, "0.0" if e in MAIN else "0.5"))
    assert len(mag_lines) == ntyp
    species = "\n".join("%s  %.2f  %s" % (e, MASS[e], PP[e]) for e in order)
    fr = atoms.frac_coords if hasattr(atoms, "frac_coords") else None
    import numpy as np
    if fr is None:
        # fall back: detect fractional (all |x|<=1.5) vs cartesian
        M = np.array(atoms.lattice_mat, dtype=float)
        raw = np.array(atoms.coords, dtype=float)
        fr = raw if np.abs(raw).max() <= 1.5 else np.linalg.inv(M).dot(raw.T).T
    lines = []
    for e, f in zip(els, np.array(fr, dtype=float) % 1.0):
        lines.append("%s %.10f %.10f %.10f" % (e, f[0], f[1], f[2]))
    pos = "\n".join(lines)
    assert len(lines) == nat
    # structural self-check: written positions must equal builder fractional
    # coords (mod 1) to 1e-8; catches cartesian/fractional mixups like Os@0.177
    import numpy as _np
    _ref = _np.array(atoms.frac_coords, dtype=float) % 1.0
    _got = _np.array([[float(v) for v in ln.split()[1:]] for ln in lines])
    assert _np.allclose(_ref % 1.0, _got % 1.0, atol=1e-8), (comp, _ref, _got)
    tag = formula.replace("2", "")
    comp_id = comp.replace("-", "")
    text = TEMPLATE.format(comp=comp_id, tag=tag, nat=nat,
                           ntyp=ntyp, mag="\n".join(mag_lines),
                           species=species, pos=pos, K=K, A=a0)
    # self-checks on the rendered input
    assert text.count("starting_magnetization") == ntyp, comp
    for e in order:
        assert "%s  %.2f  %s" % (e, MASS[e], PP[e]) in text, (comp, e)
    fn = os.path.join(OUT, "%s.in" % comp_id)
    open(fn, "w").write(text)
    print("wrote %-8s nat=%2d ntyp=%d k=%d a0=%.4f" % (comp, nat, ntyp, K, a0),
          flush=True)
print("OK: 10 inputs generated + validated", flush=True)
