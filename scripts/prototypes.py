"""Generate standardized conventional cubic prototype cells.

Prototypes (conventional cubic cells):
  CsCl       (B2,  Pm-3m)  : AB  , 2 atoms
  Rocksalt   (B1,  Fm-3m)  : AB  , 8 atoms
  Zincblende (B3,  F-43m)  : AB  , 8 atoms
  Fluorite   (C1,  Fm-3m)  : AB2 , 12 atoms

Each is built from exact fractional coordinates on a cubic lattice so the
single scalar lattice parameter `a` fully defines the geometry. Generation
does NOT substitute atoms into arbitrary MP cells -- it builds the prototype
from scratch.

FK = {element: fractional coordinates} in the conventional cubic cell.
"""
import numpy as np
from jarvis.core.atoms import Atoms

IDENTITY = np.eye(3)

FCC_FRAC = [
    [0.0, 0.0, 0.0],
    [0.0, 0.5, 0.5],
    [0.5, 0.0, 0.5],
    [0.5, 0.5, 0.0],
]


def _cubic_atoms(a, species, fracs):
    """Build jarvis Atoms from a cubic lattice constant + fractional sites."""
    lat = IDENTITY * a
    return Atoms(
        lattice_mat=lat,
        elements=list(species),
        coords=[list(map(float, f)) for f in fracs],
        cartesian=False,
    )


def cscl(a, A, B):
    """CsCl (B2), AB, conventional cubic, 2 atoms."""
    return _cubic_atoms(a, [A, B], [[0, 0, 0], [0.5, 0.5, 0.5]])


def rocksalt(a, A, B):
    """Rocksalt (B1), AB, Fm-3m, conventional cubic, 8 atoms."""
    species = []
    fracs = []
    for f in FCC_FRAC:
        species.append(A)
        fracs.append(f)                       # A at 4a
        species.append(B)
        fracs.append([x + 0.5 for x in f])    # B at 4b
    return _cubic_atoms(a, species, fracs)


def zincblende(a, A, B):
    """Zincblende (B3), AB, F-43m, conventional cubic, 8 atoms."""
    species = []
    fracs = []
    for f in FCC_FRAC:
        species.append(A)
        fracs.append(f)                                          # A at 4a
        species.append(B)
        fracs.append([x + 0.25 for x in f])                      # B at 4c
    return _cubic_atoms(a, species, fracs)


def fluorite(a, A, B):
    """Fluorite (CaF2, C1), AB2, Fm-3m, conventional cubic, 12 atoms."""
    species = []
    fracs = []
    for f in FCC_FRAC:
        species.append(A)
        fracs.append(f)                                # A at 4a
        for shift in ([0.25, 0.25, 0.25], [0.75, 0.75, 0.75]):
            species.append(B)
            fracs.append([x + s for x, s in zip(f, shift)])    # B at 8c
    return _cubic_atoms(a, species, fracs)


def inverse_fluorite(a, A, B):
    """A2B (anti-fluorite) by swapping sites of fluorite."""
    return fluorite(a, B, A)


BUILDERS = {
    "CsCl": cscl,
    "Rocksalt": rocksalt,
    "Zincblende": zincblende,
    "Fluorite": fluorite,
}


def build(prototype, a, A, B):
    """Return jarvis Atoms for a prototype at lattice constant a."""
    if prototype not in BUILDERS:
        raise KeyError(f"unknown prototype {prototype}; options {list(BUILDERS)}")
    return BUILDERS[prototype](a, A, B)
