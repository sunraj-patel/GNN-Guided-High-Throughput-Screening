"""Generate the candidate A-B pool for screening -> pairs_list.csv

Each row is ONE candidate system: (A, B, prototype-orientation).

Coverage rule (per user): do NOT screen both AB and BA for AB prototypes
(CsCl/Rocksalt/Zincblende) -- they are the same crystal with swapped
sublattice labels. A is chosen as the cation (lower electronegativity).

For the AB2 prototypes (Fluorite) both A-B2 and A2-B orientations are
distinct compounds and both are generated (if chemically allowed).

Column layout of pairs_list.csv:
  pool_id,A,B,proto,formula,elnegA,elnegB,coverage,low_coverage

where
  proto   in {CsCl,Rocksalt,Zincblende,Fluorite}
  formula  e.g. 'MoC' for AB, 'MoC2' or 'Mo2C' for fluorite orientations
  coverage = min(training-count(A), training-count(B))
  low_coverage = True if coverage < LOW_COV_THRESHOLD
"""
import os, json, argparse, itertools
from pymatgen.core import Element

BASE = os.path.dirname(os.path.abspath(__file__))
COV = json.load(open(os.path.join(BASE, "element_coverage.json")))
REF = json.load(open(os.path.join(BASE, "elemental_refs_energy.json")))
LOW_COV_THRESHOLD = 150

REF_ENERGY = {k: v for k, v in REF.items() if (v or {}).get("energy_per_atom") is not None}

NOBLE = {"Ar", "He", "Ne", "Rn", "Kr", "Xe", "H"}
# H excluded from bonding pool by default (requires different ref/prototype handling)

# ---- curated chemistry groups ----
EARLY_TM = ["Ti", "V", "Cr", "Zr", "Nb", "Mo", "Hf", "Ta", "W"]
PLATINUM = ["Tc", "Re", "Mn", "Fe", "Co", "Ni", "Ru", "Rh", "Pd", "Os", "Ir", "Pt"]
LIGHT_NM = ["B", "C", "N", "Si", "P", "S", "F", "O"]
HALIDE = ["F", "Cl", "Br", "I"]
ALK = ["Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba", "Cu", "Ag", "Au"]

# ---- electronegativity (Pauling) ----
def _en(el):
    try:
        v = Element(el).X
        return v if v is not None else 2.0
    except Exception:
        return 2.0

EN = {el: _en(el) for el in REF_ENERGY}


def usable_elements():
    return [e for e in REF_ENERGY if e not in NOBLE and e in COV]


def canon_order(A, B):
    """Return (A,B) with the lower-electronegativity element first (cation)."""
    if EN[A] <= EN[B]:
        return A, B
    return B, A


def coverage(A, B):
    return (COV.get(A, {}).get("count", 0), COV.get(B, {}).get("count", 0))


def build_rows(pairs, prototypes_ab, do_fluorite=True, include_pure=False):
    """Given (A,B) pairs, expand to candidate rows. Three nested loops: A x B x proto."""
    rows = []
    seen_formula = set()
    for A, B in pairs:
        # For pure (A==B) keep as is, no canon_order swap needed (identical)
        if A != B:
            A, B = canon_order(A, B)
        else:
            if not include_pure:
                continue
        for p in prototypes_ab:
            # Pure: formula is just element (e.g., Si for Si-Si)
            if A == B:
                formula = f"{A}"
            else:
                formula = f"{A}{B}" if p != "Zincblende" else f"{A}{B}"
            key = (tuple(sorted((A, B))), p)
            rows.append((A, B, p, formula))
        if do_fluorite:
            if A == B:
                # Pure fluorite is just elemental FCC (A) — single entry, not A/B2 double
                rows.append((A, B, "Fluorite", f"{A}"))
            else:
                # A B2
                rows.append((A, B, "Fluorite", f"{A}{B}2"))
                # A2 B (inverse fluorite)
                rows.append((A, B, "Fluorite", f"{A}2{B}"))
    # assign ids + coverage
    out = []
    for i, (A, B, p, formula) in enumerate(rows, start=1):
        cA, cB = coverage(A, B)
        out.append({
            "pool_id": i, "A": A, "B": B, "proto": p, "formula": formula,
            "elnegA": round(EN[A], 3), "elnegB": round(EN[B], 3),
            "covA": cA, "covB": cB, "coverage": min(cA, cB),
            "low_coverage": min(cA, cB) < LOW_COV_THRESHOLD,
        })
    return out


def default_pairs(mode, include_pure=False):
    sets = []
    if mode == "superhard" or mode == "all":
        tm = EARLY_TM + PLATINUM
        nm = LIGHT_NM
        for a, b in itertools.product(tm, nm):
            sets.append((a, b))
    if mode == "ionic" or mode == "all":
        for a, b in itertools.product(ALK, HALIDE):
            sets.append((a, b))
    # dedupe unordered pairs
    seen = set()
    pairs = []
    for a, b in sets:
        k = tuple(sorted((a, b)))
        if k not in seen:
            seen.add(k)
            pairs.append((a, b))
    if include_pure:
        # Add pure A-A for every element in the pool (three loops: A x B x proto includes diagonal)
        all_elems = sorted(set([e for a,b in pairs for e in (a,b)] + REF_ENERGY.keys()))
        # Use curated 29 as in comprehensive
        curated = sorted(set(EARLY_TM + PLATINUM + LIGHT_NM + HALIDE + ALK))
        # Limit to those with coverage data to avoid noble gases
        for el in curated:
            if el in NOBLE: continue
            if el not in REF_ENERGY: continue
            k = (el, el)
            if k not in seen:
                seen.add(k)
                pairs.append((el, el))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["superhard", "ionic", "all"], default="superhard")
    ap.add_argument("--metals", nargs="*", default=None)
    ap.add_argument("--nonmetals", nargs="*", default=None)
    ap.add_argument("--out", default="pairs_list.csv")
    ap.add_argument("--no-fluorite", action="store_true")
    ap.add_argument("--custom-pairs", default=None,
                    help="optional file with one 'A B' per line (unordered)")
    ap.add_argument("--include-pure", action="store_true",
                    help="include A-A pure elements (three loops: A x B x proto)")
    ap.add_argument("--three-loops", action="store_true",
                    help="full 29x29x4 comprehensive (A x B x proto, ordered)")
    args = ap.parse_args()

    if args.three_loops:
        # Explicit three nested loops as requested: for A in elems, for B in elems, for proto
        all_elems = sorted(set(REF_ENERGY.keys()) - NOBLE)
        # Curated 29 for consistency with comprehensive
        curated = [e for e in sorted(set(EARLY_TM + PLATINUM + LIGHT_NM + HALIDE + ALK)) if e not in NOBLE and e in REF_ENERGY]
        elems = curated
        pairs = [(a,b) for a in elems for b in elems]
    elif args.custom_pairs:
        pairs = []
        for ln in open(args.custom_pairs):
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            a, b = ln.split()[:2]
            pairs.append((a, b))
    else:
        metals = args.metals or (EARLY_TM + PLATINUM if args.mode != "ionic" else ALK)
        nonmetals = args.nonmetals or (LIGHT_NM if args.mode != "ionic" else HALIDE)
        pairs = []
        seen = set()
        for a in metals:
            for b in nonmetals:
                k = tuple(sorted((a, b)))
                if k not in seen:
                    seen.add(k)
                    pairs.append((a, b))
        if args.include_pure:
            pairs = default_pairs(args.mode, include_pure=True)

    rows = build_rows(pairs, ["CsCl", "Rocksalt", "Zincblende"],
                      do_fluorite=not args.no_fluorite, include_pure=(args.include_pure or args.three_loops))
    with open(args.out, "w") as fp:
        fp.write("pool_id,A,B,proto,formula,elnegA,elnegB,covA,covB,coverage,low_coverage\n")
        for r in rows:
            fp.write(",".join(str(r[k]) for k in
                    ["pool_id", "A", "B", "proto", "formula", "elnegA", "elnegB",
                     "covA", "covB", "coverage", "low_coverage"]) + "\n")
    nlow = sum(1 for r in rows if r["low_coverage"])
    nf = sum(1 for r in rows if r["proto"] == "Fluorite")
    print(f"wrote {args.out}: {len(rows)} candidate rows "
          f"({len(rows)-nf} AB-prototype, {nf} fluorite-orientation)")
    print(f"  unique pairs sampled: {len(pairs)}")
    print(f"  low-coverage rows (flag): {nlow}")
    print("  prototype breakdown:", {p: sum(1 for r in rows if r['proto'] == p)
                                      for p in ["CsCl", "Rocksalt", "Zincblende", "Fluorite"]})


if __name__ == "__main__":
    main()
