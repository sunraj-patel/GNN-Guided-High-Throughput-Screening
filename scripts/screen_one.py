"""Screen ONE candidate system (A, B, prototype-orientation).

Full pipeline per candidate:
  1. two-stage E(a) scan with GNN_energy  -> a0, E0, curvature, quality
  2. predict bulk modulus at a0 with GNN_bulk
  3. formation energy + e_above_hull vs MP binary hull (from local cache)

Reads the candidate from a line of pairs_list.csv (by pool_id) or from
arguments. Writes results/row_<pool_id>.json.

Usage:
  python screen_one.py --pool_id 5 --pairs pairs_list.csv [--outdir results]
  python screen_one.py --A Mo --B C --proto CsCl --formula MoC   (adhoc)
"""
import os, sys, json, argparse, time, logging
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import numpy as np
import prototypes as P
import ea_predict as E
import predict_bulk as PB
import hull

LOG = logging.getLogger("screen_one")

def setup_logging(logfile=None, level=logging.INFO):
    """Add timestamped, line-buffered logging to stderr/stdout (for SLURM
    log capture) and optionally a dedicated per-candidate log file."""
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(level)
    # stdout handler (flushed per line so log files update live)
    so = logging.StreamHandler(sys.stdout)
    so.setFormatter(fmt)
    so.flush = sys.stdout.flush
    root.addHandler(so)
    if logfile:
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        fh.flush = fh.stream.flush
        root.addHandler(fh)
    return root

def t0():
    return logging.makeLogRecord({})

def elapsed(start):
    return time.time() - start

# fluorite orientation routing: formula "A2B" vs "AB2"
def reverse_fluorite(prototype, a, A, B, **kw):
    return P.inverse_fluorite(a, A, B)

def flore_a2b(prototype, a, A, B, **kw):
    return P.fluorite(a, A, B)

def flore_ab2(prototype, a, A, B, **kw):
    return P.fluorite(a, A, B)


def resolve_builder_and_xB(proto, formula, A, B):
    """Return (builder, xB) for the candidate."""
    if proto in ("CsCl", "Rocksalt", "Zincblende"):
        return lambda a: P.build(proto, a, A, B), 0.5
    # Fluorite: A2B -> A on 8c (anti), AB2 -> A on 4a
    if formula.endswith("2B") or formula.endswith("2" + B + ""):
        return lambda a: P.inverse_fluorite(a, A, B), 1.0 / 3.0
    return lambda a: P.fluorite(a, A, B), 2.0 / 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_id", type=int, default=None)
    ap.add_argument("--pairs", default="pairs_list.csv")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--A", default=None)
    ap.add_argument("--B", default=None)
    ap.add_argument("--proto", default=None)
    ap.add_argument("--formula", default=None)
    ap.add_argument("--logfile", default=None,
                    help="optional dedicated log file (in addition to stdout)")
    args = ap.parse_args()

    setup_logging(args.logfile)
    os.makedirs(args.outdir, exist_ok=True)
    start_wall = time.time()

    if args.pool_id is not None:
        with open(args.pairs) as fp:
            header = fp.readline().strip().split(",")
            idx = {c: i for i, c in enumerate(header)}
            found = None
            for ln in fp:
                parts = ln.strip().split(",")
                if int(parts[idx["pool_id"]]) == args.pool_id:
                    found = {h: parts[idx[h]] for h in header}
                    break
        if found is None:
            print(f"pool_id {args.pool_id} not found"); sys.exit(2)
        A, B, proto, formula = (found["A"], found["B"],
                                found["proto"], found["formula"])
        covA, covB, coverage, low_cov = (
            int(found["covA"]), int(found["covB"]), int(found["coverage"]),
            found["low_coverage"])
        pool_id = args.pool_id
    else:
        A, B, proto, formula = args.A, args.B, args.proto, args.formula
        covA = covB = coverage = -1
        low_cov = False
        pool_id = 0

    try:
        builder, xB = resolve_builder_and_xB(proto, formula, A, B)
    except KeyError:
        LOG.error("bad prototype %s; aborting", proto); sys.exit(2)

    LOG.info("=== START candidate pool_id=%s %s %s proto=%s xB=%.4f coverage=%s ===",
             pool_id, A, B, proto, xB, coverage)

    # ---- 1. two-stage E(a) scan ----
    t = time.time()
    LOG.info("Stage 1/3: two-stage E(a) scan (energy model)")
    scan = E.scan_energy(proto, A, B, builder=builder)
    a0 = scan["a0"]
    quality = scan["quality"]
    LOG.info("  E(a) done in %.1fs -> a0=%s E0=%s curvature=%s quality=%s (a_guess=%.3f)",
             time.time() - t,
             ("%.4f" % a0) if a0 is not None else "None",
             ("%.4f" % scan["E0"]) if scan["E0"] is not None else "None",
             ("%.4f" % scan["curvature"]) if scan["curvature"] is not None else "None",
             quality, scan["a_guess"])

    # ---- 2. bulk modulus at a0 + physical plausibility guard ----
    Bmod = None
    min_nn = None
    if a0 is not None and np.isfinite(a0):
        t = time.time()
        LOG.info("Stage 2/3: predict bulk modulus at a0=%.4f (bulk model)", a0)
        try:
            atoms = builder(a0)
            Bmod = PB.predict_bulk(atoms)
            # minimum interatomic distance (with periodic images) as a sanity
            # check for egregious overlap/compression (nn < ~1.1 A is
            # physically impossible for these elements).
            dmat = np.array(atoms.raw_distance_matrix, dtype=float)
            off_diag = dmat[dmat > 1e-3]
            min_nn = float(off_diag.min()) if off_diag.size else None
            if min_nn is not None and min_nn < 1.1:
                quality = "compressed"  # override: a0 physically implausible
            LOG.info("  bulk done in %.1fs -> B=%.1f GPa min_nn=%.3f A quality=%s",
                     time.time() - t, Bmod if Bmod is not None else float("nan"),
                     min_nn if min_nn is not None else float("nan"), quality)
        except Exception as e:
            Bmod = None
            LOG.warning("  bulk prediction failed: %r", e)

    # ---- 3. formation energy + e_above_hull ----
    E0 = scan["E0"]
    eah = lower = nphases = None
    if E0 is not None:
        LOG.info("Stage 3/3: formation energy + e_above_hull vs MP binary hull")
        phases = hull.fetch_system_phases(A, B)  # cached
        nphases = len(phases)
        if phases:
            cand = E0
            cand_Ef = hull.formation_energy(cand, A, B, 1 - xB, xB)
            eah, lower = hull.e_above_hull(A, B, phases, xB, cand_Ef)
        else:
            cand_Ef = hull.formation_energy(E0, A, B, 1 - xB, xB)
        LOG.info("  hull: %d MP phases; E_f=%.4f eV/atom e_above_hull=%s",
                 nphases,
                 (cand_Ef if cand_Ef is not None else float("nan")),
                 ("%.4f" % eah) if eah is not None else "None (no hull)")

    result = {
        "pool_id": pool_id, "A": A, "B": B, "proto": proto, "formula": formula,
        "x_B": xB,
        "coverage": coverage, "covA": covA, "covB": covB,
        "low_coverage": low_cov,
        "a0": a0, "E0": E0, "curvature": scan["curvature"],
        "E_drop_fine": scan["E_drop_fine"], "quality": quality,
        "a_guess": scan["a_guess"],
        "bulk_GPa": Bmod,
        "min_nn_A": min_nn,
        "formation_energy_eV_atom": cand_Ef if E0 is not None else None,
        "e_above_hull": eah, "hull_lower": lower, "n_mp_phases": nphases,
    }
    out = os.path.join(args.outdir, f"row_{pool_id}.json")
    with open(out, "w") as fp:
        json.dump(result, fp, indent=1)
    LOG.info(
        "=== DONE %s %s proto=%s -> a0=%s E0=%s B=%.1f GPa "
        "E_f=%.4f e_above_hull=%s quality=%s (wall %.1fs) -> %s ===",
        A, B, proto,
        ("%.4f" % a0) if a0 is not None else "None",
        ("%.4f" % E0) if E0 is not None else "None",
        Bmod if Bmod is not None else float("nan"),
        (cand_Ef if cand_Ef is not None else float("nan")),
        ("%.4f" % eah) if eah is not None else "None",
        quality, time.time() - start_wall, out)


if __name__ == "__main__":
    main()
