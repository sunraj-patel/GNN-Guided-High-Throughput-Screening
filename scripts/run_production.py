"""Production lattice run (frozen G_lattice correction model).

Per unique candidate (framework_run_dedup.csv, 4,976 rows):
  1. a0 = a_guess * c_pred, c_pred from frozen ALIGNN corr checkpoint
     (glattice_run_corr_disjoint/best_model.pt) on the fixed-5.0-A prototype
     graph -- the exact gated inference path.
  2. B = g_bulk.predict at a0 (bulk_modulus_run/best_model.pt).
  3. E_hull = g_hull.predict at a0 (best_model_gpu80.pt).
Fluorite AB2 vs A2B are distinct dedup rows with distinct builders and are
preserved separately. No energy scan, no MP calls.

Output: results_production/row_<pool_id>.json + production_run.csv
"""
import csv
import json
import os
import sys
import time
import logging
from multiprocessing import Pool

import numpy as np
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
torch.set_num_threads(1)

import screen_one as S

A_REF = 5.0
CORR_RUN = os.path.join(BASE, "glattice_run_corr_disjoint")
LOG = logging.getLogger("production")

_model_cache = {}


def _load_corr():
    from alignn.config import TrainingConfig, ALIGNNAtomWiseConfig
    from alignn.models.alignn_atomwise import ALIGNNAtomWise
    raw = json.load(open(os.path.join(CORR_RUN, "config_glattice.json")))
    m = {k: v for k, v in raw["model"].items() if not isinstance(v, (dict, list))}
    cfg = TrainingConfig(**{**{k: v for k, v in raw.items() if k != "model"},
                            "model": ALIGNNAtomWiseConfig(**m)})
    model = ALIGNNAtomWise(cfg.model)
    sd = torch.load(os.path.join(CORR_RUN, "best_model.pt"),
                    map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    model.eval()
    return model, cfg


def _get_models():
    global _model_cache
    if not _model_cache:
        import predict_bulk as PB
        import predict_ehull as PH
        cm, ccfg = _load_corr()
        PB._load(); PH._load()
        _model_cache = {"corr": cm, "ccfg": ccfg, "PB": PB, "PH": PH}
    return _model_cache


def _corr_predict_batch(model, cfg, atoms_list):
    import dgl
    from alignn.graphs import Graph as JarvisGraph
    gs, lgs, lats = [], [], []
    for atoms in atoms_list:
        g, lg = JarvisGraph.atom_dgl_multigraph(
            atoms, cutoff=cfg.cutoff, max_neighbors=cfg.max_neighbors,
            atom_features=cfg.atom_features, compute_line_graph=True,
            use_canonize=cfg.use_canonize, cutoff_extra=cfg.cutoff_extra)
        gs.append(g); lgs.append(lg)
        lats.append(torch.tensor(atoms.lattice_mat, dtype=torch.float32))
    with torch.no_grad():
        out = model([dgl.batch(gs), dgl.batch(lgs), torch.stack(lats)])["out"]
    return out.detach().cpu().numpy().flatten().tolist()


def proc_one(job):
    pid, pairs, outdir = job
    M = _get_models()
    row = None
    with open(pairs) as fp:
        for r in csv.DictReader(fp):
            if int(r["pool_id"]) == pid:
                row = r
                break
    if row is None:
        raise ValueError("pool_id %s missing" % pid)
    A, B, proto, formula = row["A"], row["B"], row["proto"], row["formula"]
    a_guess = float(row["a_guess"])
    builder, xB = S.resolve_builder_and_xB(proto, formula, A, B)

    c_pred = _corr_predict_batch(M["corr"], M["ccfg"], [builder(A_REF)])[0]
    a0 = float(a_guess * c_pred)

    bulk = ehull = min_nn = None
    quality = "ok"
    try:
        atoms = builder(a0)
        bulk = M["PB"].predict_bulk(atoms)
        ehull = M["PH"].predict_ehull(atoms)
        d = np.array(atoms.raw_distance_matrix, dtype=float)
        od = d[d > 1e-3]
        min_nn = float(od.min()) if od.size else None
        if min_nn is not None and min_nn < 1.1:
            quality = "compressed"
    except Exception:
        quality = "failed"

    result = {"pool_id": pid, "A": A, "B": B, "proto": proto,
              "formula": formula, "x_B": xB, "a_guess": a_guess,
              "c_pred": float(c_pred), "a0": a0,
              "bulk_GPa": bulk, "e_hull": ehull,
              "quality": quality, "min_nn_A": min_nn}
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "row_%d.json" % pid), "w") as fp:
        json.dump(result, fp, indent=1)
    return pid, True


def _work(job):
    try:
        return proc_one(job)[0], True, ""
    except Exception:
        import traceback
        return job[0], False, traceback.format_exc()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--nthreads", type=int, default=4)
    ap.add_argument("--pairs", default="framework_run_dedup.csv")
    ap.add_argument("--outdir", default="results_production")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    with open(args.pairs) as fp:
        all_ids = [int(r["pool_id"]) for r in csv.DictReader(fp)]
    if args.start is not None:
        all_ids = [p for p in all_ids if args.start <= p <= args.end]
    # skip completed rows (resume support)
    jobs = [(p, args.pairs, args.outdir) for p in all_ids
            if not os.path.exists(os.path.join(args.outdir, "row_%d.json" % p))]
    LOG.info("production run: %d rows (%d already done), threads=%d",
             len(jobs), len(all_ids) - len(jobs), args.nthreads)
    t0, ok, fail = time.time(), 0, 0
    with Pool(processes=args.nthreads) as pool:
        for n, (pid, success, info) in enumerate(
                pool.imap_unordered(_work, jobs), 1):
            if success:
                ok += 1
            else:
                fail += 1
                if fail <= 5:
                    LOG.error("FAIL %s:\n%s", pid, info)
            if n % 200 == 0 or n == len(jobs):
                el = time.time() - t0
                LOG.info("%d/%d ok=%d fail=%d %.0fs elapsed, %.2fs/row, ETA %.1f min",
                         n, len(jobs), ok, fail, el, el / n,
                         (len(jobs) - n) * el / n / 60.0)
    LOG.info("DONE ok=%d fail=%d in %.1f min", ok, fail, (time.time() - t0) / 60.0)


if __name__ == "__main__":
    main()
