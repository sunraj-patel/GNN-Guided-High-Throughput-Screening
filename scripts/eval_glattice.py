"""Evaluate G_lattice experiment (diagnostic only).

Methods on each test tag: a_guess, xgb_abs, xgb_corr, alignn_abs, alignn_corr,
g_energy (current framework scan result). Reference: MP a0 from the strict v2
sets. Metrics per group: MAE%, median|e|%, RMSE-rel%, bias%, within 2/5/10%,
absolute MAE (A), and n. Groups: AB-all, AA-all, per (subset, prototype).

Correction diagnostics: train c=a_MP/a_guess distribution, test predicted-c
distribution, and whether ML reduces the a_guess bias.

If an ML method beats a_guess on the primary (composition-disjoint) test, the
script also writes row-level CSV, summary JSON, and the four required plots.
Otherwise it reports honestly and saves nothing (per the experiment brief).
"""
import csv
import json
import math
import os
import statistics
import sys

import numpy as np
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

sp = json.load(open(os.path.join(BASE, "glattice_splits.json")))
META, SPLITS = sp["meta"], sp["splits"]
XGB = json.load(open(os.path.join(BASE, "glattice_xgb_preds.json")))
MLAT = json.load(open(os.path.join(BASE, "glattice_mlatabc_preds.json")))

RUNS = {
    "disjoint": {"abs": "glattice_run_abs_disjoint",
                 "corr": "glattice_run_corr_disjoint"},
    "random": {"abs": "glattice_run_abs_random",
               "corr": "glattice_run_corr_random"},
}

DEVICE = "cpu"


def load_alignn_rundir(d):
    from alignn.config import TrainingConfig, ALIGNNAtomWiseConfig
    from alignn.models.alignn_atomwise import ALIGNNAtomWise
    raw = json.load(open(os.path.join(BASE, d, "config_glattice.json")))
    m = {k: v for k, v in raw["model"].items()
         if not isinstance(v, (dict, list))}
    cfg = TrainingConfig(**{**{k: v for k, v in raw.items() if k != "model"},
                            "model": ALIGNNAtomWiseConfig(**m)})
    model = ALIGNNAtomWise(cfg.model).to(DEVICE)
    ckpt = os.path.join(BASE, d, "best_model.pt")
    state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def alignn_predict(model, cfg, tags):
    from alignn.graphs import Graph as JarvisGraph
    from jarvis.core.atoms import Atoms as JAtoms
    import dgl
    outs = {}
    for i in range(0, len(tags), 32):
        chunk = tags[i:i + 32]
        gs, lgs, lats = [], [], []
        for t in chunk:
            a = JAtoms.from_poscar(os.path.join(BASE, "glattice_structures", t + ".vasp"))
            gd = JarvisGraph.atom_dgl_multigraph(
                a, cutoff=cfg.cutoff, max_neighbors=cfg.max_neighbors,
                atom_features=cfg.atom_features, compute_line_graph=True,
                use_canonize=cfg.use_canonize, cutoff_extra=cfg.cutoff_extra)
            g, lg = gd
            gs.append(g); lgs.append(lg)
            lats.append(torch.tensor(a.lattice_mat, dtype=torch.float32))
        bg, blg = dgl.batch(gs), dgl.batch(lgs)
        lat = torch.stack(lats)
        with torch.no_grad():
            o = model([bg.to(DEVICE), blg.to(DEVICE), lat.to(DEVICE)])["out"]
        for t, v in zip(chunk, o.detach().cpu().numpy().flatten().tolist()):
            outs[t] = float(v)
    return outs


def rel(a, ref):
    return 100.0 * (a - ref) / ref


def metrics(errs_pct, errs_abs):
    ae = [abs(e) for e in errs_pct]
    return {"n": len(errs_pct), "mae": statistics.mean(ae),
            "median": statistics.median(ae),
            "rms": math.sqrt(sum(e * e for e in errs_pct) / len(errs_pct)),
            "bias": statistics.mean(errs_pct),
            "w2": sum(1 for e in ae if e <= 2) / len(ae),
            "w5": sum(1 for e in ae if e <= 5) / len(ae),
            "w10": sum(1 for e in ae if e <= 10) / len(ae),
            "mae_A": statistics.mean(abs(e) for e in errs_abs)}


METHODS = ["a_guess", "xgb_abs", "xgb_corr", "mlatabc_abs", "mlatabc_corr",
           "alignn_abs", "alignn_corr", "g_energy"]


def method_pred(split, method, tag, alignn_cache):
    m = META[tag]
    if method == "a_guess":
        return m["a_guess"]
    if method == "g_energy":
        return m["a_energy"]
    if method == "xgb_abs":
        return XGB["%s_abs" % split][tag]
    if method == "xgb_corr":
        return m["a_guess"] * XGB["%s_corr" % split][tag]
    if method == "mlatabc_abs":
        return MLAT["%s_abs" % split][tag]
    if method == "mlatabc_corr":
        return m["a_guess"] * MLAT["%s_corr" % split][tag]
    if method == "alignn_abs":
        return alignn_cache[split]["abs"][tag]
    if method == "alignn_corr":
        return m["a_guess"] * alignn_cache[split]["corr"][tag]
    raise KeyError(method)


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    splits = [only] if only else ["disjoint", "random"]
    alignn_cache = {}
    for split in splits:
        alignn_cache[split] = {}
        for which in ["abs", "corr"]:
            model, cfg = load_alignn_rundir(RUNS[split][which])
            alignn_cache[split][which] = alignn_predict(
                model, cfg, SPLITS[split]["test"])
            print("predicted %s/%s n=%d" % (split, which, len(SPLITS[split]["test"])),
                  flush=True)

    report = {}
    for split in splits:
        test = SPLITS[split]["test"]
        report[split] = {}
        groups = {"AB_all": [t for t in test if META[t]["subset"] == "AB"],
                  "AA_all": [t for t in test if META[t]["subset"] == "AA"]}
        for proto in ["CsCl", "Rocksalt", "Zincblende", "Fluorite"]:
            groups["AB_%s" % proto] = [t for t in groups["AB_all"]
                                       if META[t]["proto"] == proto]
            groups["AA_%s" % proto] = [t for t in groups["AA_all"]
                                       if META[t]["proto"] == proto]
        for gname, tags in groups.items():
            if not tags:
                continue
            report[split][gname] = {}
            for meth in METHODS:
                ep, ea = [], []
                for t in tags:
                    p = method_pred(split, meth, t, alignn_cache)
                    if p is None or not np.isfinite(p):
                        continue
                    ref = META[t]["a_mp"]
                    ep.append(rel(p, ref)); ea.append(p - ref)
                if ep:
                    report[split][gname][meth] = metrics(ep, ea)

    # correction diagnostics
    diag = {}
    for split in splits:
        S = SPLITS[split]
        c_tr = [META[t]["corr_true"] for t in S["train"]]
        c_te = [META[t]["corr_true"] for t in S["test"]]
        pred_c = {}
        for meth in ["xgb_corr", "mlatabc_corr", "alignn_corr"]:
            pc = []
            for t in S["test"]:
                if meth == "xgb_corr":
                    pc.append(XGB["%s_corr" % split][t])
                elif meth == "mlatabc_corr":
                    pc.append(MLAT["%s_corr" % split][t])
                else:
                    pc.append(alignn_cache[split]["corr"][t])
            pred_c[meth] = pc
        b_ml = {}
        for meth in ["xgb_corr", "mlatabc_corr", "alignn_corr",
                     "xgb_abs", "mlatabc_abs", "alignn_abs"]:
            errs = []
            for t in S["test"]:
                p = method_pred(split, meth, t, alignn_cache)
                errs.append(rel(p, META[t]["a_mp"]))
            b_ml[meth] = statistics.mean(errs)
        b_guess = statistics.mean(
            rel(META[t]["a_guess"], META[t]["a_mp"]) for t in S["test"])
        diag[split] = {
            "train_c": {"n": len(c_tr), "mean": statistics.mean(c_tr),
                        "std": statistics.pstdev(c_tr), "min": min(c_tr),
                        "max": max(c_tr),
                        "frac_below_1": sum(1 for c in c_tr if c < 1) / len(c_tr)},
            "test_c_true": {"mean": statistics.mean(c_te),
                            "std": statistics.pstdev(c_te),
                            "min": min(c_te), "max": max(c_te)},
            "test_c_pred": {m: {"mean": statistics.mean(v),
                                "std": statistics.pstdev(v),
                                "min": min(v), "max": max(v)}
                            for m, v in pred_c.items()},
            "test_bias_guess": b_guess,
            "test_bias_ml": b_ml,
        }

    json.dump({"metrics": report, "correction": diag},
              open(os.path.join(
                  BASE, "glattice_eval_%s.json" % (only or "all")), "w"), indent=1)

    # per-tag predictions for plots + row-level CSV
    preds = {}
    for split in splits:
        preds[split] = {}
        for meth in METHODS:
            preds[split][meth] = {
                t: method_pred(split, meth, t, alignn_cache)
                for t in SPLITS[split]["test"]}
    json.dump(preds, open(os.path.join(
        BASE, "glattice_preds_%s.json" % (only or "all")), "w"))
    print("wrote glattice_preds_%s.json" % (only or "all"))

    # print compact tables for the requested split(s)
    for split in splits:
        print("\n================ G_lattice eval: %s test ================" % split)
        for gname in ["AB_all", "AA_all", "AB_CsCl", "AB_Rocksalt",
                      "AB_Zincblende", "AB_Fluorite", "AA_CsCl", "AA_Rocksalt",
                      "AA_Zincblende", "AA_Fluorite"]:
            g = report[split].get(gname)
            if not g:
                continue
            n = g["a_guess"]["n"]
            print("-- %-14s n=%d" % (gname, n))
            for meth in METHODS:
                m = g.get(meth)
                if not m:
                    print("   %-11s n=0" % meth)
                    continue
                print("   %-11s MAE %5.2f med %5.2f RMS %5.2f bias %+6.2f | "
                      "2%% %4.1f 5%% %4.1f 10%% %4.1f | MAE_A %.3f" % (
                          meth, m["mae"], m["median"], m["rms"], m["bias"],
                          100 * m["w2"], 100 * m["w5"], 100 * m["w10"], m["mae_A"]))
        print("\ncorrection diagnostics (%s test):" % split)
        d = diag[split]
        print("  train c: mean %.4f std %.4f range [%.3f, %.3f], frac<1 %.2f" % (
            d["train_c"]["mean"], d["train_c"]["std"], d["train_c"]["min"],
            d["train_c"]["max"], d["train_c"]["frac_below_1"]))
        for m, v in d["test_c_pred"].items():
            print("  pred c %-11s mean %.4f std %.4f range [%.3f, %.3f]" % (
                m, v["mean"], v["std"], v["min"], v["max"]))
        print("  test bias: a_guess %+.2f | %s" % (
            d["test_bias_guess"],
            " | ".join("%s %+.2f" % (m, v) for m, v in d["test_bias_ml"].items())))
    print("\nwrote glattice_eval_%s.json" % (only or "all"))


if __name__ == "__main__":
    main()
