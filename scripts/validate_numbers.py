#!/usr/bin/env python3
"""Recompute every number used in the manuscript from the canonical data in ../data.

Writes ../data/numbers.json (consumed by make_figures.py and build_main.py) and
runs assertions against the values quoted in the task brief / old draft.  Known
conflicts between sources are NOT silently asserted away; they are collected in
`discrepancies` and written to numbers.json so that they appear in the QC report.
"""
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             r2_score, recall_score, roc_auc_score)

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "data")
N = {}
DISC = []


def close(a, b, tol, what):
    ok = abs(a - b) <= tol
    if not ok:
        DISC.append({"what": what, "computed": float(a), "quoted": float(b)})
    return ok


def must(a, b, tol, what):
    assert abs(a - b) <= tol, "%s: computed %r vs expected %r" % (what, a, b)


# ---------------------------------------------------------------- production
P = pd.read_csv(os.path.join(D, "production_run.csv"), keep_default_na=False,
                na_values=[""])
assert len(P) == 4976 and P.pool_id.is_unique and (P.quality == "ok").all()
P["aa"] = P.A == P.B


def comp_key(r):
    c = Counter()
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", r.formula):
        c[el] += int(n) if n else 1
    return (tuple(sorted(c.items())), r.proto)


P["key"] = P.apply(comp_key, axis=1)
AB = P[~P.aa]
ABu = AB.drop_duplicates("key")
prod = {
    "rows": len(P), "AB_rows": len(AB), "AA_rows": int(P.aa.sum()),
    "distinct_configs": int(P.key.nunique()), "distinct_AB": len(ABu),
    "proto_rows": P.proto.value_counts().to_dict(),
    "fluorite_AB2": int(((P.proto == "Fluorite") & (P.x_B > 0.5)).sum()),
    "fluorite_A2B": int(((P.proto == "Fluorite") & (P.x_B < 0.5)).sum()),
    "n_le_005_rows": int((P.e_hull <= 0.05).sum()),
    "n_le_002_rows": int((P.e_hull <= 0.02).sum()),
    "n_le_005_distinctAB": int((ABu.e_hull <= 0.05).sum()),
    "n_le_002_distinctAB": int((ABu.e_hull <= 0.02).sum()),
    "n_le_002_AA": int(((P.e_hull <= 0.02) & P.aa).sum()),
    "proto_le_002_rows": P[P.e_hull <= 0.02].proto.value_counts().to_dict(),
    "proto_le_002_distinctAB": ABu[ABu.e_hull <= 0.02].proto.value_counts().to_dict(),
    "proto_le_005_rows": P[P.e_hull <= 0.05].proto.value_counts().to_dict(),
    "median_B_rows": float(P.bulk_GPa.median()),
    "median_Ehull_rows": float(P.e_hull.median()),
    "median_B_distinctAB": float(ABu.bulk_GPa.median()),
    "median_Ehull_distinctAB": float(ABu.e_hull.median()),
    "n_dup_pairs": int(len(AB) - len(ABu)),
}
must(prod["rows"], 4976, 0, "rows")
must(prod["AB_rows"], 4800, 0, "AB rows")
must(prod["AA_rows"], 176, 0, "AA rows")
must(prod["fluorite_AB2"], 1004, 0, "AB2")
must(prod["fluorite_A2B"], 960, 0, "A2B")
must(prod["n_le_005_rows"], 806, 0, "<=0.05")
must(prod["n_le_002_distinctAB"], 472, 0, "<=0.02 distinct A-B")
must(prod["median_B_rows"], 112.5, 0.05, "median B")
must(prod["median_Ehull_rows"], 0.442, 0.0005, "median Ehull")
for k, v in {"CsCl": 148, "Rocksalt": 82, "Zincblende": 58, "Fluorite": 184}.items():
    must(prod["proto_le_002_distinctAB"][k], v, 0, "eligible " + k)
N["production"] = prod
# the brief's 472 counts distinct A-B only; whole-table counts differ -> record
DISC.append({"what": "E_hull<=0.02 count over all 4976 rows (incl. 176 A-A and 70 mirror-duplicate A-B rows)",
             "computed": prod["n_le_002_rows"], "quoted": 472})
DISC.append({"what": "E_hull<=0.05 count: brief/tab_search 806 is over all 4976 rows; distinct A-B only",
             "computed": prod["n_le_005_distinctAB"], "quoted": 806})

# radii / prototype factors
R = json.load(open(os.path.join(D, "atomic_radii.json")))
FAC_CODE = {"CsCl": 1.155, "Rocksalt": 2.0, "Zincblende": 2.309, "Fluorite": 2.309}  # ea_predict.py
FAC_EXACT = {"CsCl": 2 / np.sqrt(3), "Rocksalt": 2.0, "Zincblende": 4 / np.sqrt(3),
             "Fluorite": 4 / np.sqrt(3)}
chk = P.apply(lambda r: FAC_CODE[r.proto] * (R[r.A] + R[r.B]), axis=1)
must((chk - P.a_guess).abs().max(), 0, 1e-3, "a_guess = factor*(rA+rB) with code constants")
N["a_guess_factor_check"] = {
    "code_constants": FAC_CODE, "exact": FAC_EXACT,
    "max_abs_diff_codeconst_vs_data_A": float((chk - P.a_guess).abs().max()),
    "max_rel_rounding_A": float(max(abs(FAC_CODE[k] / FAC_EXACT[k] - 1) for k in FAC_CODE)),
}
assert abs(P.a0 - P.a_guess * P.c_pred).max() < 1e-6

# ---------------------------------------------------------------- G_lattice
gl = json.load(open(os.path.join(D, "glattice_summary_disjoint.json")))["metrics"]
glr = json.load(open(os.path.join(D, "glattice_summary_random.json")))["metrics"]
sp = json.load(open(os.path.join(D, "glattice_splits.json")))["splits"]
meth = ["a_guess", "g_energy", "xgb_corr", "mlatabc_corr", "alignn_abs", "alignn_corr"]
brief = {"AB_all": [5.47, 12.27, 2.60, 2.07, 1.90, 1.56],
         "AA_all": [16.94, 19.19, 5.72, 5.10, 6.75, 4.18]}
for sub in brief:
    for m, b in zip(meth, brief[sub]):
        must(gl[sub][m]["mae"], b, 0.006, sub + m)
must(gl["AB_all"]["alignn_corr"]["median"], 1.19, 0.006, "median")
must(gl["AB_all"]["alignn_corr"]["mae_A"], 0.068, 0.0006, "mae A")
N["glattice"] = {
    "disjoint": {s: {m: gl[s][m] for m in meth + ["xgb_abs", "mlatabc_abs"]} for s in brief},
    "random": {s: {m: glr[s][m] for m in meth + ["xgb_abs", "mlatabc_abs"]} for s in brief},
    "n_split_disjoint": {k: len(v) for k, v in sp["disjoint"].items()},
    "n_split_random": {k: len(v) for k, v in sp["random"].items()},
    "n_test_AB": gl["AB_all"]["a_guess"]["n"], "n_test_AA": gl["AA_all"]["a_guess"]["n"],
    "n_test_AB_random": glr["AB_all"]["a_guess"]["n"], "n_test_AA_random": glr["AA_all"]["a_guess"]["n"],
}
must(N["glattice"]["n_test_AB"], 48, 0, "n AB")
must(N["glattice"]["n_test_AA"], 9, 0, "n AA")

# ---------------------------------------------------------------- retrospective
def stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {"n": len(x), "mae": float(np.mean(abs(x))), "median_abs": float(np.median(abs(x))),
            "rms": float(np.sqrt(np.mean(x ** 2))), "bias": float(np.mean(x))}


RA = pd.read_csv(os.path.join(D, "retrospective_v2_exact.csv"))
RB = pd.read_csv(os.path.join(D, "retrospective_v2_aa.csv"))
retro = {"AB": {"lattice_pct": stats(RA.a_err_pct), "B_pct": stats(RA.b_err_pct),
                "B_GPa": stats(RA.b_err_GPa)}}
bB = RB.dropna(subset=["b_err_GPa"])
retro["AA"] = {"lattice_pct": stats(RB.a_err_pct),
               "B_GPa_all": stats(bB.b_err_GPa),
               "B_pct_all": stats(bB.b_err_pct)}
bB5 = bB[bB.b_mp >= 5]
retro["AA"]["B_pct_Bmp_ge5"] = stats(bB5.b_err_pct)
retro["AA"]["B_GPa_Bmp_ge5"] = stats(bB5.b_err_GPa)
N["retrospective"] = retro
a, b, g = retro["AB"]["lattice_pct"], retro["AB"]["B_pct"], retro["AB"]["B_GPa"]
must(a["n"], 325, 0, "n AB lattice"); must(a["median_abs"], 9.39, 0.006, "med")
must(a["mae"], 10.49, 0.006, "mae"); must(a["rms"], 12.85, 0.006, "rms"); must(a["bias"], -7.11, 0.006, "bias")
must(b["n"], 250, 0, "nB"); must(b["median_abs"], 9.86, 0.006, "Bmed"); must(b["mae"], 17.96, 0.006, "Bmae")
must(b["rms"], 29.17, 0.006, "Brms"); must(g["bias"], 1.88, 0.006, "Bbias"); must(g["mae"], 21.32, 0.006, "BmaeGPa")
a = retro["AA"]["lattice_pct"]
must(a["n"], 87, 0, "n AA"); must(a["median_abs"], 11.49, 0.006, "AA med"); must(a["mae"], 13.11, 0.006, "AA mae")
must(a["rms"], 17.06, 0.006, "AA rms"); must(a["bias"], -2.27, 0.006, "AA bias")
bb = retro["AA"]["B_pct_Bmp_ge5"]; gg = retro["AA"]["B_GPa_Bmp_ge5"]
close(bb["n"], 64, 0, "AA B n (B_mp>=5 GPa)")
close(bb["median_abs"], 14.79, 0.006, "AA B median abs rel (%)")
close(bb["mae"], 21.19, 0.006, "AA B rel MAE (%)")
close(bb["rms"], 28.97, 0.006, "AA B rms rel (%)")
close(gg["bias"], -11.20, 0.006, "AA B bias GPa")
close(gg["mae"], 28.17, 0.006, "AA B abs MAE GPa")
# the retrospective a_pred is the energy-scan (G_energy) value, not G_lattice:
fr = pd.read_csv(os.path.join(D, "framework_run_dedup.csv"), keep_default_na=False)
m = RA.merge(fr[["pool_id", "a0"]], on="pool_id")
N["retrospective_apred_equals_energy_scan_a0"] = bool(np.allclose(m.a_pred, m.a0.astype(float)))

# ---------------------------------------------------------------- G_bulk
T = pd.read_csv(os.path.join(D, "gbulk_test_predictions.csv"), skipinitialspace=True)
y, p = T.target.values, T.prediction.values
gb = {"n": len(T), "mae": float(np.mean(abs(y - p))), "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
      "r2": float(r2_score(y, p)), "spearman": float(spearmanr(y, p)[0])}
must(gb["n"], 492, 0, "gbulk n"); must(gb["mae"], 10.1, 0.05, "mae"); must(gb["rmse"], 22.2, 0.05, "rmse")
must(gb["r2"], 0.933, 0.0005, "r2"); must(gb["spearman"], 0.976, 0.0005, "rho")
gb["pretrained_reference_from_deck_only"] = {"mae": 10.5, "rmse": 23.0, "r2": 0.93, "spearman": 0.97}
N["gbulk"] = gb

# ---------------------------------------------------------------- G_hull
Z = np.load(os.path.join(D, "alignn_ehull_preds.npz"), allow_pickle=True)
yt, pt = Z["y"], Z["p"]
st, pr = yt <= 0.02, pt <= 0.02
gh = {"n_test": len(yt), "pos_frac": float(st.mean()),
      "mae": float(np.mean(abs(yt - pt))), "rmse": float(np.sqrt(np.mean((yt - pt) ** 2))),
      "r2": float(r2_score(yt, pt)), "f1": float(f1_score(st, pr)),
      "precision": float(precision_score(st, pr)), "recall": float(recall_score(st, pr)),
      "roc_auc_ranking": float(roc_auc_score(st, -pt)), "pr_auc_ranking": float(average_precision_score(st, -pt)),
      "roc_auc_hardlabel": float(roc_auc_score(st, pr)), "pr_auc_hardlabel": float(average_precision_score(st, pr))}
must(gh["mae"], 0.0604, 0.00005, "ghull mae"); must(gh["rmse"], 0.1263, 0.00005, "rmse"); must(gh["r2"], 0.861, 0.0005, "r2")
must(gh["f1"], 0.794, 0.0005, "f1"); must(gh["precision"], 0.772, 0.0005, "prec"); must(gh["recall"], 0.818, 0.0005, "rec")
close(gh["roc_auc_hardlabel"], 0.832, 0.0005, "ALIGNN ROC-AUC 0.832 equals hard-label AUC (=balanced accuracy)")
close(gh["roc_auc_ranking"], 0.832, 0.0005, "ALIGNN ROC-AUC from continuous score")
close(gh["pr_auc_ranking"], 0.702, 0.0005, "ALIGNN PR-AUC from continuous score")
close(gh["n_test"], 2077, 0, "n_test (brief 2077; ALIGNN file has 2075)")
close(gh["pos_frac"], 0.393, 0.0005, "positive fraction (brief 39.3%)")
# competing models: values ONLY from RESULTS_FRAMEWORK_LOG.md (raw predictions not in repo)
gh["others_from_log"] = {
    "XGBoost": dict(n=2077, mae=0.1165, rmse=0.2331, r2=0.522, roc=0.884, pr=0.821, f1=0.753, prec=0.757, rec=0.749),
    "Roost": dict(n=2077, mae=0.1478, rmse=0.2938, r2=0.237, roc=0.799, pr=0.685, f1=0.634, prec=0.644, rec=0.624),
    "CGCNN": dict(n=2077, mae=0.0820, rmse=0.1736, r2=0.734, roc=0.883, pr=0.802, f1=0.771, prec=0.780, rec=0.760)}
DISC.append({"what": "CGCNN MAE/RMSE/R2: group-meeting decks 0.0823/0.1864/0.693 vs RESULTS_FRAMEWORK_LOG.md 0.0820/0.1736/0.734",
             "computed": "log: 0.0820/0.1736/0.734", "quoted": "decks/brief: 0.0823/0.1864/0.693"})
N["ghull"] = gh

# ---------------------------------------------------------------- DFT candidates
POOL = {"Re-Os": 6073, "Tc-Os": 6108, "P-Os2": 6056, "Re-Rh": 7173, "B-Ir2": 521,
        "Re-Ir": 3664, "Ni-Os": 5610, "Re-B": 598, "Tc-Re": 8702, "Ru-O": 5865}
DIR = {"Re-Os": "ReOs", "Tc-Os": "TcOs", "P-Os2": "POs2", "Re-Rh": "ReRh", "B-Ir2": "BIr2",
       "Re-Ir": "ReIr", "Ni-Os": "NiOs", "Re-B": "ReB", "Tc-Re": "TcRe", "Ru-O": "RuO"}
# data/final_qc.json and data/eos_fits.json are the frozen final DFT records (gnn/final_qc.json,
# gnn/eos_fits.json, 17 Oct): P-Os2 and B-Ir2 strain sets are 24/24; all EOS fits except Ru-O pass QC.
# Ru-O: gnn/final_results_package/FINAL_QC.md s.7b withdraws its elastic/EOS numbers (strain and EOS
# SCFs left the 7.49 muB reference branch); only a_DFT (vc-relax) stays valid.
WITHDRAWN = {"Ru-O"}
QC = json.load(open(os.path.join(D, "final_qc.json")))
EOS = json.load(open(os.path.join(D, "eos_fits.json")))
cands = []
for name, pid in POOL.items():
    r = P[P.pool_id == pid].iloc[0]
    q, e = QC[DIR[name]], EOS[DIR[name]]
    C11, C12, C44 = q["C11"], q["C12"], q["C44"]
    Bel = (C11 + 2 * C12) / 3
    assert abs(Bel - q["B_el"]) < 1e-6
    info = json.load(open(os.path.join(D, "candidates_info", DIR[name] + ".json")))
    aD = info["relaxed_cell_A"][0][0]  # vc-relax cell fed to the reference SCF
    assert abs(info["relaxed_cell_A"][1][1] - aD) < 1e-9
    c = dict(name=name, proto=r.proto, pool_id=pid, a_gnn=float(r.a0), B_gnn=float(r.bulk_GPa),
             ehull=float(r.e_hull), a_dft=aD, da=100 * (aD - r.a0) / r.a0,
             a_eos=e.get("a0_EOS"), B_eos=e.get("B0") if e.get("fit") else None,
             B_el=Bel, dB=100 * (Bel - r.bulk_GPa) / r.bulk_GPa, C11=C11, C12=C12, C44=C44,
             Cp=(C11 - C12) / 2, n_strain=q["n_strain"], n_eos=e["n"], mag=info["reference_scf"]["mag_tot"], a_qcjson=q["a_relax"], relax_src=info["relax_src"],
             R2min=q["R2min"], P_scf=q["P_scf"],
             born=(C11 - C12 > 0 and C11 + 2 * C12 > 0 and C44 > 0),
             complete=(q["n_strain"] == 24), withdrawn=(name in WITHDRAWN))
    c["violated"] = ("C44<0" if C44 <= 0 else "C11<C12" if C11 - C12 <= 0 else "")
    if c["withdrawn"]:
        c["born"] = None
        c["violated"] = "withdrawn (C44<0 on a non-ground-state magnetic branch)"
    if c["born"]:
        c["GV"] = (C11 - C12 + 3 * C44) / 5
        c["BG"] = Bel / c["GV"]
        c["AZ"] = 2 * C44 / (C11 - C12)
    c["dB_eos_vs_el"] = (100 * (c["B_eos"] - Bel) / Bel) if c["B_eos"] else None
    c["da_eos_vs_relax"] = (100 * (c["a_eos"] - aD) / aD) if c["a_eos"] else None
    cands.append(c)
N["candidates"] = cands
byn = {c["name"]: c for c in cands}
# brief rows (a_DFT there is a mixed EOS/relax source; compare B, C, Born and dB only)
brief_B = {"Ni-Os": (293.9, 302.9, 3.06), "Ru-O": (266.7, 183.4, -31.23), "Re-B": (346.0, 348.1, 0.61),
           "Tc-Os": (352.5, 325.3, -7.72), "Re-Rh": (306.2, 300.3, -1.93), "Re-Ir": (344.9, 331.3, -3.94),
           "Tc-Re": (340.4, 327.2, -3.88), "Re-Os": (361.5, 354.6, -1.91), "P-Os2": (319.9, 293.5, -8.25),
           "B-Ir2": (313.2, 303.7, -3.03)}
for k, (bg, bd, db) in brief_B.items():
    must(byn[k]["B_gnn"], bg, 0.06, k + " B_gnn"); must(byn[k]["B_el"], bd, 0.06, k + " B_el")
    must(byn[k]["dB"], db, 0.03, k + " dB (brief used rounded inputs)")
brief_a = {"Ni-Os": (2.8721, 2.9270, 1.92), "Ru-O": (4.2673, 4.4698, 4.74), "Re-B": (2.6651, 2.7462, 3.04),
           "Tc-Os": (2.8171, 3.0877, 9.60), "Re-Rh": (2.9300, 3.0887, 5.42), "Re-Ir": (2.9511, 3.1183, 5.66),
           "Tc-Re": (2.8748, 3.0937, 7.61), "Re-Os": (2.9401, 3.1005, 5.46), "P-Os2": (5.4781, 5.5823, 1.90),
           "B-Ir2": (4.9346, 5.3476, 8.37)}
for k, (ag, ad, da) in brief_a.items():
    must(byn[k]["a_gnn"], ag, 6e-5, k + " a_gnn")
    if abs(byn[k]["a_dft"] - ad) > 6e-5:
        DISC.append({"what": "a_DFT %s: brief (source mixed) vs vc-relax cell used here" % k,
                     "computed": round(byn[k]["a_dft"], 4), "quoted": ad,
                     "note": "brief value equals EOS a0" if byn[k]["a_eos"] and abs(byn[k]["a_eos"] - ad) < 6e-4 else ""})
na = [c["da"] for c in cands]
nb = [c["dB"] for c in cands if not c["withdrawn"]]
nb_all = [c["dB"] for c in cands]
N["candidate_summary"] = {
    "da_mean_signed": float(np.mean(na)), "da_mape": float(np.mean(np.abs(na))),
    "da_min": float(min(na)), "da_max": float(max(na)),
    "dB_mean_signed": float(np.mean(nb)), "dB_mape": float(np.mean(np.abs(nb))), "dB_max_abs": float(np.max(np.abs(nb))),
    "dB_n": len(nb), "dB_mape_incl_RuO_withdrawn": float(np.mean(np.abs(nb_all))),
    "dB_median_abs": float(np.median(np.abs(nb))),
    "n_GNN_B_above_DFT": int(sum(c["dB"] < 0 for c in cands if not c["withdrawn"])),
    "n_born_violate": int(sum(c["born"] is False for c in cands)),
    "n_dft_a_above_gnn": int(sum(c["da"] > 0 for c in cands)),
    "n_born_pass": int(sum(c["born"] is True for c in cands)),
    "n_born_pass_complete": int(sum(c["born"] is True and c["complete"] for c in cands)),
    "n_complete": int(sum(c["complete"] for c in cands)),
    "n_ehull_gt_002": int(sum(c["ehull"] > 0.02 for c in cands)),
    "max_abs_dB_eos_vs_el": float(max(abs(c["dB_eos_vs_el"]) for c in cands if c["dB_eos_vs_el"] is not None and not c["withdrawn"])),
    "n_eos_pass": int(sum(1 for c in cands if c["B_eos"] and not c["withdrawn"])),
    "max_abs_da_eos_vs_relax": float(max(abs(c["da_eos_vs_relax"]) for c in cands if c["da_eos_vs_relax"] is not None and not c["withdrawn"])),
}
cs = N["candidate_summary"]
close(cs["da_mean_signed"], 5.4, 0.05, "mean signed da (abstract 5.4%)")
close(cs["dB_mape"], 3.82, 0.005, "MAPE of B over the nine QC-passing compounds (final package 3.82)")
close(cs["dB_max_abs"], 8.27, 0.005, "max |dB| (final package 8.27)")
close(cs["n_GNN_B_above_DFT"], 7, 0, "GNN B above DFT for seven of nine")
must(cs["n_born_violate"], 6, 0, "six Born-violating (QC-passing)")
must(cs["n_ehull_gt_002"], 3, 0, "three candidates above 0.02")
must(cs["n_born_pass"], 3, 0, "three Born-positive")
must(cs["n_born_pass_complete"], 3, 0, "all three Born-positive sets are 24/24")
for k in ["Re-B", "Tc-Re", "Re-Os"]:
    pass
must(byn["Ru-O"]["C44"], -147.7, 0.05, "RuO C44"); must(byn["Ni-Os"]["C44"], 229.0, 0.05, "NiOs C44")

# ---------------------------------------------------------------- rankings
EL = ABu[ABu.e_hull <= 0.02].sort_values(["bulk_GPa", "pool_id"], ascending=[False, True])
rank = {}
for proto in ["CsCl", "Rocksalt", "Fluorite", "Zincblende"]:
    s = EL[EL.proto == proto].head(10)
    rank[proto] = [dict(formula=r.formula, A=r.A, B=r.B, x_B=float(r.x_B), a0=float(r.a0),
                        B_GPa=float(r.bulk_GPa), ehull=float(r.e_hull), pool_id=int(r.pool_id)) for r in s.itertuples()]
N["rankings"] = rank

# ---------------------------------------------------------------- per-prototype counts
prod["proto_distinctAB"] = ABu.proto.value_counts().to_dict()
prod["proto_le_005_distinctAB"] = ABu[ABu.e_hull <= 0.05].proto.value_counts().to_dict()
prod["proto_AB_rows"] = AB.proto.value_counts().to_dict()
prod["proto_AA_rows"] = P[P.aa].proto.value_counts().to_dict()

# ---------------------------------------------------------------- local MP snapshot status (composition + space group)
MP = pd.read_csv(os.path.join(D, "mp_snapshot_slim.csv"), keep_default_na=False)
PROTO_SG = {"CsCl": 221, "Rocksalt": 225, "Zincblende": 216, "Fluorite": 225}
mp_idx = {}
for r in MP.itertuples():
    c = Counter()
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", r.formula):
        c[el] += int(n) if n else 1
    if len(c) != 2:
        continue
    mp_idx.setdefault(tuple(sorted(c.items())), []).append((r.material_id, int(r.space_group_number)))


def mp_status(A, B, formula, proto):
    c = Counter()
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        c[el] += int(n) if n else 1
    hits = mp_idx.get(tuple(sorted(c.items())), [])
    if not hits:
        return {"status": "no_entry", "sgs": []}
    sgs = sorted({sg for _, sg in hits})
    if PROTO_SG[proto] in sgs:
        return {"status": "same_sg_entry", "sgs": sgs, "ids": [m for m, sg in hits if sg == PROTO_SG[proto]][:2]}
    return {"status": "other_sg_only", "sgs": sgs}


JV = {x["pool_id"]: x for x in json.load(open(os.path.join(D, "jarvis_shortlist_check.json")))}
for c in N["candidates"]:
    r = P[P.pool_id == c["pool_id"]].iloc[0]
    c["mp"] = mp_status(r.A, r.B, r.formula, r.proto)
    j = JV.get(str(c["pool_id"]))
    c["jarvis"] = ({"status": "not_checked"} if j is None else
                   {"status": "no_entry" if j["n_jarvis"] == 0 else ("same_sg" if j["same_sg"] else "other_sg_only"),
                    "sgs": j["all_sg"], "n": j["n_jarvis"]})
for proto, lst in N["rankings"].items():
    for e in lst:
        e["mp"] = mp_status(e["A"], e["B"], e["formula"], proto)
exp_mp = {"Re-Os": "other_sg_only", "Tc-Os": "no_entry", "P-Os2": "other_sg_only", "Re-Rh": "no_entry", "B-Ir2": "no_entry",
          "Re-Ir": "other_sg_only", "Ni-Os": "no_entry", "Re-B": "other_sg_only", "Tc-Re": "other_sg_only", "Ru-O": "no_entry"}
for c in N["candidates"]:
    if c["mp"]["status"] != exp_mp[c["name"]]:
        DISC.append({"what": "MP status %s: earlier shortlist (mirrored stoichiometry lookup) vs exact composition" % c["name"],
                     "computed": c["mp"], "quoted": exp_mp[c["name"]]})
    assert c["mp"]["status"] != "same_sg_entry"
# explicit mirror check for the two fluorite candidates (earlier lookup also matched the reversed stoichiometry)
N["mirror_stoichiometry_check"] = {
    "Os2P (candidate P-Os2)": mp_idx.get((("Os", 2), ("P", 1)), []),
    "OsP2 (reversed stoichiometry)": mp_idx.get((("Os", 1), ("P", 2)), []),
    "BIr2": mp_idx.get((("B", 1), ("Ir", 2)), []), "B2Ir": mp_idx.get((("B", 2), ("Ir", 1)), [])}

# ---------------------------------------------------------------- energy-curvature cross-checks (raw outputs in data/energy_check)
RY = 14710.5  # GPa per Ry/bohr^3 (convention used in the project)


def _e(path):
    t = open(path).read()
    e = re.findall(r"^!\s+total energy\s+=\s+(-?\d+\.\d+)", t, re.M)
    v = re.findall(r"unit-cell volume\s+=\s+(\d+\.\d+)", t)
    assert "JOB DONE" in t and e
    return float(e[-1]), float(v[0])


ec = {}
for c in ["NiOs", "RuO"]:
    d0 = os.path.join(D, "energy_check", c)
    V = _e(os.path.join(d0, "scf.out"))[1]
    ds = [-0.01, -0.005, 0.005, 0.01]
    ex = np.array([_e(os.path.join(d0, "xx_%g.out" % d))[0] for d in ds])
    exy = np.array([_e(os.path.join(d0, "xy_%g.out" % d))[0] for d in ds])
    p2 = np.polyfit(ds, ex, 2)[0]                       # E = E0 + a d + b d^2 -> C11 = 2b/V
    C11 = 2 * p2 * RY / V
    A = np.vstack([np.ones(4), np.array(ds) ** 2]).T    # even fit for shear: E = E0 + a d^2, d2E/dd2 = C44*V/4
    a = np.linalg.lstsq(A, exy, rcond=None)[0][1]
    C44 = 8 * a * RY / V
    ec[c] = {"C11_energy": float(C11), "C44_energy": float(C44), "V_bohr3": V}
td = {}
files = sorted(os.listdir(os.path.join(D, "energy_check", "NiOs_tet")))
dd, ee = [], []
for f in files:
    if f.endswith(".out"):
        dd.append(float(re.search(r"tet_([+-]\d\.\d+)", f).group(1)))
        e_, v_ = _e(os.path.join(D, "energy_check", "NiOs_tet", f)); ee.append(e_)
Cp = np.polyfit(dd, ee, 2)[0] / (6 * v_) * RY
ec["NiOs_tet_Cprime_energy"] = float(Cp)
N["energy_curvature"] = ec
close(ec["NiOs"]["C11_energy"], 269.2, 1.0, "NiOs C11 energy-curvature (FINAL_QC 269.2)")
close(ec["NiOs"]["C44_energy"], 227.9, 1.0, "NiOs C44 energy-curvature (FINAL_QC 227.9)")
close(ec["RuO"]["C44_energy"], -148.1, 1.0, "RuO C44 energy-curvature (FINAL_QC -148.1)")
close(Cp, -23.5, 0.3, "NiOs C' from tetragonal scan (FINAL_QC -23.5)")

N["discrepancies"] = DISC
json.dump(N, open(os.path.join(D, "numbers.json"), "w"), indent=1, default=float)
print("ALL ASSERTIONS PASSED; %d logged discrepancies" % len(DISC))
for d in DISC:
    print(" -", d)
