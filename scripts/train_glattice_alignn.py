"""Train ALIGNN G_lattice models (diagnostic only, CPU).

Four runs: {abs,corr} x {disjoint,random}. Each run dir holds id_prop.csv in
[train..., val..., test...] order; training uses keep_data_order with explicit
n_train/n_val/n_test so the composition-disjoint (or random) test is exactly
the manifest test set. Small model + early stopping: 412 rows cannot support
the full 6-layer production architecture.

Does NOT touch G_energy / G_hull / G_bulk checkpoints or configs.
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "alignn"))

from alignn.train_alignn import train_for_folder

REF = json.load(open(os.path.join(BASE, "config_ehull.json")))
SPLITS = json.load(open(os.path.join(BASE, "glattice_splits.json")))["splits"]

RUNS = {
    "glattice_run_abs_disjoint": ("glattice_abs_disjoint", "disjoint", "a0_A"),
    "glattice_run_corr_disjoint": ("glattice_corr_disjoint", "disjoint", "corr_factor"),
    "glattice_run_abs_random": ("glattice_abs_random", "random", "a0_A"),
    "glattice_run_corr_random": ("glattice_corr_random", "random", "corr_factor"),
}


def make_config(split, target, outdir):
    c = json.loads(json.dumps(REF))
    S = SPLITS[split]
    c["target"] = "energy_total"  # label-only in this ALIGNN version;
    # the regression key is hardcoded "target". Run identity lives in
    # filename/output_dir. (Custom target strings fail Literal validation.)
    # NB: filename MUST be unique per run: the LMDB cache path is CWD-relative
    # (alignn/lmdb_dataset.py uses bare tmp_name), so two runs sharing a
    # filename silently reuse each other's cached graphs.
    c["filename"] = "glattice_%s_%s" % (target, split)
    c["output_dir"] = outdir
    c["train_ratio"], c["val_ratio"], c["test_ratio"] = 0.0, 0.0, 0.0
    c["n_train"], c["n_val"], c["n_test"] = len(S["train"]), len(S["val"]), len(S["test"])
    c["keep_data_order"] = True
    c["epochs"] = 150
    c["batch_size"] = 16
    c["warmup_steps"] = 200
    c["n_early_stopping"] = 25
    c["num_workers"] = 0
    # NB: keep use_lmdb=True. The non-LMDB dataset path in this repo snapshot
    # lacks the target_additional_output kwarg that train_alignn passes;
    # the LMDB path (also used by the production runs) accepts it.
    c["use_lmdb"] = True
    # small diagnostic architecture (412 rows total)
    c["model"]["alignn_layers"] = 3
    c["model"]["gcn_layers"] = 3
    c["model"]["embedding_features"] = 32
    c["model"]["hidden_features"] = 128
    return c


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else None
    jobs = [which] if which else sorted(RUNS)
    for name in jobs:
        datadir, split, target = RUNS[name]
        outdir = os.path.join(BASE, name)
        os.makedirs(outdir, exist_ok=True)
        cfg_path = os.path.join(outdir, "config_glattice.json")
        json.dump(make_config(split, target, outdir), open(cfg_path, "w"), indent=1)
        print("=== training %s [%s/%s] ===" % (name, split, target), flush=True)
        train_for_folder(
            0, 0,
            root_dir=os.path.join(BASE, datadir),
            config_name=cfg_path,
            file_format="poscar",
            output_dir=outdir,
        )
        print("=== done %s ===" % name, flush=True)


if __name__ == "__main__":
    main()
