import ast
import os
import pandas as pd
import numpy as np

OUTPUT_CSV = "mp_binary_bulk_modulus.csv"
STRUCTURE_DIR = "structures"
DATA_DIR = "bulk_modulus_data"

os.makedirs(DATA_DIR, exist_ok=True)

df = pd.read_csv(OUTPUT_CSV)
withbm = df[df["bulk_modulus_GPa"].notna()].copy()

rows = []
for _, row in withbm.iterrows():
    try:
        d = ast.literal_eval(row["bulk_modulus_GPa"])
        vrh = float(d.get("vrh"))
    except Exception:
        continue
    if not (0.0 < vrh < 500.0):
        continue
    rows.append((row["material_id"], vrh))

print(f"Selected {len(rows)} materials with physical bulk modulus (0 < vrh < 500 GPa)")

cif_ids = set(f.split(".")[0] for f in os.listdir(STRUCTURE_DIR))
missing = 0
with open(os.path.join(DATA_DIR, "id_prop.csv"), "w") as f:
    f.write("jid,bulk_modulus_vrh\n")
    for mid, vrh in rows:
        if mid not in cif_ids:
            missing += 1
            continue
        src = os.path.join(STRUCTURE_DIR, f"{mid}.cif")
        dst = os.path.join(DATA_DIR, f"{mid}.cif")
        if not os.path.lexists(dst):
            os.symlink(os.path.abspath(src), dst)
        f.write(f"{mid}.cif,{vrh:.6f}\n")

print(f"Missing CIFs: {missing}")
print(f"Wrote id_prop.csv with {len(rows) - missing} entries to {DATA_DIR}/")
