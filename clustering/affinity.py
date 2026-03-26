import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from pathlib import Path

# ---------- CONFIG ----------
BASE = Path(__file__).parent
DATA_PATH = BASE / "../raw_order.csv"
ORDER_COL = "order_id"
SKU_COL = "item_code"
SEP = ";"

# ---------- LOAD ----------
df = pd.read_csv(DATA_PATH, sep=SEP, usecols=[ORDER_COL, SKU_COL])
df[ORDER_COL] = df[ORDER_COL].astype(str)
df[SKU_COL] = df[SKU_COL].astype(str)

# Drop duplicates so r_oi is binary (presence/absence)
df = df.drop_duplicates([ORDER_COL, SKU_COL])

# ---------- INDEXING ----------
order_codes, order_idx = np.unique(df[ORDER_COL].values, return_inverse=True)
sku_codes, sku_idx = np.unique(df[SKU_COL].values, return_inverse=True)

num_orders = len(order_codes)
num_skus = len(sku_codes)

print(f"Orders: {num_orders:,}")
print(f"SKUs:   {num_skus:,}")
print(f"Rows in (order,sku) after dedup: {len(df):,}")

# ---------- BUILD SPARSE R (Orders x SKUs) ----------
data = np.ones(len(df), dtype=np.int8)
R = csr_matrix((data, (order_idx, sku_idx)), shape=(num_orders, num_skus))

# ---------- AFFINITY: A = (R^T R) / |O| ----------
# co-occurrence counts
C = (R.T @ R).astype(np.float32)

# convert counts to affinity
A = C * (1.0 / float(num_orders))  # still sparse

# optional: remove diagonal (self-affinity)
A.setdiag(0)
A.eliminate_zeros()

# ---------- CHECK AFFINITY FUNCTION ----------

from scipy.sparse import load_npz

def check_affinity(sku_i: str, sku_j: str):
    # Load saved sparse matrix
    A = load_npz(str(BASE / "affinity_sparse.npz"))

    # Load SKU index mapping
    sku_codes = pd.read_csv(BASE / "affinity_sku_index.csv")["item_code"].tolist()
    sku_to_idx = {sku: i for i, sku in enumerate(sku_codes)}

    if sku_i not in sku_to_idx:
        print(f"{sku_i} not found.")
        return
    if sku_j not in sku_to_idx:
        print(f"{sku_j} not found.")
        return

    i = sku_to_idx[sku_i]
    j = sku_to_idx[sku_j]

    affinity_value = A[i, j]

    print(f"Affinity between {sku_i} and {sku_j}: {float(affinity_value):.6f}")
    return float(affinity_value)


# ---------- SAVE ----------
# Save affinity sparse matrix
save_npz(str(BASE / "affinity_sparse.npz"), A)

# Save sku index mapping (so you can interpret rows/cols)
pd.Series(sku_codes, name="item_code").to_csv(BASE / "affinity_sku_index.csv", index=False)

# Save sparse affinity as triplet format (i, j, value)
A_coo = A.tocoo()

affinity_sparse_df = pd.DataFrame({
    "sku_i": [sku_codes[i] for i in A_coo.row],
    "sku_j": [sku_codes[j] for j in A_coo.col],
    "affinity": A_coo.data
})

affinity_sparse_df.to_csv(BASE / "affinity_sparse_pairs.csv", index=False)

print("Saved: affinity_sparse.npz")
print("Saved: affinity_sku_index.csv")
print("Saved non-zero affinity pairs to CSV.")

# Example usage:
check_affinity("14101074001", "10002014001")