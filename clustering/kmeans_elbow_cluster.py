import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse import load_npz
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------- CONFIG ----------------
DEMAND_PATH = Path("total_demand.csv")
AFFINITY_NPZ = Path("affinity_sparse.npz")
SKU_INDEX_PATH = Path("affinity_sku_index.csv")

K_MIN = 2
K_MAX = 20
RANDOM_STATE = 42
N_INIT = 20

# ---------------- LOAD DEMAND ----------------
demand_df = pd.read_csv(DEMAND_PATH)
demand_df["item_code"] = demand_df["item_code"].astype(str)

if "total_demand" not in demand_df.columns and "item_quantity" in demand_df.columns:
    demand_df = demand_df.rename(columns={"item_quantity": "total_demand"})

demand_df["total_demand"] = pd.to_numeric(
    demand_df["total_demand"], errors="coerce"
).fillna(0)

# ---------------- LOAD AFFINITY ----------------
A = load_npz(AFFINITY_NPZ).tocsr()
sku_codes = pd.read_csv(SKU_INDEX_PATH)["item_code"].astype(str).tolist()

avg_affinity = A.mean(axis=1).A1.astype(np.float32)
max_affinity = A.max(axis=1).toarray().ravel().astype(np.float32)

affinity_features = pd.DataFrame({
    "item_code": sku_codes,
    "avg_affinity": avg_affinity,
    "max_affinity": max_affinity
})

# ---------------- MERGE ----------------
features = demand_df.merge(affinity_features, on="item_code", how="inner")

print("Total SKUs used:", len(features))

# ---------------- BUILD FEATURE MATRIX ----------------
X = features[["total_demand", "avg_affinity", "max_affinity"]].copy()

# Log transform demand (important for skew)
X["total_demand"] = np.log1p(X["total_demand"])

# ---------------- NORMALIZATION ----------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- ELBOW METHOD ----------------
ks = list(range(K_MIN, K_MAX + 1))
wcss = []

print("\nComputing WCSS...")
for k in ks:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
    km.fit(X_scaled)
    wcss.append(km.inertia_)
    print(f"K={k:2d}  WCSS={km.inertia_:.4f}")

plt.figure()
plt.plot(ks, wcss, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Inertia)")
plt.title("Elbow Method (No Degree)")
plt.xticks(ks)
plt.show()

# ---------------- MANUAL K ----------------
best_k = 5   # ← manually set based on elbow inspection

print(f"\nUsing manual K = {best_k}")


# ---------------- FINAL CLUSTER ----------------
final_km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=N_INIT)
features["cluster"] = final_km.fit_predict(X_scaled)

features.to_csv("sku_kmeans_clusters_no_degree.csv", index=False)

print("\nSaved: sku_kmeans_clusters_no_degree.csv")

# ---------------- SUMMARY ----------------
summary = features.groupby("cluster").agg(
    n_skus=("item_code", "count"),
    mean_total_demand=("total_demand", "mean"),
    mean_avg_affinity=("avg_affinity", "mean"),
    mean_max_affinity=("max_affinity", "mean")
).sort_values("n_skus", ascending=False)

print("\nCluster Summary:")
print(summary)