import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage, dendrogram


RANDOM_STATE = 42
K_RANGE = range(2, 16)

# Dendrogram sample size (full dendrogram is too large for 7k+ points)
DENDRO_SAMPLE_N = 1200

# Linkage method + distance metric
# Common: ward + euclidean (ward requires euclidean)
LINKAGE_METHOD = "ward"
METRIC = "euclidean"


def run_ahc_pipeline(demand_file: Path, affinity_file: Path, tag: str):
    print(f"\n==============================")
    print(f"AHC Clustering for: {tag}")
    print(f"==============================")

    # -----------------------------
    # Load + merge
    # -----------------------------
    demand = pd.read_csv(demand_file)
    affinity = pd.read_csv(affinity_file)

    df = demand.merge(
        affinity[["item_code", "avg_affinity", "max_affinity"]],
        on="item_code",
        how="inner"
    )

    feature_cols = [
        "avg_affinity",
        "max_affinity",
        "mean_demand",
        "cv_demand",
        "demand_frequency"
    ]

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # -----------------------------
    # Normalize
    # -----------------------------
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    print("Total SKUs used:", len(df))

    # -----------------------------
    # 1) Dendrogram on a sample
    # -----------------------------
    np.random.seed(RANDOM_STATE)
    if len(df) > DENDRO_SAMPLE_N:
        sample_idx = np.random.choice(len(df), DENDRO_SAMPLE_N, replace=False)
    else:
        sample_idx = np.arange(len(df))

    X_s = X_scaled[sample_idx]

    print(f"Building dendrogram sample: {len(sample_idx)} points")

    Z = linkage(X_s, method=LINKAGE_METHOD, metric=METRIC)

    plt.figure(figsize=(12, 6))
    dendrogram(Z, no_labels=True, color_threshold=None)
    plt.title(f"Dendrogram (sample) - {tag} [{LINKAGE_METHOD}]")
    plt.xlabel("Sample index")
    plt.ylabel("Linkage distance")
    plt.tight_layout()
    plt.savefig(f"dendrogram_{tag}.png", dpi=200)
    plt.close()

    print(f"Saved dendrogram_{tag}.png")

    # Also plot last merge distances (helps see big jumps)
    # The last column of Z is distance
    last_dist = Z[-50:, 2] if Z.shape[0] >= 50 else Z[:, 2]
    plt.figure()
    plt.plot(range(1, len(last_dist) + 1), last_dist, marker="o")
    plt.title(f"Last merge distances (sample) - {tag}")
    plt.xlabel("Merge step (last 50)")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(f"ahc_merge_dist_{tag}.png", dpi=200)
    plt.close()
    print(f"Saved ahc_merge_dist_{tag}.png")

    # -----------------------------
    # 2) Evaluate k using silhouette (on full data)
    # -----------------------------
    silhouettes = []
    print("\nComputing silhouette for AHC...")
    for k in K_RANGE:
        ahc = AgglomerativeClustering(
            n_clusters=k,
            linkage=LINKAGE_METHOD,
            metric=METRIC
        )
        labels = ahc.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        silhouettes.append(sil)
        print(f"K={k:2d}  silhouette={sil:.4f}")

    plt.figure()
    plt.plot(list(K_RANGE), silhouettes, marker="o")
    plt.title(f"AHC Silhouette vs K - {tag}")
    plt.xlabel("K")
    plt.ylabel("Silhouette")
    plt.tight_layout()
    plt.savefig(f"ahc_silhouette_{tag}.png", dpi=200)
    plt.close()
    print(f"Saved ahc_silhouette_{tag}.png")

    # -----------------------------
    # 3) Choose K (manual input)
    # -----------------------------
    chosen_k = int(input(f"\nEnter chosen K for AHC ({tag}): "))

    # -----------------------------
    # 4) Final clustering (full data)
    # -----------------------------
    final_ahc = AgglomerativeClustering(
        n_clusters=chosen_k,
        linkage=LINKAGE_METHOD,
        metric=METRIC
    )
    df["cluster"] = final_ahc.fit_predict(X_scaled)

    # -----------------------------
    # 5) Summary + save
    # -----------------------------
    summary = (
        df.groupby("cluster")
          .agg(
              n_skus=("item_code", "count"),
              mean_demand=("mean_demand", "mean"),
              mean_cv=("cv_demand", "mean"),
              mean_frequency=("demand_frequency", "mean"),
              mean_avg_affinity=("avg_affinity", "mean"),
              mean_max_affinity=("max_affinity", "mean")
          )
          .sort_values("mean_demand")
    )

    print("\nCluster Summary (AHC):")
    print(summary)

    df.to_csv(f"sku_ahc_clusters_{tag}.csv", index=False)
    summary.to_csv(f"ahc_cluster_summary_{tag}.csv")

    print(f"\nSaved sku_ahc_clusters_{tag}.csv")
    print(f"Saved ahc_cluster_summary_{tag}.csv")


def main():
    # cleaned
    run_ahc_pipeline(
        demand_file=Path("demand_features_cleaned.csv"),
        affinity_file=Path("sku_features_for_clustering_cleaned.csv"),
        tag="cleaned"
    )

    # no outlier
    run_ahc_pipeline(
        demand_file=Path("demand_features_no_outlier.csv"),
        affinity_file=Path("sku_features_for_clustering_no_outlier.csv"),
        tag="no_outlier"
    )


if __name__ == "__main__":
    main()