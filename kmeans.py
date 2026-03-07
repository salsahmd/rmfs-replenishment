import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


RANDOM_STATE = 42
N_INIT = 20
K_RANGE = range(2, 16)


def run_kmeans_pipeline(demand_file: Path,
                        affinity_file: Path,
                        tag: str):
    print(f"\n==============================")
    print(f"Clustering for: {tag}")
    print(f"==============================")

    # -----------------------------
    # Load data
    # -----------------------------
    demand = pd.read_csv(demand_file)
    affinity = pd.read_csv(affinity_file)

    # Merge on item_code
    df = demand.merge(
        affinity[["item_code", "avg_affinity", "max_affinity"]],
        on="item_code",
        how="inner"
    )

    print("Total SKUs used:", len(df))

    # -----------------------------
    # Select features
    # -----------------------------
    feature_cols = [
        "avg_affinity",
        "max_affinity",
        "mean_demand",
        "cv_demand",
        "demand_frequency"
    ]

    X = df[feature_cols].copy()

    # Replace inf values if any (CV might produce inf)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # -----------------------------
    # Normalization (MinMax)
    # -----------------------------
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # Elbow Method
    # -----------------------------
    print("\nComputing WCSS (Elbow)...")

    wcss = []
    silhouette_scores = []

    for k in K_RANGE:
        km = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=N_INIT
        )
        labels = km.fit_predict(X_scaled)

        wcss.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels)
        silhouette_scores.append(sil)

        print(f"K={k:2d}  WCSS={km.inertia_:.4f}  Silhouette={sil:.4f}")

    # Plot Elbow
    plt.figure()
    plt.plot(list(K_RANGE), wcss, marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS")
    plt.title(f"Elbow Method ({tag})")
    plt.tight_layout()
    plt.savefig(f"elbow_{tag}.png", dpi=200)
    plt.close()

    # Plot Silhouette
    plt.figure()
    plt.plot(list(K_RANGE), silhouette_scores, marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title(f"Silhouette Score ({tag})")
    plt.tight_layout()
    plt.savefig(f"silhouette_{tag}.png", dpi=200)
    plt.close()

    print(f"\nSaved elbow_{tag}.png")
    print(f"Saved silhouette_{tag}.png")

    # -----------------------------
    # Choose K (manual selection recommended)
    # -----------------------------
    chosen_k = int(input("\nEnter chosen K based on elbow/silhouette: "))

    final_km = KMeans(
        n_clusters=chosen_k,
        random_state=RANDOM_STATE,
        n_init=N_INIT
    )

    df["cluster"] = final_km.fit_predict(X_scaled)

    # -----------------------------
    # Cluster summary
    # -----------------------------
    cluster_summary = (
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

    print("\nCluster Summary:")
    print(cluster_summary)

    # Save results
    df.to_csv(f"sku_kmeans_clusters_{tag}.csv", index=False)
    cluster_summary.to_csv(f"cluster_summary_{tag}.csv")

    print(f"\nSaved sku_kmeans_clusters_{tag}.csv")
    print(f"Saved cluster_summary_{tag}.csv")


def main():
    # CLEANED version
    run_kmeans_pipeline(
        demand_file=Path("demand_features_cleaned.csv"),
        affinity_file=Path("sku_features_for_clustering_cleaned.csv"),
        tag="cleaned"
    )

    # NO OUTLIER version
    run_kmeans_pipeline(
        demand_file=Path("demand_features_no_outlier.csv"),
        affinity_file=Path("sku_features_for_clustering_no_outlier.csv"),
        tag="no_outlier"
    )


if __name__ == "__main__":
    main()