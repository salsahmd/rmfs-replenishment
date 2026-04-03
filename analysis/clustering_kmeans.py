import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# ==============================
# 1. LOAD DATA
# ==============================
def load_data(file_path):
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
    print("Shape:", df.shape)
    return df


# ==============================
# 2. PREPARE FEATURES
# ==============================
def prepare_features(df, feature_columns):
    X = df[feature_columns].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled


# ==============================
# 3. EVALUATE K (Silhouette + Elbow)
# ==============================
def evaluate_k(X_scaled, min_k=2, max_k=8):
    silhouette_scores = {}
    inertias = {}

    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        silhouette_scores[k] = silhouette_score(X_scaled, labels)
        inertias[k] = model.inertia_
        print(f"k={k}, silhouette={silhouette_scores[k]:.4f}, inertia={inertias[k]:.2f}")

    ks = list(range(min_k, max_k + 1))
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(ks, [silhouette_scores[k] for k in ks], marker='o')
    ax1.set_title("Silhouette Score")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Score")

    ax2.plot(ks, [inertias[k] for k in ks], marker='o')
    ax2.set_title("Elbow Graph (Inertia)")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Inertia")

    plt.tight_layout()
    plt.show()


# ==============================
# 4. CLUSTERING
# ==============================
def perform_clustering(df, X_scaled, best_k):
    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    df["cluster"] = labels
    return df


# ==============================
# 5. MAIN PROGRAM
# ==============================
if __name__ == "__main__":

    # 🔹 CHANGE THIS TO YOUR CSV FILE
    file_path = os.path.join(os.path.dirname(__file__), "../sku_sample.csv")

    feature_columns = ["mean_demand", "cv_demand", "demand_frequency", "max_affinity"]

    df = load_data(file_path)
    X, X_scaled = prepare_features(df, feature_columns)

    evaluate_k(X_scaled, 2, 8)
    best_k = int(input("\nEnter number of clusters (k): "))

    df_clustered = perform_clustering(df, X_scaled, best_k)

    # Save cluster column back to sku_sample.csv
    df_clustered.to_csv(file_path, index=False)
    print("Cluster results saved to sku_sample.csv\n")

    # Display cluster summary
    summary = df_clustered.groupby("cluster").agg(
        num_sku=("item_code", "count"),
        mean_demand=("mean_demand", "mean"),
        mean_cv=("cv_demand", "mean"),
        mean_frequency=("demand_frequency", "mean"),
        mean_max_affinity=("max_affinity", "mean"),
    ).reset_index()

    print("Cluster Summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))