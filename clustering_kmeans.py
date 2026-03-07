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
# 3. FIND BEST K (Silhouette)
# ==============================
def find_best_k(X_scaled, min_k=2, max_k=8):
    best_k = 2
    best_score = -1
    scores = {}

    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)

        score = silhouette_score(X_scaled, labels)
        scores[k] = score
        print(f"k={k}, silhouette score={score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    print("\nBest k:", best_k)
    return best_k


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
    file_path = "data.csv"

    # 🔹 CHANGE TO YOUR FEATURE COLUMNS
    feature_columns = ["feature1", "feature2"]

    df = load_data(file_path)
    X, X_scaled = prepare_features(df, feature_columns)

    best_k = find_best_k(X_scaled, 2, 8)

    df_clustered = perform_clustering(df, X_scaled, best_k)

    df_clustered.to_csv("clustered_output.csv", index=False)
    print("Clustering finished. Saved as clustered_output.csv")

    # Optional: plot first 2 features
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df_clustered["cluster"])
    plt.title("Clustering Result")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()