import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix, save_npz, triu


def read_csv_auto_sep(path: Path) -> pd.DataFrame:
    """Try ';' then ',' separator."""
    df = pd.read_csv(path, sep=";")
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=",")
    return df


def build_affinity_and_features(input_csv: Path, tag: str) -> None:
    """
    Build affinity matrix A (rho_ij), compute pair-bin counts,
    compute avg_affinity/max_affinity per SKU, and save outputs.
    """
    print(f"\n==============================")
    print(f"Processing: {input_csv}  (tag={tag})")
    print(f"==============================")

    df = read_csv_auto_sep(input_csv)

    needed = {"order_id", "item_code", "item_quantity"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{input_csv.name} missing columns: {missing}. Found: {list(df.columns)}")

    # Standardize types
    df["order_id"] = df["order_id"].astype(str)
    df["item_code"] = df["item_code"].astype(str)
    df["item_quantity"] = pd.to_numeric(df["item_quantity"], errors="coerce").fillna(0)

    # Demand per SKU (useful for clustering)
    demand = (
        df.groupby("item_code", as_index=False)["item_quantity"]
          .sum()
          .rename(columns={"item_quantity": "total_demand"})
    )

    # Build binary incidence matrix R[o,i] = 1 if order o contains sku i, else 0
    df_bin = df[["order_id", "item_code"]].drop_duplicates()

    order_codes, order_idx = np.unique(df_bin["order_id"].values, return_inverse=True)
    sku_codes, sku_idx = np.unique(df_bin["item_code"].values, return_inverse=True)

    n_orders = len(order_codes)
    n_skus = len(sku_codes)

    print(f"Rows: {len(df):,} | Orders: {n_orders:,} | SKUs: {n_skus:,}")

    R = csr_matrix(
        (np.ones(len(df_bin), dtype=np.int8), (order_idx, sku_idx)),
        shape=(n_orders, n_skus)
    )

    # Co-occurrence counts: C = R^T R
    # Affinity: A = C / |O|
    C = (R.T @ R).astype(np.float32)
    A = C * (1.0 / float(n_orders))
    A.setdiag(0)  # i != j
    A.eliminate_zeros()

    # Save sparse affinity matrix + sku index mapping
    out_npz = Path(f"affinity_{tag}.npz")
    out_index = Path(f"affinity_sku_index_{tag}.csv")
    save_npz(out_npz, A)
    pd.Series(sku_codes, name="item_code").to_csv(out_index, index=False)
    print(f"Saved: {out_npz}")
    print(f"Saved: {out_index}")

    # -----------------------------
    # Pair-bin counts (unique pairs i<j)
    # -----------------------------
    # Use upper triangle so each pair counted once
    A_upper = triu(A, k=1).tocoo()
    values = A_upper.data  # non-zero affinities for i<j

    total_pairs = n_skus * (n_skus - 1) // 2
    nonzero_pairs = len(values)
    zero_pairs = total_pairs - nonzero_pairs  # affinity == 0

    # Bins (interpreting "< 0,000" as zeros)
    # 2) 0.000 <= x < 0.003
    # 3) 0.003 <= x < 0.007
    # 4) 0.007 <= x < 0.010
    # 5) 0.010 <= x < 0.013
    b2 = np.sum((values >= 0.000) & (values < 0.003))
    b3 = np.sum((values >= 0.003) & (values < 0.007))
    b4 = np.sum((values >= 0.007) & (values < 0.010))
    b5 = np.sum((values >= 0.010) & (values < 0.013))
    b6 = np.sum(values >= 0.013)  # extra info (not requested, but helpful)

    bin_summary = pd.DataFrame([
        {"range": "< 0.000 (zero)", "pair_count": int(zero_pairs)},
        {"range": "0.000 - <0.003", "pair_count": int(b2)},
        {"range": "0.003 - <0.007", "pair_count": int(b3)},
        {"range": "0.007 - <0.010", "pair_count": int(b4)},
        {"range": "0.010 - <0.013", "pair_count": int(b5)},
        {"range": ">= 0.013 (extra)", "pair_count": int(b6)},
        {"range": "TOTAL unique pairs", "pair_count": int(total_pairs)},
        {"range": "NONZERO unique pairs", "pair_count": int(nonzero_pairs)},
    ])

    out_bins = Path(f"affinity_pair_bins_{tag}.csv")
    bin_summary.to_csv(out_bins, index=False)
    print(f"Saved: {out_bins}")

    # -----------------------------
    # Avg affinity & Max affinity per SKU (for clustering)
    # -----------------------------
    # avg_affinity includes zeros across all other SKUs:
    # avg_i = sum_j rho_ij / (n_skus - 1)
    row_sum = np.asarray(A.sum(axis=1)).ravel()
    avg_affinity = row_sum / max(n_skus - 1, 1)

    # max affinity per SKU
    max_affinity = np.asarray(A.max(axis=1).todense()).ravel()

    features = pd.DataFrame({
        "item_code": sku_codes,
        "avg_affinity": avg_affinity,
        "max_affinity": max_affinity,
    })

    # Merge demand so clustering has demand + affinity features
    features = features.merge(demand, on="item_code", how="left").fillna({"total_demand": 0})

    out_features = Path(f"sku_features_for_clustering_{tag}.csv")
    features.to_csv(out_features, index=False)
    print(f"Saved: {out_features}")

    # Console quick view
    print("\n--- Pair bin summary (top) ---")
    print(bin_summary.head(8).to_string(index=False))

    print("\n--- Feature preview ---")
    print(features.head(5).to_string(index=False))


def main():
    # Two inputs as requested
    build_affinity_and_features(Path("data_cleaned.csv"), tag="cleaned")
    build_affinity_and_features(Path("data_cleaned_no_outlier.csv"), tag="no_outlier")


if __name__ == "__main__":
    main()