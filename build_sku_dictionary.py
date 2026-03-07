import pandas as pd
import numpy as np
from pathlib import Path
from math import ceil, sqrt

# ==============================
# PARAMETERS
# ==============================
LEAD_TIME = 1
Z_VALUE = 1.2816  # service level 90%


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)
    return df


def compute_initial_inventory(mean_demand, std_demand):
    """
    item_initial_inventory = ceil(mu + z * sigma * sqrt(L))
    """
    mu = 0 if pd.isna(mean_demand) else float(mean_demand)
    sigma = 0 if pd.isna(std_demand) else float(std_demand)

    value = mu + Z_VALUE * sigma * sqrt(LEAD_TIME)
    return int(ceil(value))


def compute_box_inventory(items, items_per_box):
    """
    box_initial_inventory = ceil(item_initial_inventory / items_per_box)
    """
    if pd.isna(items_per_box) or items_per_box <= 0:
        return 0

    return int(ceil(items / items_per_box))


def main():

    # ==============================
    # FILE PATH
    # ==============================
    ITEMS_DICT_PATH = Path("items_dictionary_cleaned.csv")
    CLUSTER_PATH = Path("sku_kmeans_clusters_cleaned.csv")
    OUTPUT_PATH = Path("sku_dictionary.csv")

    # ==============================
    # LOAD DATA
    # ==============================
    items = pd.read_csv(ITEMS_DICT_PATH)
    clusters = pd.read_csv(CLUSTER_PATH)

    items = normalize_columns(items)
    clusters = normalize_columns(clusters)

    print("Items dictionary shape:", items.shape)
    print("Cluster data shape:", clusters.shape)

    # ==============================
    # SELECT ITEM COLUMNS
    # ==============================
    item_cols = [
        "item_code",
        "item_name",
        "box_length",
        "box_width",
        "box_height",
        "box_volume",
        "box_weight",
        "number_of_item_in_a_box"
    ]

    items_selected = items[item_cols].copy()

    # ==============================
    # MERGE DATA
    # ==============================
    sku_dictionary = pd.merge(
        items_selected,
        clusters,
        on="item_code",
        how="inner"
    )

    # ==============================
    # PREPARE std_demand
    # ==============================
    if "std_demand" not in sku_dictionary.columns:
        if {"cv_demand", "mean_demand"}.issubset(sku_dictionary.columns):
            sku_dictionary["std_demand"] = (
                sku_dictionary["cv_demand"] * sku_dictionary["mean_demand"]
            )
        else:
            raise ValueError(
                "std_demand tidak ada dan tidak bisa dihitung dari cv_demand."
            )

    # convert numeric
    sku_dictionary["mean_demand"] = pd.to_numeric(
        sku_dictionary["mean_demand"], errors="coerce"
    ).fillna(0)

    sku_dictionary["std_demand"] = pd.to_numeric(
        sku_dictionary["std_demand"], errors="coerce"
    ).fillna(0)

    sku_dictionary["number_of_item_in_a_box"] = pd.to_numeric(
        sku_dictionary["number_of_item_in_a_box"], errors="coerce"
    ).fillna(1)

    # ==============================
    # CALCULATE item_initial_inventory
    # ==============================
    sku_dictionary["item_initial_inventory"] = sku_dictionary.apply(
        lambda row: compute_initial_inventory(
            row["mean_demand"],
            row["std_demand"]
        ),
        axis=1
    )

    # ==============================
    # CALCULATE box_initial_inventory
    # ==============================
    sku_dictionary["box_initial_inventory"] = sku_dictionary.apply(
        lambda row: compute_box_inventory(
            row["item_initial_inventory"],
            row["number_of_item_in_a_box"]
        ),
        axis=1
    )

    # ==============================
    # SAVE RESULT
    # ==============================
    sku_dictionary.to_csv(OUTPUT_PATH, index=False)

    print("\nSKU dictionary created successfully")
    print("Rows:", len(sku_dictionary))
    print("Saved to:", OUTPUT_PATH.resolve())

    print("\nPreview:")
    print(
        sku_dictionary[
            [
                "item_code",
                "mean_demand",
                "std_demand",
                "item_initial_inventory",
                "number_of_item_in_a_box",
                "box_initial_inventory",
            ]
        ].head(10)
    )


if __name__ == "__main__":
    main()