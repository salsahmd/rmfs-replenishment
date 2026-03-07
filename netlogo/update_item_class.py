"""
Update items_dictionary.csv to use cluster labels from sku_sample.csv
instead of the old A/B/C item_class classification.

Clusters: 0, 1, 2, 3, 4 (from kmeans clustering)

Items present in sku_sample.csv get their cluster value.
Items NOT in sku_sample.csv are assigned a default cluster (3 = largest cluster).
"""

import os
import pandas as pd

def update_item_class_from_clusters(
    items_dict_path="items_dictionary.csv",
    sku_sample_path="../sku_sample.csv",
    default_cluster=3
):
    """
    Replace the item_class column in items_dictionary.csv with cluster
    labels from sku_sample.csv.
    
    Args:
        items_dict_path: Path to items_dictionary.csv
        sku_sample_path: Path to sku_sample.csv with cluster column
        default_cluster: Default cluster for items not in sku_sample
    """
    # Load data
    items_dict = pd.read_csv(items_dict_path)
    sku_sample = pd.read_csv(sku_sample_path)
    
    # Create mapping from item_code -> cluster
    cluster_map = sku_sample.set_index("item_code")["cluster"].to_dict()
    
    # Backup old item_class
    items_dict["item_class_old"] = items_dict["item_class"]
    
    # Map cluster values to item_class column
    items_dict["item_class"] = items_dict["item_code"].map(cluster_map).fillna(default_cluster).astype(int)
    
    # Save updated file
    items_dict.to_csv(items_dict_path, index=False)
    
    # Print summary
    print("Updated item_class in items_dictionary.csv:")
    print(items_dict["item_class"].value_counts().sort_index())
    print(f"\nItems mapped from sku_sample: {items_dict['item_code'].isin(cluster_map).sum()}")
    print(f"Items assigned default cluster ({default_cluster}): {(~items_dict['item_code'].isin(cluster_map)).sum()}")
    
    return items_dict


if __name__ == "__main__":
    # Run from the netlogo/ directory
    update_item_class_from_clusters()
