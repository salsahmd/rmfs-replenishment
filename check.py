import pandas as pd

# Load data
df = pd.read_csv("sku_sample.csv")

# Count number of SKUs per cluster
cluster_counts = df['cluster'].value_counts().sort_index()

print(cluster_counts)