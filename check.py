import pandas as pd

config = pd.read_csv("items_slots_configuration.csv")

print("Total rows:", len(config))
print("Unique SKU in config:", config["item_code"].nunique())