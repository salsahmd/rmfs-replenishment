import pandas as pd

# 1️⃣ Load data (semicolon separated)
df = pd.read_csv("raw_order.csv", sep=';')

# 2️⃣ Ensure correct data types
df['item_code'] = df['item_code'].astype(str)
df['item_quantity'] = pd.to_numeric(df['item_quantity'], errors='coerce').fillna(0)

# 3️⃣ Calculate total demand per SKU
total_demand = (
    df.groupby('item_code')['item_quantity']
      .sum()
      .reset_index()
      .sort_values(by='item_quantity', ascending=False)
)

# 4️⃣ Print result
print(total_demand.head())

# 5️⃣ Save to CSV (recommended)
total_demand.to_csv("total_demand.csv", index=False)

print("Total demand calculation completed.")