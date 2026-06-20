import pandas as pd
import numpy as np

print("\n📄 Loading products dataset...")
df = pd.read_csv("clothes_clean.csv")

# ---- Standardize missing numeric values ----
numeric_cols = ["overall_rating", "product_rating", "discounted_price", "retail_price"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# ---- Standardize text fields ----
df["product_name"] = df["product_name"].astype(str).str.strip().str.lower()
df["brand"] = df["brand"].astype(str).str.strip().str.lower()
df["description"] = df["description"].astype(str)

# ---- Create composite key: product name + brand ----
df["key"] = df["product_name"] + "_--_" + df["brand"]

print("\n🔍 Total unique product groups:", df["key"].nunique())

# ---- SMART SELECTION FUNCTION ----
def choose_best(group):
    # 1. Highest rating
    group = group.sort_values(by="overall_rating", ascending=False)

    # 2. If rating ties → lowest discounted price
    group = group.sort_values(by="discounted_price", ascending=True)

    # 3. If still tied → longest description
    group = group.sort_values(by=df["description"].str.len(), ascending=False)

    # 4. Return the best product
    return group.iloc[0]

print("\n⚙️ Performing smart deduplication... (this may take a few seconds)")
cleaned = df.groupby("key", as_index=False).apply(choose_best)

# Remove the grouping index added by pandas
cleaned.reset_index(drop=True, inplace=True)

print("\n✅ DONE!")
print("Original rows:", len(df))
print("Cleaned rows:", len(cleaned))
print("Duplicates removed:", len(df) - len(cleaned))

# ---- Save cleaned dataset ----
cleaned.to_csv("products_clean_smart.csv", index=False)
print("\n💾 Saved cleaned dataset as: products_clean_smart.csv")