import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

df = pd.read_csv("styles.csv", on_bad_lines='skip')

# Keep only useful columns
df = df[[
    "id", 
    "productDisplayName",
    "brand",
    "gender",
    "masterCategory",
    "subCategory",
    "articleType",
    "baseColour",
    "season",
    "year",
    "usage",
    "price",
    "description"
]]

# Drop rows without title
df = df.dropna(subset=["productDisplayName"])

# Clean description
df["description"] = df["description"].astype(str).str.replace("\n", " ")

# Build combined column
def combine(row):
    parts = [
        str(row["productDisplayName"]),
        str(row["brand"]),
        str(row["gender"]),
        str(row["masterCategory"]),
        str(row["subCategory"]),
        str(row["articleType"]),
        str(row["baseColour"]),
        str(row["season"]),
        str(row["usage"]),
        f"price {row['price']}",
        str(row["description"])
    ]
    return " ".join([p for p in parts if p and p.strip()])

df["combined"] = df.apply(combine, axis=1)

# Remove duplicates (keep one per name + brand)
df = df.drop_duplicates(subset=["productDisplayName", "brand"], keep="first")

# Remove empty rows
df = df[df["combined"].str.strip() != ""]

df.to_csv("myntra_clean.csv", index=False)
print("Saved myntra_clean.csv")
