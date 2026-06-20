import pandas as pd
import os

os.makedirs("sample_data", exist_ok=True)

files = {
    "movies": "data/movies_clean.csv",
    "food": "data/food_clean.csv",
    "products": "data/products_clean.csv"
}

for name, path in files.items():
    df = pd.read_csv(path)
    sample_size = min(200, len(df))
    df.sample(sample_size, random_state=42).to_csv(
        f"sample_data/{name}_sample.csv",
        index=False
    )

print("Sample datasets created!")