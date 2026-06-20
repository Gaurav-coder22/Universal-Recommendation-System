# build_embeddings_faiss.py
import os
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


DATA_DIR = Path(".")
OUT_DIR = Path("faiss_data")
OUT_DIR.mkdir(exist_ok=True)

DATASETS = [
    ("clothes", "clothes_clean.csv", "uniq_id")   # FIXED ID COLUMN
]

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 256
USE_CUDA = False

import ast

def smart_combine(row):
    parts = []

    # Product name + brand
    parts.append(str(row.get("product_name", "")))
    parts.append(str(row.get("brand", "")))

    # Description
    desc = row.get("description", "")
    if isinstance(desc, str):
        parts.append(desc)

    # Category tree
    cat = row.get("product_category_tree", "")
    if isinstance(cat, str):
        cat = cat.replace(">>", " ").replace(",", " ")
        parts.append(cat)

    # Product specifications
    specs = row.get("product_specifications", "")
    try:
        specs = specs.replace("=>", ":")
        json_obj = ast.literal_eval(specs)
        for spec in json_obj.get("product_specification", []):
            key = spec.get("key", "")
            value = spec.get("value", "")
            parts.append(f"{key} {value}")
    except:
        pass

    # Join all
    return " ".join([p for p in parts if isinstance(p, str) and p.strip()])


def load_csv(path):
    df = pd.read_csv(path)

    # Build "combined" column if missing
    if "combined" not in df.columns:
        print("🛠 Adding smart combined column...")
        df["combined"] = df.apply(smart_combine, axis=1)

    # Drop empty rows
    df = df[df["combined"].astype(str).str.strip() != ""]
    return df


def prepare_text_column(df):
    return df["combined"].astype(str).tolist()



print("Loading SentenceTransformer model:", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME, device="cuda" if USE_CUDA else "cpu")



for domain, filename, id_col in DATASETS:
    csv_path = DATA_DIR / filename

    if not csv_path.exists():
        print(f"[SKIP] {csv_path} missing")
        continue

    print(f"\n=== Processing: {domain} ===")
    df = load_csv(csv_path)
    print("Rows:", len(df))

    # ID column exists?
    if id_col not in df.columns:
        raise ValueError(f"❌ ID column '{id_col}' missing in dataset!")

    ids = df[id_col].astype(str).tolist()
    texts = prepare_text_column(df)

    # ---------------------------------------------
    # ENCODING
    # ---------------------------------------------
    embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"Encoding {domain}"):
        batch = texts[i: i + BATCH_SIZE]
        emb = model.encode(
            batch,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        embeddings.append(emb)

    embeddings = np.vstack(embeddings).astype("float32")
    print(f"Embedding shape: {embeddings.shape}")

    # ---------------------------------------------
    # FAISS INDEX
    # ---------------------------------------------
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print("FAISS index ntotal:", index.ntotal)

    # ---------------------------------------------
    # SAVE DATA
    # ---------------------------------------------
    domain_dir = OUT_DIR / domain
    domain_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(domain_dir / f"{domain}_embeddings.npy"), embeddings)
    faiss.write_index(index, str(domain_dir / f"{domain}_faiss.index"))

    with open(domain_dir / f"{domain}_ids.json", "w", encoding="utf8") as f:
        json.dump(ids, f)

    print("✔ Saved FAISS index, embeddings, and IDs")


print("\n🎉 ALL DONE — Clothes embeddings rebuilt successfully!\n")
