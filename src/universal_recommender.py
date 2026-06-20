# universal_recommender.py
import faiss
import numpy as np
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pandas as pd
from difflib import get_close_matches
import math
import requests

MODEL_NAME = "all-MiniLM-L6-v2"
USE_CUDA = False

# Load model
model = SentenceTransformer(MODEL_NAME, device="cuda" if USE_CUDA else "cpu")

BASE = Path("faiss_data")

# Mapping domain to their CSV files
CSV_PATHS = {
    "movies": "movies_clean.csv",
    "food": "food_clean.csv",
    "products": "products_clean.csv",
    "clothes": "clothes_clean.csv",
    "videos": "videos_clean.csv"
}

# Cache metadata to avoid reloading
METADATA = {}


def load_metadata(domain):
    if domain in METADATA:
        return METADATA[domain]
    df = pd.read_csv(CSV_PATHS[domain])
    METADATA[domain] = df
    return df


def get_item(domain, item_id):
    df = load_metadata(domain)

    for col in ["id", "uniq_id", "pid", "RecipeId", "video_id"]:
        if col in df.columns:
            id_col = col
            break

    row = df[df[id_col].astype(str) == str(item_id)]
    if row.empty:
        return {"title": "Unknown"}

    row = row.iloc[0]

    for name_field in ["title", "Name", "product_name", "recipeName", "channel_title", "name"]:
        if name_field in df.columns:
            return {"title": str(row[name_field])}

    return {"title": "Unknown"}


def load_domain(domain):
    domain_path = BASE / domain

    embeddings = np.load(str(domain_path / f"{domain}_embeddings.npy")).astype("float32")
    index = faiss.read_index(str(domain_path / f"{domain}_faiss.index"))

    with open(domain_path / f"{domain}_ids.json", "r", encoding="utf8") as f:
        ids = json.load(f)

    return embeddings, index, ids


def recommend(domain, query, k=5):
    embeddings, index, ids = load_domain(domain)
    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    distances, indices = index.search(query_vec, k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        item_id = ids[idx]
        item = get_item(domain, item_id)
        item["score"] = float(score)
        results.append(item)

    return results


# --------------------------- UTILS ------------------------------

def jaccard_similarity(a, b):
    if not isinstance(a, list) or not isinstance(b, list):
        return 0.0
    if len(a) == 0 or len(b) == 0:
        return 0.0
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def popularity_score(row):
    try:
        votes = float(row.get("vote_count", 0))
        rating = float(row.get("vote_average", 0))
        return rating * math.log1p(votes)
    except:
        return 0


# --------------------------- MOVIE HYBRID ------------------------------

def hybrid_recommend_movie(title, k=5):
    df = load_metadata("movies")
    name_col = "title"

    match = get_close_matches(title, df[name_col].astype(str).tolist(), n=1, cutoff=0.3)
    if not match:
        return f"Movie '{title}' not found."

    matched_title = match[0]
    base = df[df[name_col] == matched_title].iloc[0]

    print(f"\nFound: {matched_title}")
    print("Recommending hybrid suggestions...\n")

    semantic_results = recommend("movies", base["combined"], k=50)
    hybrid = []

    for item in semantic_results:
        movie_title = item["title"]
        row = df[df[name_col] == movie_title]

        if row.empty:
            continue
        row = row.iloc[0]

        g_sim = jaccard_similarity(base["genre_names"], row["genre_names"])
        k_sim = jaccard_similarity(base["keyword_names"], row["keyword_names"])
        c_sim = jaccard_similarity(base["cast_names"], row["cast_names"])
        d_sim = 1.0 if base["director"] == row["director"] else 0.0
        p_sim = popularity_score(row)

        final_score = (
            0.50 * item["score"] +
            0.20 * g_sim +
            0.10 * k_sim +
            0.10 * c_sim +
            0.05 * d_sim +
            0.05 * p_sim
        )

        hybrid.append({"title": movie_title, "score": final_score})

    hybrid = [h for h in hybrid if h["title"].lower() != matched_title.lower()]
    hybrid.sort(key=lambda x: x["score"], reverse=True)
    return hybrid[:k]


# --------------------------- GENERIC NAME SEARCH ------------------------

def recommend_by_item_name(domain, item_name, k=5):
    df = load_metadata(domain)

    name_cols = ["title", "Name", "product_name", "recipeName", "channel_title", "name"]
    name_col = next((c for c in name_cols if c in df.columns), None)

    if name_col is None:
        return "No title column found."

    if "combined" not in df.columns:
        return "Dataset missing 'combined' column."

    match = get_close_matches(item_name, df[name_col].astype(str).tolist(), n=1, cutoff=0.3)
    if not match:
        return f"No item like '{item_name}' found."

    matched_name = match[0]
    row = df[df[name_col] == matched_name].iloc[0]

    print(f"\nFound: {matched_name}")
    print("Recommending similar items...\n")

    query_text = row["combined"]
    results = recommend(domain, query_text, k + 1)

    cleaned = []
    for item in results:
        if item["title"].lower() != matched_name.lower():
            cleaned.append(item)
        if len(cleaned) == k:
            break

    return cleaned


# --------------------------- MUSIC (iTunes API) ------------------------------

def search_song_itunes(query):
    url = f"https://itunes.apple.com/search?term={query}&entity=song&limit=1"
    data = requests.get(url).json()
    if data["resultCount"] == 0:
        return None
    return data["results"][0]


def recommend_from_artist(artist_name, k=5):
    url = f"https://itunes.apple.com/search?term={artist_name}&entity=song&limit={k}"
    data = requests.get(url).json()
    if data["resultCount"] == 0:
        return []
    return data["results"]


def recommend_music_itunes(song_name, k=5):
    song = search_song_itunes(song_name)
    if not song:
        print("\n❌ Song not found on iTunes.")
        return []

    track = song["trackName"]
    artist = song["artistName"]

    print(f"\n🎵 Found: {track} — {artist}")
    print("\n🔍 Searching similar songs...\n")

    similar = recommend_from_artist(artist, k)

    results = []
    for item in similar:
        results.append({
            "title": item["trackName"],
            "score": 1.0
        })
        print("-", item["trackName"], "-", item["artistName"])

    return results


# --------------------------- INTERACTIVE MODE ------------------------------

if __name__ == "__main__":
    print("\n=== Universal Hybrid Recommender ===")

    valid = ["movies", "food", "clothes", "products", "videos", "music"]
    domain = input("\nChoose category (movies, food, clothes, products, videos, music): ").strip().lower()

    if domain not in valid:
        print("Invalid domain!")
        exit()

    query = input("\nEnter your query: ").strip()
    k = int(input("\nHow many results? (default 5): ") or 5)

    print("\nSearching...\n")

    # MOVIES
    if domain == "movies":
        results = hybrid_recommend_movie(query, k)
        if isinstance(results, str):
            print(results)
            print("\nFallback to semantic search...\n")
            results = recommend("movies", query, k)

    # MUSIC (iTunes API)
    elif domain == "music":
        results = recommend_music_itunes(query, k)

    # OTHER DOMAINS (FAISS)
    else:
        results = recommend_by_item_name(domain, query, k)
        if isinstance(results, str):
            print(results)
            results = recommend(domain, query, k)

    print("\nResults:\n")
    for i, item in enumerate(results, 1):
        print(f"{i}. {item['title']} ({item['score']})")
