# prepare_tmdb_movies_fixed.py
import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
import kagglehub
from pathlib import Path

tqdm.pandas()

print("\n📥 Downloading TMDB dataset...")
path = Path(kagglehub.dataset_download("rounakbanik/the-movies-dataset"))
print("📁 Downloaded to:", path)

movies = pd.read_csv(path / "movies_metadata.csv", low_memory=False)
keywords = pd.read_csv(path / "keywords.csv")
credits = pd.read_csv(path / "credits.csv")

# IDs to string
movies["id"] = movies["id"].astype(str)
keywords["id"] = keywords["id"].astype(str)
credits["id"] = credits["id"].astype(str)

# Merge
movies = movies.merge(keywords, on="id", how="left")
movies = movies.merge(credits, on="id", how="left")

# Safe JSON loads
def safe_load(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

json_cols = ["genres", "keywords", "cast", "crew"]
for col in json_cols:
    movies[col] = movies[col].progress_apply(safe_load)

# Extract metadata
def extract_names(list_dict, key="name", top=None):
    names = [d.get(key, "") for d in list_dict if isinstance(d, dict)]
    if top:
        return names[:top]
    return names

def extract_director(crew_list):
    for c in crew_list:
        if c.get("job") == "Director":
            return c.get("name", "")
    return ""

movies["genre_names"] = movies["genres"].apply(lambda x: extract_names(x))
movies["keyword_names"] = movies["keywords"].apply(lambda x: extract_names(x, top=10))
movies["cast_names"] = movies["cast"].apply(lambda x: extract_names(x, top=3))
movies["director"] = movies["crew"].apply(extract_director)

# Popularity proxy
movies["vote_average"] = pd.to_numeric(movies["vote_average"], errors="coerce").fillna(0)
movies["vote_count"] = pd.to_numeric(movies["vote_count"], errors="coerce").fillna(0)

def safe_str(x):
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)

def combine_row(row):
    return " ".join([
        safe_str(row.get("overview", "")) * 2,
        " ".join(row.get("genre_names", [])),
        (" ".join(row.get("keyword_names", []))) * 2,
        " ".join(row.get("cast_names", [])),
        safe_str(row.get("director", "")),
        safe_str(row.get("tagline", "")),
        safe_str(row.get("title", ""))
    ])


movies["combined"] = movies.progress_apply(combine_row, axis=1)

movies = movies[
    movies["title"].notna() &
    movies["combined"].str.strip().ne("")
]

movies.to_csv("movies_clean.csv", index=False, encoding="utf-8")
print("\n🎉 Saved as: movies_clean.csv")
