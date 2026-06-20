import kagglehub
from pathlib import Path

print("Downloading Myntra dataset...")
path = Path(kagglehub.dataset_download("paramaggarwal/fashion-product-images-small"))

print("Dataset downloaded at:", path)
