# Universal Recommendation System

A multi-domain, content-based recommendation system that uses semantic embeddings and FAISS to recommend relevant items across movies, products, food, clothing, and videos. The project also includes a voice-assistant interface for interacting with the recommendation engine.

**Repository:** [Universal Recommendation System](https://github.com/Gaurav-coder22/Universal-Recommendation-System)

## Overview

Traditional recommendation systems often rely on exact keyword matching or user interaction history. This project explores semantic search to identify relevant items based on the meaning of a user's query and the textual information associated with each item.

It uses **Sentence Transformers** to convert text into numerical embeddings and **FAISS (Facebook AI Similarity Search)** to retrieve similar items efficiently.

For movie recommendations, an additional hybrid ranking approach combines semantic similarity with metadata-based signals such as genres, keywords, cast, director, and popularity.

## Key Features

- **Multi-domain recommendations:** Supports movies, products, food, clothing, and videos through a reusable recommendation architecture.
- **Semantic search:** Finds relevant items based on textual meaning rather than exact keyword matches alone.
- **Vector similarity search:** Uses FAISS to retrieve the nearest item embeddings.
- **Metadata-based recommendations:** Combines relevant item attributes to create richer text representations.
- **Hybrid movie ranking:** Combines semantic similarity with additional movie metadata signals.
- **Voice assistant:** Supports voice-based interaction through speech recognition and text-to-speech components.
- **Modular architecture:** Separates data preparation, embedding generation, indexing, recommendation logic, and assistant functionality.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Sentence Transformers | Generate semantic embeddings |
| `all-MiniLM-L6-v2` | Pretrained sentence-embedding model |
| FAISS | Efficient vector similarity search |
| Pandas | Data cleaning and preprocessing |
| NumPy | Numerical operations and vector processing |
| SpeechRecognition | Speech-to-text functionality |
| Piper TTS | Local text-to-speech functionality |
| iTunes Search API | External music search functionality |

## System Architecture

The project follows two main pipelines: an offline indexing pipeline and an online recommendation pipeline.

### 1. Offline indexing pipeline

```text
Item Datasets
     |
     v
Data Cleaning and Preprocessing
     |
     v
Combine Relevant Metadata
     |
     v
Sentence Transformer
(all-MiniLM-L6-v2)
     |
     v
Generate and Normalize Embeddings
     |
     v
Build FAISS Index
     |
     v
Save Index and Item Metadata
```

This pipeline prepares the item representations and vector indexes used during recommendation.

### 2. Online recommendation pipeline

```text
User Text or Voice Query
          |
          v
Speech-to-Text (voice input)
          |
          v
Domain and Query Identification
          |
          v
Generate Query Embedding
          |
          v
Search FAISS Index
          |
          v
Retrieve Similar Items
          |
          v
Hybrid Ranking (Movies)
          |
          v
Return Top-K Recommendations
```

For voice interactions, the assistant can convert the response into speech using a text-to-speech component.

## How It Works

### Step 1: Data preprocessing

The datasets are cleaned and relevant item attributes are combined into a single textual representation.

For example, a product representation might contain:

```text
Product Name: Running Shoes
Brand: Example Brand
Category: Sports Footwear
Description: Lightweight shoes for daily running
Specifications: Mesh material, black color
```

Combining useful metadata provides more context for semantic embedding generation.

### Step 2: Generate embeddings

The pretrained `all-MiniLM-L6-v2` model converts the combined text into a dense numerical vector.

Conceptually:

```text
Item Text
    |
    v
Sentence Transformer
    |
    v
Numerical Embedding
```

Similar textual meanings can produce embeddings that are close in the vector space.

### Step 3: Build a FAISS index

The generated embeddings are indexed using FAISS. The original item identifiers and metadata are retained so that retrieved vector IDs can be mapped back to the corresponding items.

### Step 4: Process a user query

When a user enters a natural-language query, the system identifies the relevant recommendation domain and encodes the query using the same embedding model.

For example:

```text
Query: "Suggest futuristic movies about space exploration"
```

The query is converted into an embedding and searched against the relevant FAISS index.

### Step 5: Retrieve recommendations

FAISS returns the nearest indexed vectors. The system uses their identifiers to retrieve the associated item information and produce the recommendations.

For movies, the retrieved candidates can undergo an additional hybrid-ranking step.

## Hybrid Movie Recommendation

For movie recommendations, the system combines multiple ranking signals rather than relying exclusively on semantic similarity.

The scoring approach uses the following weights:

| Signal | Weight |
|---|---:|
| Semantic similarity | 50% |
| Genre similarity | 20% |
| Keyword similarity | 10% |
| Cast similarity | 10% |
| Director similarity | 5% |
| Popularity | 5% |

The weighted score is calculated as:

```text
Final Score =
    0.50 × Semantic Similarity
  + 0.20 × Genre Similarity
  + 0.10 × Keyword Similarity
  + 0.10 × Cast Similarity
  + 0.05 × Director Similarity
  + 0.05 × Popularity
```

Jaccard similarity is used to compare set-based attributes such as genres, keywords, and cast members. The popularity component considers rating and vote count.

This approach aims to combine the semantic relevance of a movie with additional metadata signals when ordering the retrieved candidates.

## Project Structure

The repository contains the following important files and directories:

```text
Universal-Recommendation-System/
├── assistant.py
├── create_samples.py
├── preprocessing.ipynb
├── requirements.txt
├── sample_data/
└── src/
    ├── building_embeddings_faiss.py
    ├── clean_clothes_smartway.py
    ├── clean_products_smartway.py
    ├── download_anazon.py
    ├── prepare_myntra_products.py
    ├── prepare_tmdb_movies.py
    ├── preprcessing.py
    ├── spotify.py
    └── universal_recommender.py
```

**Important modules:**

- `assistant.py` — Handles the assistant interface, query routing, and voice-related functionality.
- `src/universal_recommender.py` — Contains the core recommendation and retrieval logic.
- `src/building_embeddings_faiss.py` — Contains functionality for preparing item representations, generating embeddings, and building FAISS indexes.
- `preprocessing.ipynb` — Provides a notebook for data exploration and preprocessing.

## Installation and Setup

### Prerequisites

- Python installed on your system
- Git
- Access to the datasets required for the domains you want to use

### 1. Clone the repository

```bash
git clone https://github.com/Gaurav-coder22/Universal-Recommendation-System.git
cd Universal-Recommendation-System
```

### 2. Create a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Some dependencies, particularly FAISS and audio-related packages, may require platform-specific installation steps.

### 4. Prepare the datasets and indexes

The datasets and generated FAISS indexes may not be included in the repository because of their size.

Before running recommendations, ensure that the required datasets are available and that the corresponding embeddings and indexes have been generated.

Review the data preparation scripts in `src/` and the configuration in `src/universal_recommender.py` to identify the required input files, output directories, and supported domains.

### 5. Run the assistant

After installing dependencies and preparing the required local assets, start the assistant using:

```bash
python assistant.py
```

**Note:** The assistant may require additional local configuration, speech-recognition access, Piper TTS assets, and domain-specific datasets or indexes. Check the relevant configuration and file paths before running it.

## Design Decisions

### Why Sentence Transformers?

Sentence Transformers creates semantic representations of text. This allows the system to retrieve items based on meaning and contextual similarity, rather than depending entirely on exact keyword matches.

### Why FAISS?

FAISS is designed for efficient similarity search over dense vectors. It provides a dedicated retrieval layer that can be scaled to larger embedding collections.

### Why content-based recommendation?

Content-based recommendation can operate without historical user-item interactions. It uses item descriptions and metadata to identify relevant results, making it useful when interaction data is unavailable or limited.

### Why a hybrid ranking approach for movies?

Semantic similarity alone may not capture all relevant attributes. Incorporating genres, keywords, cast, director, and popularity provides additional signals for ranking movie candidates.

## Limitations

- Recommendations depend on the quality and completeness of the available item metadata.
- The core recommendation approach is primarily content-based and does not currently implement conventional collaborative filtering based on user-item interaction history.
- Personalized recommendations based on long-term user preferences are not a core capability of the current implementation.
- New or updated items require their embeddings and indexes to be updated appropriately.
- Dataset availability, generated indexes, external API access, and local voice-assistant configuration can affect reproducibility.
- Recommendation quality should be assessed using a suitable evaluation dataset and ranking metrics before drawing quantitative conclusions.

## Future Improvements

- Add collaborative filtering and user-profile-based personalization.
- Evaluate recommendation quality using Precision@K, Recall@K, NDCG@K, and Mean Reciprocal Rank (MRR).
- Improve query understanding and domain classification.
- Support incremental updates to vector indexes.
- Expose the recommendation engine through a FastAPI REST API.
- Add caching, logging, and latency monitoring.
- Improve configuration management and portability for deployment.

## Learning Outcomes

Through this project, I explored:

- Semantic embeddings using pretrained transformer models
- Content-based recommendation techniques
- Vector indexing and similarity search with FAISS
- Text preprocessing and metadata engineering
- Hybrid ranking using multiple similarity signals
- Modular design for multi-domain retrieval
- Voice-based interaction with a Python recommendation engine

## Author

**Gaurav**

GitHub: [@Gaurav-coder22](https://github.com/Gaurav-coder22)

## License

Check the repository for the applicable license and usage terms.
