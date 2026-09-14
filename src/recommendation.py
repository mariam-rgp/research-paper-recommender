import pandas as pd
import numpy as np

from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Load dataset
df = pd.read_csv(
    PROJECT_ROOT / "data" / "arxiv_cleaned.csv"
)


# Load transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Load precomputed paper embeddings
embeddings = np.load(
    PROJECT_ROOT / "models" / "transformer_embeddings.npy"
)



def recommend_from_query(query, top_k=10):


    # Validate query
    if not isinstance(query, str) or not query.strip():
        raise ValueError(
            "Query must be a non-empty string."
        )

    # Validate top_k
    if not isinstance(top_k, int):
        raise ValueError(
            "top_k must be an integer."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0."
        )

    if top_k > len(df):
        raise ValueError(
            f"top_k cannot be greater than the number of papers ({len(df)})."
        )

    # Convert query into embedding
    query_embedding = model.encode([query])

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        query_embedding,
        embeddings
    ).flatten()

    # Get top-k most similar papers
    top_indices = similarity_scores.argsort()[::-1][:top_k]

    # Build recommendations DataFrame
    recommendations = df.iloc[top_indices][
        ["titles", "abstracts", "terms"]
    ].copy()
    recommendations["rank"] = range(1, top_k + 1)


    # Add similarity scores
    recommendations["similarity_score"] = (
        similarity_scores[top_indices]
    )

    return recommendations

def get_paper_by_rank(recommendations, rank):

    if not isinstance(rank, int):
        raise ValueError(
            "rank must be an integer."
        )

    if rank <= 0 or rank > len(recommendations):
        raise ValueError(
            f"rank must be between 1 and {len(recommendations)}."
        )

    paper = recommendations[
        recommendations["rank"] == rank
    ].iloc[0]

    return paper