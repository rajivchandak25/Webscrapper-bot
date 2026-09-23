"""
Retriever Module: Dense Vector Retrieval & Semantic Reranking
-------------------------------------------------------------
Uses a lightweight bi-encoder transformer (all-MiniLM-L6-v2) to map
user queries and text chunks into a shared dense vector embedding space.
Computes cosine similarity to rerank candidate chunks by semantic relevance.
"""

from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer, util
import torch


class SemanticReranker:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the sentence transformer bi-encoder.
        """
        self.embedder = SentenceTransformer(model_name)

    def rerank_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 3,
        similarity_threshold: float = 0.15
    ) -> List[Dict[str, Any]]:
        """
        Reranks text chunks based on semantic similarity to the query.

        Args:
            query (str): The user's input query.
            chunks (list): List of chunk dictionaries containing 'text'.
            top_k (int): Number of top results to return.
            similarity_threshold (float): Minimum cosine similarity required.

        Returns:
            List[Dict[str, Any]]: Top-k relevant chunks sorted by similarity score.
        """
        if not chunks:
            return []

        # Extract texts for batch embedding
        chunk_texts = [c["text"] for c in chunks]

        # Generate dense embeddings
        query_emb = self.embedder.encode(query, convert_to_tensor=True)
        chunk_embs = self.embedder.encode(chunk_texts, convert_to_tensor=True)

        # Compute cosine similarities
        cosine_scores = util.pytorch_cos_sim(query_emb, chunk_embs)[0]

        # Attach score and sort
        scored_chunks = []
        for i, score in enumerate(cosine_scores):
            chunk_copy = dict(chunks[i])
            chunk_copy["similarity_score"] = round(float(score.item()), 4)
            scored_chunks.append(chunk_copy)

        # Sort descending by similarity score
        scored_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Filter by threshold
        filtered_chunks = [c for c in scored_chunks if c["similarity_score"] >= similarity_threshold]

        # Return top_k
        return filtered_chunks[:top_k]
