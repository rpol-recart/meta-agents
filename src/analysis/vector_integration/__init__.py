"""
Vector Integration Service for Project Analysis Tool

This module provides semantic embedding generation and vector database
integration for similarity search and clustering operations.
"""

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""

    text: str
    embedding: list[float]
    model_name: str
    dimension: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "embedding": self.embedding,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class SearchResult:
    """Result of similarity search."""

    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class ClusterResult:
    """Result of clustering operation."""

    cluster_id: int
    members: list[str]
    centroid: list[float] | None = None
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "member_count": len(self.members),
            "members": self.members[:10],
            "centroid": self.centroid,
            "keywords": self.keywords,
        }


class EmbeddingService:
    """
    Service for generating semantic embeddings.

    This class provides:
    - Sentence embedding generation
    - Batch processing capabilities
    - Caching for improved performance
    - Multiple embedding model support
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the embedding service.

        Args:
            config: Configuration dictionary with model settings
        """
        self.config = config or {}
        self.model_name = self.config.get("model_name", "all-MiniLM-L6-v2")
        self.cache: dict[str, list[float]] = {}
        self.cache_size = self.config.get("cache_size", 10000)
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the embedding model."""
        self._model = None
        self._dimension = 384

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Initialized embedding model: {self.model_name} (dim: {self._dimension})")
        except ImportError:
            logger.warning("sentence-transformers not available, using fallback embedding")
            self._dimension = self.config.get("fallback_dimension", 384)

    def embed_text(self, text: str, metadata: dict[str, Any] | None = None) -> EmbeddingResult:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed
            metadata: Optional metadata to include with result

        Returns:
            EmbeddingResult with embedding vector
        """
        cache_key = text[:100]

        if cache_key in self.cache:
            return EmbeddingResult(
                text=text,
                embedding=self.cache[cache_key],
                model_name=self.model_name,
                dimension=self._dimension,
                metadata={"source": "cache", **(metadata or {})},
            )

        if self._model is not None:
            embedding = self._model.encode(text).tolist()
        else:
            embedding = self._generate_fallback_embedding(text)

        if len(self.cache) < self.cache_size:
            self.cache[cache_key] = embedding

        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model_name=self.model_name,
            dimension=self._dimension,
            metadata=metadata or {},
        )

    def embed_texts(
        self,
        texts: list[str],
        metadata: dict[str, Any] | None = None,
        show_progress: bool = False,
    ) -> list[EmbeddingResult]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts to embed
            metadata: Optional metadata to include with results
            show_progress: Whether to show progress indicator

        Returns:
            List of EmbeddingResult objects
        """
        results = []

        if self._model is not None:
            embeddings = self._model.encode(texts, show_progress_bar=show_progress)

            for text, embedding in zip(texts, embeddings):
                embedding_list = embedding.tolist()
                cache_key = text[:100]

                if len(self.cache) < self.cache_size:
                    self.cache[cache_key] = embedding_list

                results.append(
                    EmbeddingResult(
                        text=text,
                        embedding=embedding_list,
                        model_name=self.model_name,
                        dimension=self._dimension,
                        metadata=metadata or {},
                    )
                )
        else:
            for text in texts:
                embedding = self._generate_fallback_embedding(text)
                results.append(self.embed_text(text, metadata))

        return results

    def _generate_fallback_embedding(self, text: str) -> list[float]:
        """Generate a simple fallback embedding based on word frequencies."""
        words = text.lower().split()
        word_freq = defaultdict(float)

        for i, word in enumerate(words):
            weight = 1.0 / (i + 1)
            word_freq[word] += weight

        embedding = []
        for i in range(self._dimension):
            if i < len(words):
                embedding.append(word_freq.get(words[i], 0.0))
            else:
                embedding.append(0.0)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [v / norm for v in embedding]

        return embedding

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.cache.clear()
        logger.info("Embedding cache cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get statistics about the embedding cache."""
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_size,
            "hit_rate": getattr(self, "_cache_hits", 0)
            / max(1, getattr(self, "_cache_requests", 1)),
        }


class VectorStore:
    """
    In-memory vector store for similarity search.

    This class provides:
    - Vector storage and indexing
    - Similarity search (cosine similarity)
    - Basic clustering capabilities
    - Hybrid retrieval combining with graph traversal
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the vector store.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.vectors: dict[str, list[float]] = {}
        self.metadata: dict[str, dict[str, Any]] = {}
        self._index: np.ndarray | None = None
        self._vector_matrix: np.ndarray | None = None
        self._needs_reindex = True

    def add(
        self, id: str, embedding: list[float], metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Add a vector to the store.

        Args:
            id: Unique identifier for the vector
            embedding: Embedding vector
            metadata: Optional metadata
        """
        self.vectors[id] = np.array(embedding, dtype=np.float32)
        self.metadata[id] = metadata or {}
        self._needs_reindex = True

    def add_batch(self, items: list[tuple[str, list[float], dict[str, Any] | None]]) -> None:
        """
        Add multiple vectors to the store.

        Args:
            items: List of (id, embedding, metadata) tuples
        """
        for id, embedding, metadata in items:
            self.add(id, embedding, metadata)

    def search(
        self, query_embedding: list[float], top_k: int = 10, exclude_ids: list[str] | None = None
    ) -> list[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            exclude_ids: IDs to exclude from results

        Returns:
            List of SearchResult objects sorted by similarity
        """
        if not self.vectors:
            return []

        self._ensure_index()

        query = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        scores = np.dot(self._vector_matrix, query.T).flatten()
        norms = np.linalg.norm(self._vector_matrix, axis=1) * np.linalg.norm(query)
        similarities = scores / (norms + 1e-10)

        exclude_set = set(exclude_ids or [])
        results = []

        sorted_indices = np.argsort(similarities)[::-1]

        for idx in sorted_indices:
            if len(results) >= top_k:
                break

            id = list(self.vectors.keys())[idx]
            if id in exclude_set:
                continue

            results.append(
                SearchResult(
                    text=self.metadata[id].get("text", id),
                    score=float(similarities[idx]),
                    metadata=self.metadata[id],
                )
            )

        return results

    def _ensure_index(self) -> None:
        """Ensure the index is built and up to date."""
        if self._needs_reindex and self.vectors:
            ids = list(self.vectors.keys())
            self._vector_matrix = np.array([self.vectors[id] for id in ids], dtype=np.float32)
            self._index = ids
            self._needs_reindex = False

    def delete(self, id: str) -> bool:
        """
        Delete a vector from the store.

        Args:
            id: ID of vector to delete

        Returns:
            True if deleted, False if not found
        """
        if id in self.vectors:
            del self.vectors[id]
            del self.metadata[id]
            self._needs_reindex = True
            return True
        return False

    def clear(self) -> None:
        """Clear all vectors from the store."""
        self.vectors.clear()
        self.metadata.clear()
        self._index = None
        self._vector_matrix = None
        self._needs_reindex = True

    def count(self) -> int:
        """Get the number of vectors in the store."""
        return len(self.vectors)

    def cluster(self, num_clusters: int, max_iterations: int = 100) -> list[ClusterResult]:
        """
        Cluster vectors using K-means.

        Args:
            num_clusters: Number of clusters to create
            max_iterations: Maximum K-means iterations

        Returns:
            List of ClusterResult objects
        """
        if len(self.vectors) < num_clusters:
            num_clusters = max(1, len(self.vectors))

        self._ensure_index()

        vectors = self._vector_matrix
        n_samples = vectors.shape[0]

        np.random.seed(42)
        centroids = vectors[np.random.choice(n_samples, num_clusters, replace=False)]

        for _ in range(max_iterations):
            distances = np.zeros((n_samples, num_clusters))
            for k in range(num_clusters):
                distances[:, k] = np.linalg.norm(vectors - centroids[k], axis=1)

            labels = np.argmin(distances, axis=1)

            new_centroids = np.zeros_like(centroids)
            for k in range(num_clusters):
                cluster_points = vectors[labels == k]
                if len(cluster_points) > 0:
                    new_centroids[k] = np.mean(cluster_points, axis=0)

            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        results = []
        for k in range(num_clusters):
            cluster_ids = [self._index[i] for i in range(n_samples) if labels[i] == k]
            members = [self.metadata[id].get("text", id) for id in cluster_ids[:20]]

            keyword_indices = np.argsort(centroids[k])[-5:][::-1]
            keywords = []
            for idx in keyword_indices[:5]:
                if idx < len(self._index):
                    keywords.append(self._index[idx])

            results.append(
                ClusterResult(
                    cluster_id=k,
                    members=members,
                    centroid=centroids[k].tolist(),
                    keywords=keywords,
                )
            )

        return results


class VectorIntegrationService:
    """
    Complete vector integration service combining embedding generation,
    vector storage, and hybrid retrieval with graph.

    This class provides:
    - End-to-end embedding pipeline
    - Vector storage and search
    - Graph-vector hybrid retrieval
    - Clustering and grouping
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the vector integration service.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.embedding_service = EmbeddingService(self.config.get("embedding", {}))
        self.vector_store = VectorStore(self.config.get("vector_store", {}))
        self._graph = None

    def set_graph(self, graph: Any) -> None:
        """
        Set the knowledge graph for hybrid retrieval.

        Args:
            graph: HybridGraph instance
        """
        self._graph = graph

    def process_texts(
        self,
        texts: list[str],
        metadata: dict[str, Any] | None = None,
        generate_ids: bool = False,
    ) -> list[EmbeddingResult]:
        """
        Process texts and store embeddings.

        Args:
            texts: List of texts to process
            metadata: Optional metadata for all texts
            generate_ids: Whether to generate IDs for texts

        Returns:
            List of EmbeddingResult objects
        """
        results = self.embedding_service.embed_texts(texts, metadata)

        items = []
        for i, result in enumerate(results):
            if generate_ids:
                id = f"text_{i}_{hash(result.text) % 10000}"
            else:
                id = f"embedding_{i}"
            items.append((id, result.embedding, {"text": result.text, **result.metadata}))

        self.vector_store.add_batch(items)

        return results

    def find_similar(
        self, query: str, top_k: int = 10, exclude_texts: list[str] | None = None
    ) -> list[SearchResult]:
        """
        Find texts similar to query.

        Args:
            query: Query text
            top_k: Number of results
            exclude_texts: Texts to exclude

        Returns:
            List of SearchResult objects
        """
        query_embedding = self.embedding_service.embed_text(query)

        exclude_ids = []
        if exclude_texts:
            for id, meta in self.vector_store.metadata.items():
                if meta.get("text") in exclude_texts:
                    exclude_ids.append(id)

        return self.vector_store.search(query_embedding.embedding, top_k, exclude_ids)

    def cluster_texts(self, num_clusters: int | None = None) -> list[ClusterResult]:
        """
        Cluster stored texts.

        Args:
            num_clusters: Number of clusters (auto if not specified)

        Returns:
            List of ClusterResult objects
        """
        if num_clusters is None:
            num_clusters = max(2, int(np.sqrt(self.vector_store.count())))

        return self.vector_store.cluster(num_clusters)

    def hybrid_search(
        self,
        query: str,
        graph_query: Callable | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """
        Perform hybrid search combining vector similarity with graph traversal.

        Args:
            query: Query text
            graph_query: Optional function for graph traversal
            top_k: Number of results

        Returns:
            Dictionary with vector results, graph results, and combined ranking
        """
        vector_results = self.find_similar(query, top_k * 2)

        graph_entities = []
        if graph_query and self._graph:
            graph_entities = graph_query(self._graph)

        return {
            "vector_results": [r.to_dict() for r in vector_results],
            "graph_entities": graph_entities,
            "combined_results": self._combine_results(vector_results, graph_entities, top_k),
        }

    def _combine_results(
        self,
        vector_results: list[SearchResult],
        graph_entities: list[str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Combine vector and graph results."""
        entity_set = set(graph_entities)
        combined = []

        for result in vector_results:
            text = result.text
            graph_boost = 1.0 if text in entity_set else 1.0
            combined_score = result.score * graph_boost

            combined.append(
                {
                    "text": text,
                    "score": combined_score,
                    "vector_score": result.score,
                    "source": "graph" if text in entity_set else "vector",
                    "metadata": result.metadata,
                }
            )

        combined.sort(key=lambda x: x["score"], reverse=True)

        return combined[:top_k]

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the vector integration service."""
        return {
            "embedding_model": self.embedding_service.model_name,
            "embedding_dimension": self.embedding_service._dimension,
            "vector_store_count": self.vector_store.count(),
            "cache_stats": self.embedding_service.get_cache_stats(),
            "has_graph": self._graph is not None,
        }
