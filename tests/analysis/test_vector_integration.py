"""
Tests for Vector Integration Service Module
"""

import pytest
import sys

sys.path.insert(0, "/home/meta_agent/src")

import numpy as np

from src.analysis.vector_integration import (
    VectorIntegrationService,
    EmbeddingService,
    VectorStore,
    EmbeddingResult,
    SearchResult,
    ClusterResult,
)


class TestEmbeddingService:
    """Tests for EmbeddingService class."""

    @pytest.fixture
    def service(self):
        """Create an EmbeddingService instance."""
        return EmbeddingService({"model_name": "all-MiniLM-L6-v2"})

    def test_initialization(self, service):
        """Test service initialization."""
        assert service is not None
        assert service.model_name == "all-MiniLM-L6-v2"
        assert service.cache == {}

    def test_embed_text(self, service):
        """Test single text embedding."""
        result = service.embed_text("This is a test sentence")

        assert isinstance(result, EmbeddingResult)
        assert result.text == "This is a test sentence"
        assert len(result.embedding) > 0
        assert result.model_name == "all-MiniLM-L6-v2"

    def test_embed_text_caching(self, service):
        """Test that embeddings are cached."""
        text = "Test sentence for caching"
        result1 = service.embed_text(text)
        result2 = service.embed_text(text)

        assert result2.metadata.get("source") == "cache"

    def test_embed_texts_batch(self, service):
        """Test batch text embedding."""
        texts = [
            "First sentence",
            "Second sentence",
            "Third sentence",
        ]

        results = service.embed_texts(texts)

        assert len(results) == 3
        assert all(isinstance(r, EmbeddingResult) for r in results)

    def test_clear_cache(self, service):
        """Test cache clearing."""
        service.embed_text("Test sentence")
        service.clear_cache()

        assert len(service.cache) == 0

    def test_get_cache_stats(self, service):
        """Test cache statistics."""
        stats = service.get_cache_stats()

        assert "cache_size" in stats
        assert "max_cache_size" in stats


class TestVectorStore:
    """Tests for VectorStore class."""

    @pytest.fixture
    def store(self):
        """Create a VectorStore instance."""
        return VectorStore()

    def test_initialization(self, store):
        """Test store initialization."""
        assert store is not None
        assert store.vectors == {}
        assert store.metadata == {}

    def test_add_vector(self, store):
        """Test adding a vector."""
        embedding = [0.1] * 10
        store.add("test_id", embedding, {"text": "test"})

        assert "test_id" in store.vectors
        assert store.metadata["test_id"]["text"] == "test"

    def test_add_batch(self, store):
        """Test batch vector addition."""
        items = [
            ("id1", [0.1] * 10, {"text": "text1"}),
            ("id2", [0.2] * 10, {"text": "text2"}),
            ("id3", [0.3] * 10, {"text": "text3"}),
        ]

        store.add_batch(items)

        assert store.count() == 3

    def test_search(self, store):
        """Test similarity search."""
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

        for i, emb in enumerate(embeddings):
            store.add(f"vec{i}", emb, {"text": f"vector {i}"})

        query = [0.9, 0.1, 0.0]
        results = store.search(query, top_k=2)

        assert len(results) == 2
        assert results[0].score >= results[1].score

    def test_search_with_exclusion(self, store):
        """Test search with excluded IDs."""
        store.add("id1", [1.0, 0.0, 0.0], {"text": "test1"})
        store.add("id2", [0.5, 0.5, 0.0], {"text": "test2"})

        results = store.search([0.8, 0.2, 0.0], top_k=1, exclude_ids=["id1"])

        assert len(results) == 1
        assert results[0].text == "test2"

    def test_delete_vector(self, store):
        """Test vector deletion."""
        store.add("test_id", [0.1] * 10)
        assert store.count() == 1

        deleted = store.delete("test_id")
        assert deleted is True
        assert store.count() == 0

    def test_delete_nonexistent(self, store):
        """Test deletion of nonexistent vector."""
        deleted = store.delete("nonexistent")
        assert deleted is False

    def test_clear_store(self, store):
        """Test store clearing."""
        store.add("id1", [0.1] * 10)
        store.add("id2", [0.2] * 10)

        store.clear()

        assert store.count() == 0

    def test_cluster(self, store):
        """Test vector clustering."""
        cluster1 = [[0.1] * 10 for _ in range(5)]
        cluster2 = [[0.9] * 10 for _ in range(5)]

        for i, emb in enumerate(cluster1 + cluster2):
            store.add(f"vec{i}", emb, {"text": f"vector {i}"})

        results = store.cluster(num_clusters=2)

        assert len(results) == 2
        assert all(isinstance(r, ClusterResult) for r in results)


class TestVectorIntegrationService:
    """Tests for VectorIntegrationService class."""

    @pytest.fixture
    def service(self):
        """Create a VectorIntegrationService instance."""
        return VectorIntegrationService()

    def test_initialization(self, service):
        """Test service initialization."""
        assert service is not None
        assert service.embedding_service is not None
        assert service.vector_store is not None

    def test_set_graph(self, service):
        """Test graph setting."""
        from src.data.graph import HybridGraph

        graph = HybridGraph()
        service.set_graph(graph)

        assert service._graph is graph

    def test_process_texts(self, service):
        """Test text processing and embedding storage."""
        texts = [
            "Machine learning is transforming industries",
            "Artificial intelligence enables new capabilities",
            "Deep learning models achieve state-of-the-art results",
        ]

        results = service.process_texts(texts, generate_ids=True)

        assert len(results) == 3
        assert service.vector_store.count() == 3

    def test_find_similar(self, service):
        """Test similarity search."""
        texts = [
            "AI and machine learning",
            "Natural language processing",
            "Computer vision applications",
        ]
        service.process_texts(texts)

        results = service.find_similar("artificial intelligence", top_k=2)

        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_cluster_texts(self, service):
        """Test text clustering."""
        texts = [
            "AI machine learning neural networks",
            "ML algorithms deep learning",
            "Python programming software development",
            "JavaScript web frontend",
        ]
        service.process_texts(texts)

        results = service.cluster_texts(num_clusters=2)

        assert len(results) == 2

    def test_hybrid_search_without_graph(self, service):
        """Test hybrid search without graph."""
        texts = ["AI technology advances", "ML applications grow"]
        service.process_texts(texts)

        result = service.hybrid_search("artificial intelligence")

        assert "vector_results" in result
        assert "combined_results" in result

    def test_get_statistics(self, service):
        """Test statistics retrieval."""
        texts = ["Test text 1", "Test text 2"]
        service.process_texts(texts)

        stats = service.get_statistics()

        assert "embedding_model" in stats
        assert "vector_store_count" in stats
        assert stats["vector_store_count"] == 2


class TestEmbeddingResult:
    """Tests for EmbeddingResult class."""

    def test_creation(self):
        """Test result creation."""
        result = EmbeddingResult(
            text="Test",
            embedding=[0.1, 0.2, 0.3],
            model_name="test-model",
            dimension=3,
        )

        assert result.text == "Test"
        assert len(result.embedding) == 3

    def test_to_dict(self):
        """Test serialization."""
        result = EmbeddingResult(
            text="Example",
            embedding=[0.5, 0.5],
            model_name="model",
            dimension=2,
            metadata={"key": "value"},
        )

        data = result.to_dict()

        assert data["text"] == "Example"
        assert data["embedding"] == [0.5, 0.5]
        assert data["metadata"] == {"key": "value"}


class TestSearchResult:
    """Tests for SearchResult class."""

    def test_creation(self):
        """Test result creation."""
        result = SearchResult(
            text="Found text",
            score=0.95,
            metadata={"source": "database"},
        )

        assert result.text == "Found text"
        assert result.score == 0.95

    def test_to_dict(self):
        """Test serialization."""
        result = SearchResult(
            text="Search result",
            score=0.8,
        )

        data = result.to_dict()

        assert data["text"] == "Search result"
        assert data["score"] == 0.8


class TestClusterResult:
    """Tests for ClusterResult class."""

    def test_creation(self):
        """Test result creation."""
        result = ClusterResult(
            cluster_id=0,
            members=["item1", "item2", "item3"],
            centroid=[0.5, 0.5, 0.5],
            keywords=["keyword1", "keyword2"],
        )

        assert result.cluster_id == 0
        assert len(result.members) == 3

    def test_to_dict(self):
        """Test serialization."""
        result = ClusterResult(
            cluster_id=1,
            members=["a", "b", "c"],
        )

        data = result.to_dict()

        assert data["cluster_id"] == 1
        assert data["member_count"] == 3
        assert len(data["members"]) <= 10
