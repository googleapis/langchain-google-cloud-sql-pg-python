"""Test cloudSQLVectorStore functionality."""
import os
from typing import List

from langchain_core.documents import Document

from langchain_community.vectorstores.cloudSQL import cloudSQLVectorStore
from langchain_community.vectorstores.cloudSQL import cloudSQLEngine
from langchain_community.embeddings import FakeEmbeddings

# from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

engine = cloudSQLEngine.from_instance(
    project_id = os.environ.get("PROJECT_ID", None),
    instance = os.environ.get("INSTANCE_NAME"),
    region = os.environ.get("REGION_NAME"),
    database = os.environ.get("DATABASE_NAME")
)

ADA_TOKEN_COUNT = 1536

class FakeEmbeddingsWithAdaDimension(FakeEmbeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings."""
        return [
            [float(1.0)] * (ADA_TOKEN_COUNT - 1) + [float(i)] for i in range(len(texts))
        ]

    def embed_query(self, text: str) -> List[float]:
        """Return simple embeddings."""
        return [float(1.0)] * (ADA_TOKEN_COUNT - 1) + [float(0.0)]


async def test_cloudSQLVectorStore() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = await cloudSQL.afrom_texts(
        texts=texts,
        table_name="test_table",
        embedding=FakeEmbeddingsWithAdaDimension(),
        engine=engine,
    )
    output = await docsearch.asimilarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


async def test_cloudSQLVectorStore_embeddings() -> None:
    """Test end to end construction with embeddings and search."""
    texts = ["foo", "bar", "baz"]
    text_embeddings = FakeEmbeddingsWithAdaDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = cloudSQLVectorStore.afrom_embeddings(
        text_embeddings=text_embedding_pairs,
        table_name="test_table",
        embedding=FakeEmbeddingsWithAdaDimension(),
        engine=engine,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


async def test_cloudSQLVectorStore_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = await cloudSQLVectorStore.afrom_texts(
        texts=texts,
        table_name="test_table",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        engine=engine,
    )
    output = await docsearch.asimilarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": "0"})]


async def test_cloudSQLVectorStore_with_metadatas_with_scores() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = await cloudSQLVectorStore.afrom_texts(
        texts=texts,
        table_name="test_table",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        engine=engine,
    )
    output = await docsearch.asimilarity_search_with_score("foo", k=1)
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


async def test_cloudSQLVectorStore_with_filter_match() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = await cloudSQLVectorStore.afrom_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        engine=engine,
    )
    output = await docsearch.asimilarity_search_with_score("foo", k=1, filter={"page": "0"})
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


async def test_cloudSQLVectorStore_with_filter_distant_match() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = await cloudSQLVectorStore.afrom_texts(
        texts=texts,
        table_name="test_table",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        engine=engine,
    )
    output = await docsearch.asimilarity_search_with_score("foo", k=1, filter={"page": "2"})
    assert output == [
        (Document(page_content="baz", metadata={"page": "2"}), 0.0013003906671379406)
    ]


async def test_cloudSQLVectorStore_with_filter_no_match() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = await cloudSQLVectorStore.afrom_texts(
        texts=texts,
        table_name="test_table",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        engine=engine,
    )
    output = await docsearch.asimilarity_search_with_score("foo", k=1, filter={"page": "5"})
    assert output == []

async def test_cloudSQLVectorStore_relevance_score() -> None:
    """Test to make sure the relevance score is scaled to 0-1."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = await cloudSQLVectorStore.from_texts(
        texts=texts,
        table_name="test_table",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        engine=engine,
    )

    output = await docsearch.asimilarity_search_with_relevance_scores("foo", k=3)
    assert output == [
        (Document(page_content="foo", metadata={"page": "0"}), 1.0),
        (Document(page_content="bar", metadata={"page": "1"}), 0.9996744261675065),
        (Document(page_content="baz", metadata={"page": "2"}), 0.9986996093328621),
    ]


async def test_cloudSQLVectorStore_max_marginal_relevance_search() -> None:
    """Test max marginal relevance search."""
    texts = ["foo", "bar", "baz"]
    docsearch = await cloudSQLVectorStore.afrom_texts(
        texts=texts,
        table_name="test_table",
        embedding=FakeEmbeddingsWithAdaDimension(),
        engine=engine,
    )
    output = await docsearch.max_marginal_relevance_search("foo", k=1, fetch_k=3)
    assert output == [Document(page_content="foo")]


async def test_cloudSQLVectorStore_max_marginal_relevance_search_with_score() -> None:
    """Test max marginal relevance search with relevance scores."""
    texts = ["foo", "bar", "baz"]
    docsearch = await cloudSQLVectorStore.afrom_texts(
        texts=texts,
        collection_name="test_table",
        embedding=FakeEmbeddingsWithAdaDimension(),
        engine=engine,
    )
    output = await docsearch.amax_marginal_relevance_search_with_score("foo", k=1, fetch_k=3)
    assert output == [(Document(page_content="foo"), 0.0)]