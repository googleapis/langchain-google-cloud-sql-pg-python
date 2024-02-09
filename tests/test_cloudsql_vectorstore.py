# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import uuid
from typing import List

import pytest
import pytest_asyncio
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.documents import Document
from langchain_google_vertexai import VertexAIEmbeddings
from sqlalchemy import TEXT, VARCHAR, Column

from langchain_google_cloud_sql_pg import CloudSQLVectorStore, PostgreSQLEngine
from langchain_google_cloud_sql_pg.indexes import (
    DEFAULT_DISTANCE_STRATEGY,
    BruteForce,
    DistanceStrategy,
    HNSWIndex,
    IVFFlatIndex,
)

PROJECT_ID = os.environ.get("PROJECT_ID")
INSTANCE = os.environ.get("INSTANCE_ID")
DATABASE = os.environ.get("DATABASE_ID")
REGION = os.environ.get("REGION")
ADA_TOKEN_COUNT = 768
DEFAULT_TABLE = "test_table"
CUSTOM_COL = "page"
CUSTOM_TABLE = "test_table_custom"

embeddings_service = VertexAIEmbeddings()


class FakeEmbeddingsWithAdaDimension(FakeEmbeddings):
    """Fake embeddings functionality for testing."""

    size: int = ADA_TOKEN_COUNT

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings."""
        return [
            [float(1.0)] * (ADA_TOKEN_COUNT - 1) + [float(i)] for i in range(len(texts))
        ]

    def embed_query(self, text: str = "default") -> List[float]:
        """Return simple embeddings."""
        return [float(1.0)] * (ADA_TOKEN_COUNT - 1) + [float(0.0)]


@pytest.mark.asyncio
class TestEngine:
    @pytest_asyncio.fixture
    async def engine(self) -> None:
        engine = PostgreSQLEngine.from_instance(
            project_id=PROJECT_ID,
            instance=INSTANCE,
            region=REGION,
            database=DATABASE,
        )
        await engine.init_vectorstore_table(
            "table4",
            ADA_TOKEN_COUNT,
            content_column="product",
            embedding_column="product_embedding",
            store_metadata=True,
        )
        yield engine
        await engine._aexecute_update(f"DROP TABLE table4")

    async def test_metadata_upload(self, engine):
        texts = ["Hello, World!"]
        metadatas = [{"field1": "value1", "field2": "value2"}]
        vs = CloudSQLVectorStore(
            table_name="table4",
            embedding_service=FakeEmbeddingsWithAdaDimension(),
            engine=engine,
            content_column="product",
            embedding_column="product_embedding",
        )
        await vs.aadd_texts(
            texts=texts,
            metadatas=metadatas,
        )
        output = await vs.asimilarity_search("Hello", k=1)
        assert output[0].metadata == metadatas[0]

    async def test_override_on_init(self, engine):
        await engine.init_vectorstore_table(
            "table4",
            ADA_TOKEN_COUNT,
            content_column="product",
            embedding_column="product_embedding",
            store_metadata=True,
            overwrite_existing=True,
        )
        vs = CloudSQLVectorStore(
            table_name="table4",
            embedding_service=FakeEmbeddingsWithAdaDimension(),
            engine=engine,
            content_column="product",
            embedding_column="product_embedding",
        )

        output = await vs.asimilarity_search("Hello", k=10)
        assert len(output) == 0

    async def test_override(self, engine):
        texts = ["foo", "bar"]
        vs = await CloudSQLVectorStore.afrom_texts(
            texts=texts,
            table_name="table4",
            embedding_service=FakeEmbeddingsWithAdaDimension(),
            engine=engine,
            content_column="product",
            embedding_column="product_embedding",
        )
        vs2 = await CloudSQLVectorStore.afrom_texts(
            texts=texts,
            table_name="table4",
            embedding_service=FakeEmbeddingsWithAdaDimension(),
            engine=engine,
            content_column="product",
            embedding_column="product_embedding",
            overwrite_existing=True,
        )
        output = await vs2.asimilarity_search("foo", k=10)
        assert len(output) == 2


@pytest.mark.asyncio
class TestAsync:
    @pytest_asyncio.fixture  # (scope="function")
    async def engine(self):
        engine = PostgreSQLEngine.from_instance(
            project_id=PROJECT_ID,
            instance=INSTANCE,
            region=REGION,
            database=DATABASE,
        )
        await engine.init_vectorstore_table(DEFAULT_TABLE, ADA_TOKEN_COUNT)
        yield engine
        await engine._aexecute_update(f"DROP TABLE {DEFAULT_TABLE}")

    @pytest_asyncio.fixture  # (scope="function")
    async def engine_custom(self):
        table_name = CUSTOM_TABLE
        engine = PostgreSQLEngine.from_instance(
            project_id=PROJECT_ID,
            instance=INSTANCE,
            region=REGION,
            database=DATABASE,
        )
        await engine.init_vectorstore_table(
            table_name,
            ADA_TOKEN_COUNT,
            metadata_columns=[Column("page", TEXT)],
        )
        yield engine
        # return engine
        await engine._aexecute_update(f"DROP TABLE {table_name}")

    async def test_similarity_search(self, engine) -> None:
        """Test end to end construction and search."""
        texts = ["foo", "bar", "baz"]
        vs = await CloudSQLVectorStore.afrom_texts(
            texts=texts,
            table_name=DEFAULT_TABLE,
            embedding_service=FakeEmbeddingsWithAdaDimension(),
            engine=engine,
        )
        output = await vs.asimilarity_search("foo", k=1)
        assert output == [Document(page_content="foo")]

    async def test_with_metadatas(self, engine_custom) -> None:
        """Test end to end construction and search."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i)} for i in range(len(texts))]
        vs = await CloudSQLVectorStore.afrom_texts(
            texts=texts,
            metadatas=metadatas,
            table_name=CUSTOM_TABLE,
            embedding_service=FakeEmbeddingsWithAdaDimension(),
            engine=engine_custom,
            metadata_columns=["page"],
        )
        output = await vs.asimilarity_search("foo", k=1)
        assert output == [Document(page_content="foo", metadata={"page": "0"})]

    async def test_with_ids(self, engine_custom) -> None:
        """Test end to end construction and search."""
        texts = ["foo", "bar", "baz"]
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        metadatas = [{"page": str(i)} for i in range(len(texts))]
        vs = await CloudSQLVectorStore.afrom_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            table_name=CUSTOM_TABLE,
            embedding_service=FakeEmbeddingsWithAdaDimension(),
            engine=engine_custom,
            metadata_columns=["page"],
        )
        output = await vs.adelete(ids)
        assert output

    async def test_with_metadatas_with_scores(
        self,
        engine_custom,
    ) -> None:
        """Test end to end construction and search."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i)} for i in range(len(texts))]
        vs = await CloudSQLVectorStore.afrom_texts(
            texts=texts,
            table_name=CUSTOM_TABLE,
            embedding_service=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            engine=engine_custom,
            metadata_columns=["page"],
        )
        output = await vs.asimilarity_search_with_score("foo", k=1)
        assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]

    async def test_with_filter_match(self, engine_custom) -> None:
        """Test end to end construction and search."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i)} for i in range(len(texts))]
        vs = await CloudSQLVectorStore.afrom_texts(
            texts=texts,
            table_name=CUSTOM_TABLE,
            embedding_service=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            engine=engine_custom,
            metadata_columns=["page"],
        )
        output = await vs.asimilarity_search_with_score("foo", k=1, filter="page = '0'")
        assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]

    async def test_with_filter_distant_match(
        self,
        engine_custom,
    ) -> None:
        """Test end to end construction and search."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i)} for i in range(len(texts))]
        vs = await CloudSQLVectorStore.afrom_texts(
            texts=texts,
            table_name=CUSTOM_TABLE,
            embedding_service=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            engine=engine_custom,
            metadata_columns=["page"],
        )
        output = await vs.asimilarity_search_with_score("foo", k=1, filter="page = '2'")
        assert output == [
            (
                Document(page_content="baz", metadata={"page": "2"}),
                0.0025974069839586056,
            )
        ]

    async def test_with_filter_no_match(
        self,
        engine_custom,
    ) -> None:
        """Test end to end construction and search."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i)} for i in range(len(texts))]
        vs = await CloudSQLVectorStore.afrom_texts(
            texts=texts,
            table_name=CUSTOM_TABLE,
            embedding_service=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            engine=engine_custom,
            metadata_columns=["page"],
        )
        output = await vs.asimilarity_search_with_score("foo", k=1, filter="page = '5'")
        assert output == []

    async def test_relevance_score(self, engine_custom) -> None:
        """Test to make sure the relevance score is scaled to 0-1."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i)} for i in range(len(texts))]
        vs = await CloudSQLVectorStore.afrom_texts(
            texts=texts,
            table_name=CUSTOM_TABLE,
            embedding_service=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            engine=engine_custom,
            metadata_columns=["page"],
        )

        output = await vs.asimilarity_search_with_relevance_scores("foo", k=3)
        assert output == [
            (Document(page_content="foo", metadata={"page": "0"}), 1.0),
            (
                Document(page_content="bar", metadata={"page": "1"}),
                0.9993487462676214,
            ),
            (
                Document(page_content="baz", metadata={"page": "2"}),
                0.9974025930160414,
            ),
        ]

    async def test_max_marginal_relevance_search(
        self,
        engine,
    ) -> None:
        """Test max marginal relevance search."""
        texts = ["foo", "bar", "baz"]
        vs = await CloudSQLVectorStore.afrom_texts(
            texts=texts,
            table_name=DEFAULT_TABLE,
            embedding_service=embeddings_service,
            engine=engine,
        )
        output = await vs.amax_marginal_relevance_search("foo", k=1, fetch_k=3)
        assert output == [Document(page_content="foo")]

    async def test_max_marginal_relevance_search_with_score(
        self,
        engine,
    ) -> None:
        """Test max marginal relevance search with relevance scores."""
        texts = ["foo", "bar", "baz"]
        vs = await CloudSQLVectorStore.afrom_texts(
            texts=texts,
            table_name=DEFAULT_TABLE,
            embedding_service=embeddings_service,
            engine=engine,
        )
        embedding = embeddings_service.embed_query(text="foo")
        output = await vs.amax_marginal_relevance_search_with_score_by_vector(
            embedding, k=1, fetch_k=3
        )
        assert output[0][0] == Document(page_content="foo")
        assert output[0][1] > 0

    async def test_max_marginal_relevance_search_amenities(
        self,
        engine_custom,
    ) -> None:
        """Test max marginal relevance search."""
        vs = CloudSQLVectorStore(
            table_name="amenities",
            embedding_service=embeddings_service,
            engine=engine_custom,
        )
        output = await vs.amax_marginal_relevance_search("coffee", k=1, fetch_k=3)
        assert "coffee" in output[0].page_content


@pytest.mark.asyncio
class TestIndex:
    @pytest_asyncio.fixture()
    async def vs(self):
        table_name = "test_table2"
        engine = PostgreSQLEngine.from_instance(
            project_id=PROJECT_ID,
            instance=INSTANCE,
            region=REGION,
            database=DATABASE,
        )
        await engine.init_vectorstore_table(table_name, ADA_TOKEN_COUNT)
        texts = ["foo", "bar", "baz"]
        vs = await CloudSQLVectorStore.afrom_texts(
            texts=texts,
            table_name="test_table2",
            embedding_service=embeddings_service,
            engine=engine,
        )
        yield vs
        # return engine
        # await engine._aexecute_update(f"DROP TABLE {table_name}")

    async def test_applyindex(self, vs) -> None:
        index = HNSWIndex()
        await vs.aapply_index(index)

    async def test_applyindex_l2(self, vs) -> None:
        index = HNSWIndex(name="hnswl2", distance_strategy=DistanceStrategy.EUCLIDEAN)
        await vs.aapply_index(index)

    async def test_applyindex_ip(self, vs) -> None:
        index = IVFFlatIndex(distance_strategy=DistanceStrategy.INNER_PRODUCT)
        await vs.aapply_index(index)

    async def test_reindex(self, vs) -> None:
        """Test the creation and reindexing of index"""
        output = await vs.areindex("langchainhnsw")
        assert output is None

    async def test_dropindex(self, vs) -> None:
        """Test the dropping of index"""
        output = await vs.adrop_index("langchainivfflat")
        await vs.adrop_index("langchainhnsw")
        await vs.adrop_index("hnswl2")
        assert output is None


# @pytest.mark.asyncio
# class TestSync:
#     @pytest_asyncio.fixture(scope="function")
#     async def engine(self):
#         table_name = "test_table_sync"
#         engine = PostgreSQLEngine.from_instance(
#             project_id=PROJECT_ID,
#             instance=INSTANCE,
#             region=REGION,
#             database=DATABASE,
#         )
#         await engine.init_vectorstore_table(table_name, ADA_TOKEN_COUNT)
#         yield engine
#         # return engine
#         await engine._aexecute_update(f"DROP TABLE {table_name}")

#     @pytest_asyncio.fixture(scope="function")
#     async def engine_custom(self):
#         table_name = CUSTOM_TABLE
#         engine = PostgreSQLEngine.from_instance(
#             project_id=PROJECT_ID,
#             instance=INSTANCE,
#             region=REGION,
#             database=DATABASE,
#         )
#         await engine.init_vectorstore_table(
#             table_name,
#             ADA_TOKEN_COUNT,
#             metadata_columns=[Column("page", TEXT)],
#         )
#         yield engine
#         # return engine
#         await engine._aexecute_update(f"DROP TABLE {table_name}")

#     def test(self, engine) -> None:
#         """Test end to end construction and search."""
#         texts = ["foo", "bar", "baz"]
#         vs = CloudSQLVectorStore.from_texts(
#             texts=texts,
#             table_name="test_table_sync",
#             embedding_service=FakeEmbeddingsWithAdaDimension(),
#             engine=engine,
#         )
#         output = vs.similarity_search("foo", k=1)
#         assert output == [Document(page_content="foo")]

#     def test_with_metadatas(self, engine_custom) -> None:
#         """Test end to end construction and search."""
#         texts = ["foo", "bar", "baz"]
#         metadatas = [{"page": str(i)} for i in range(len(texts))]
#         vs = CloudSQLVectorStore.from_texts(
#             texts=texts,
#             metadatas=metadatas,
#             table_name=CUSTOM_TABLE,
#             embedding_service=FakeEmbeddingsWithAdaDimension(),
#             engine=engine_custom,
#             metadata_columns=["page"],
#         )
#         output = vs.similarity_search("foo", k=1)
#         assert output == [Document(page_content="foo", metadata={"page": "0"})]

#     def test_with_metadatas_with_scores(
#         self,
#         engine_custom,
#     ) -> None:
#         """Test end to end construction and search."""
#         texts = ["foo", "bar", "baz"]
#         metadatas = [{"page": str(i)} for i in range(len(texts))]
#         vs = CloudSQLVectorStore.from_texts(
#             texts=texts,
#             table_name=CUSTOM_TABLE,
#             embedding_service=FakeEmbeddingsWithAdaDimension(),
#             metadatas=metadatas,
#             engine=engine_custom,
#             metadata_columns=["page"],
#         )
#         output = vs.similarity_search_with_score("foo", k=1)
#         assert output == [
#             (Document(page_content="foo", metadata={"page": "0"}), 0.0)
#         ]

#     def test_with_filter_match(self, engine_custom) -> None:
#         """Test end to end construction and search."""
#         texts = ["foo", "bar", "baz"]
#         metadatas = [{"page": str(i)} for i in range(len(texts))]
#         vs = CloudSQLVectorStore.from_texts(
#             texts=texts,
#             table_name=CUSTOM_TABLE,
#             embedding_service=FakeEmbeddingsWithAdaDimension(),
#             metadatas=metadatas,
#             engine=engine_custom,
#             metadata_columns=["page"],
#         )
#         output = vs.similarity_search_with_score(
#             "foo", k=1, filter="page = '0'"
#         )
#         assert output == [
#             (Document(page_content="foo", metadata={"page": "0"}), 0.0)
#         ]

#     def test_with_filter_distant_match(
#         self,
#         engine_custom,
#     ) -> None:
#         """Test end to end construction and search."""
#         texts = ["foo", "bar", "baz"]
#         metadatas = [{"page": str(i)} for i in range(len(texts))]
#         vs = CloudSQLVectorStore.from_texts(
#             texts=texts,
#             table_name=CUSTOM_TABLE,
#             embedding_service=FakeEmbeddingsWithAdaDimension(),
#             metadatas=metadatas,
#             engine=engine_custom,
#             metadata_columns=["page"],
#         )
#         output = vs.similarity_search_with_score(
#             "foo", k=1, filter="page = '2'"
#         )
#         assert output == [
#             (
#                 Document(page_content="baz", metadata={"page": "2"}),
#                 0.0025974069839586056,
#             )
#         ]

#     def test_with_filter_no_match(
#         self,
#         engine_custom,
#     ) -> None:
#         """Test end to end construction and search."""
#         texts = ["foo", "bar", "baz"]
#         metadatas = [{"page": str(i)} for i in range(len(texts))]
#         vs = CloudSQLVectorStore.from_texts(
#             texts=texts,
#             table_name=CUSTOM_TABLE,
#             embedding_service=FakeEmbeddingsWithAdaDimension(),
#             metadatas=metadatas,
#             engine=engine_custom,
#             metadata_columns=["page"],
#         )
#         output = vs.similarity_search_with_score(
#             "foo", k=1, filter="page = '5'"
#         )
#         assert output == []

#     def test_relevance_score(self, engine_custom) -> None:
#         """Test to make sure the relevance score is scaled to 0-1."""
#         texts = ["foo", "bar", "baz"]
#         metadatas = [{"page": str(i)} for i in range(len(texts))]
#         vs = CloudSQLVectorStore.from_texts(
#             texts=texts,
#             table_name=CUSTOM_TABLE,
#             embedding_service=FakeEmbeddingsWithAdaDimension(),
#             metadatas=metadatas,
#             engine=engine_custom,
#             metadata_columns=["page"],
#         )

#         output = vs.similarity_search_with_relevance_scores("foo", k=3)
#         assert output == [
#             (Document(page_content="foo", metadata={"page": "0"}), 1.0),
#             (
#                 Document(page_content="bar", metadata={"page": "1"}),
#                 0.9993487462676214,
#             ),
#             (
#                 Document(page_content="baz", metadata={"page": "2"}),
#                 0.9974025930160414,
#             ),
#         ]

#     def test_max_marginal_relevance_search(
#         self,
#         engine,
#     ) -> None:
#         """Test max marginal relevance search."""
#         texts = ["foo", "bar", "baz"]
#         vs = CloudSQLVectorStore.from_texts(
#             texts=texts,
#             table_name="test_table_sync",
#             embedding_service=embeddings_service,
#             engine=engine,
#         )
#         output = vs.max_marginal_relevance_search("foo", k=1, fetch_k=3)
#         assert output == [Document(page_content="foo")]

#     def test_max_marginal_relevance_search_with_score(
#         self,
#         engine,
#     ) -> None:
#         """Test max marginal relevance search with relevance scores."""
#         texts = ["foo", "bar", "baz"]
#         vs = CloudSQLVectorStore.from_texts(
#             texts=texts,
#             table_name="test_table_sync",
#             embedding_service=embeddings_service,
#             engine=engine,
#         )
#         embedding = embeddings_service.embed_query(text="foo")
#         output = vs.max_marginal_relevance_search_with_score_by_vector(
#             embedding, k=1, fetch_k=3
#         )
#         assert output[0][0] == Document(page_content="foo")
#         assert output[0][1] > 0

#     def test_max_marginal_relevance_search_amenities(
#         self,
#         engine_custom,
#     ) -> None:
#         """Test max marginal relevance search."""
#         vs = CloudSQLVectorStore(
#             table_name="amenities",
#             embedding_service=embeddings_service,
#             engine=engine_custom,
#         )
#         output = vs.max_marginal_relevance_search(
#             "coffee", k=1, fetch_k=3
#         )
#         assert "coffee" in output[0].page_content
