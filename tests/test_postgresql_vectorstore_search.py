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

import pytest
import pytest_asyncio
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_google_cloud_sql_pg import Column, PostgresEngine, PostgresVectorStore
from langchain_google_cloud_sql_pg.indexes import DistanceStrategy, HNSWQueryOptions

DEFAULT_TABLE = "test_table" + str(uuid.uuid4()).replace("-", "_")
CUSTOM_TABLE = "test_table_custom" + str(uuid.uuid4()).replace("-", "_")
VECTOR_SIZE = 768

embeddings_service = DeterministicFakeEmbedding(size=VECTOR_SIZE)

texts = ["foo", "bar", "baz", "boo"]
ids = [str(uuid.uuid4()) for i in range(len(texts))]
metadatas = [{"page": str(i), "source": "google.com"} for i in range(len(texts))]
docs = [
    Document(page_content=texts[i], metadata=metadatas[i]) for i in range(len(texts))
]

embeddings = [embeddings_service.embed_query("foo") for i in range(len(texts))]


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


@pytest.mark.asyncio(scope="class")
class TestVectorStoreSearch:
    @pytest.fixture(scope="module")
    def db_project(self) -> str:
        return get_env_var("PROJECT_ID", "project id for google cloud")

    @pytest.fixture(scope="module")
    def db_region(self) -> str:
        return get_env_var("REGION", "region for cloud sql instance")

    @pytest.fixture(scope="module")
    def db_instance(self) -> str:
        return get_env_var("INSTANCE_ID", "instance for cloud sql")

    @pytest.fixture(scope="module")
    def db_name(self) -> str:
        return get_env_var("DATABASE_ID", "instance for cloud sql")

    @pytest_asyncio.fixture(scope="class")
    async def engine(self, db_project, db_region, db_instance, db_name):
        engine = await PostgresEngine.afrom_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            database=db_name,
        )
        yield engine

        await engine._connector.close_async()
        await engine._engine.dispose()

    @pytest_asyncio.fixture(scope="class")
    async def vs(self, engine):
        await engine.ainit_vectorstore_table(
            DEFAULT_TABLE, VECTOR_SIZE, store_metadata=False
        )
        vs = await PostgresVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=DEFAULT_TABLE,
        )
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_documents(docs, ids=ids)
        yield vs
        await engine._aexecute(f"DROP TABLE IF EXISTS {DEFAULT_TABLE}")

    @pytest_asyncio.fixture(scope="class")
    def engine_sync(self, db_project, db_region, db_instance, db_name):
        engine = PostgresEngine.from_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            database=db_name,
        )
        yield engine

        engine._run_as_sync(engine._connector.close_async())
        engine._run_as_sync(engine._engine.dispose())

    @pytest_asyncio.fixture(scope="class")
    async def vs_custom(self, engine_sync):
        engine_sync.init_vectorstore_table(
            CUSTOM_TABLE,
            VECTOR_SIZE,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=[
                Column("page", "TEXT"),
                Column("source", "TEXT"),
            ],
            store_metadata=False,
        )

        vs_custom = PostgresVectorStore.create_sync(
            engine_sync,
            embedding_service=embeddings_service,
            table_name=CUSTOM_TABLE,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            index_query_options=HNSWQueryOptions(ef_search=1),
        )
        vs_custom.add_documents(docs, ids=ids)
        yield vs_custom
        engine_sync._aexecute(f"DROP TABLE IF EXISTS {CUSTOM_TABLE}")

    async def test_asimilarity_search(self, vs):
        results = await vs.asimilarity_search("foo", k=1)
        assert len(results) == 1
        assert results == [Document(page_content="foo")]
        results = await vs.asimilarity_search("foo", k=1, filter="content = 'bar'")
        assert results == [Document(page_content="bar")]

    def test_asimilarity_search_cross_env(self, vs):
        results = vs.similarity_search("foo", k=1)
        assert len(results) == 1
        assert results == [Document(page_content="foo")]

    async def test_asimilarity_search_score(self, vs):
        results = await vs.asimilarity_search_with_score("foo")
        assert len(results) == 4
        assert results[0][0] == Document(page_content="foo")
        assert results[0][1] == 0

    async def test_asimilarity_search_by_vector(self, vs):
        embedding = embeddings_service.embed_query("foo")
        results = await vs.asimilarity_search_by_vector(embedding)
        assert len(results) == 4
        assert results[0] == Document(page_content="foo")
        results = await vs.asimilarity_search_with_score_by_vector(embedding)
        assert results[0][0] == Document(page_content="foo")
        assert results[0][1] == 0

    async def test_similarity_search_with_relevance_scores_threshold_cosine(self, vs):
        score_threshold = {"score_threshold": 0}
        results = await vs.asimilarity_search_with_relevance_scores(
            "foo", **score_threshold
        )
        assert len(results) == 4

        score_threshold = {"score_threshold": 0.02}
        results = await vs.asimilarity_search_with_relevance_scores(
            "foo", **score_threshold
        )
        assert len(results) == 2

        score_threshold = {"score_threshold": 0.9}
        results = await vs.asimilarity_search_with_relevance_scores(
            "foo", **score_threshold
        )
        assert len(results) == 1
        assert results[0][0] == Document(page_content="foo")

    async def test_similarity_search_with_relevance_scores_threshold_euclidean(
        self, engine
    ):
        vs = await PostgresVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=DEFAULT_TABLE,
            distance_strategy=DistanceStrategy.EUCLIDEAN,
        )

        score_threshold = {"score_threshold": 0.9}
        results = await vs.asimilarity_search_with_relevance_scores(
            "foo", **score_threshold
        )
        assert len(results) == 1
        assert results[0][0] == Document(page_content="foo")

    async def test_amax_marginal_relevance_search(self, vs):
        results = await vs.amax_marginal_relevance_search("bar")
        assert results[0] == Document(page_content="bar")
        results = await vs.amax_marginal_relevance_search(
            "bar", filter="content = 'boo'"
        )
        assert results[0] == Document(page_content="boo")

    async def test_amax_marginal_relevance_search_vector(self, vs):
        embedding = embeddings_service.embed_query("bar")
        results = await vs.amax_marginal_relevance_search_by_vector(embedding)
        assert results[0] == Document(page_content="bar")

    async def test_amax_marginal_relevance_search_vector_score(self, vs):
        embedding = embeddings_service.embed_query("bar")
        results = await vs.amax_marginal_relevance_search_with_score_by_vector(
            embedding
        )
        assert results[0][0] == Document(page_content="bar")

        results = await vs.amax_marginal_relevance_search_with_score_by_vector(
            embedding, lambda_mult=0.75, fetch_k=10
        )
        assert results[0][0] == Document(page_content="bar")

    def test_similarity_search(self, vs_custom):
        results = vs_custom.similarity_search("foo", k=1)
        assert len(results) == 1
        assert results == [Document(page_content="foo")]
        results = vs_custom.similarity_search("foo", k=1, filter="mycontent = 'bar'")
        assert results == [Document(page_content="bar")]

    def test_similarity_search_score(self, vs_custom):
        results = vs_custom.similarity_search_with_score("foo")
        assert len(results) == 4
        assert results[0][0] == Document(page_content="foo")
        assert results[0][1] == 0

    def test_similarity_search_by_vector(self, vs_custom):
        embedding = embeddings_service.embed_query("foo")
        results = vs_custom.similarity_search_by_vector(embedding)
        assert len(results) == 4
        assert results[0] == Document(page_content="foo")
        results = vs_custom.similarity_search_with_score_by_vector(embedding)
        assert results[0][0] == Document(page_content="foo")
        assert results[0][1] == 0

    def test_max_marginal_relevance_search(self, vs_custom):
        results = vs_custom.max_marginal_relevance_search("bar")
        assert results[0] == Document(page_content="bar")
        results = vs_custom.max_marginal_relevance_search(
            "bar", filter="mycontent = 'boo'"
        )
        assert results[0] == Document(page_content="boo")

    def test_max_marginal_relevance_search_vector(self, vs_custom):
        embedding = embeddings_service.embed_query("bar")
        results = vs_custom.max_marginal_relevance_search_by_vector(embedding)
        assert results[0] == Document(page_content="bar")

    def test_max_marginal_relevance_search_vector_score(self, vs_custom):
        embedding = embeddings_service.embed_query("bar")
        results = vs_custom.max_marginal_relevance_search_with_score_by_vector(
            embedding
        )
        assert results[0][0] == Document(page_content="bar")

        results = vs_custom.max_marginal_relevance_search_with_score_by_vector(
            embedding, lambda_mult=0.75, fetch_k=10
        )
        assert results[0][0] == Document(page_content="bar")
