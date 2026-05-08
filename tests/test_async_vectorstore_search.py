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

import asyncio
import os
import uuid
from typing import Any, Coroutine

import pytest
import pytest_asyncio
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from metadata_filtering_data import FILTERING_TEST_CASES, METADATAS
from sqlalchemy import text

from langchain_google_cloud_sql_pg import (  # type: ignore
    Column,
    HybridSearchConfig,
    PostgresEngine,
    reciprocal_rank_fusion,
    weighted_sum_ranking,
)
from langchain_google_cloud_sql_pg.async_vectorstore import AsyncPostgresVectorStore
from langchain_google_cloud_sql_pg.indexes import DistanceStrategy, HNSWQueryOptions

DEFAULT_TABLE = "test_table" + str(uuid.uuid4()).replace("-", "_")
CUSTOM_TABLE = "test_table_custom" + str(uuid.uuid4()).replace("-", "_")
CUSTOM_FILTER_TABLE = "custom_filter" + str(uuid.uuid4()).replace("-", "_")
HYBRID_SEARCH_TABLE1 = "hybrid1" + str(uuid.uuid4()).replace("-", "_")
HYBRID_SEARCH_TABLE2 = "hybrid2" + str(uuid.uuid4()).replace("-", "_")
VECTOR_SIZE = 768

embeddings_service = DeterministicFakeEmbedding(size=VECTOR_SIZE)

texts = ["foo", "bar", "baz", "boo"]
ids = [str(uuid.uuid4()) for i in range(len(texts))]
metadatas = [{"page": str(i), "source": "google.com"} for i in range(len(texts))]
docs = [
    Document(page_content=texts[i], metadata=metadatas[i]) for i in range(len(texts))
]
filter_docs = [
    Document(page_content=texts[i], metadata=METADATAS[i]) for i in range(len(texts))
]
embeddings = [embeddings_service.embed_query("foo") for i in range(len(texts))]

# Documents designed for hybrid search testing
hybrid_docs_content = {
    "hs_doc_apple_fruit": "An apple is a sweet and edible fruit produced by an apple tree. Apples are very common.",
    "hs_doc_apple_tech": "Apple Inc. is a multinational technology company. Their latest tech is amazing.",
    "hs_doc_orange_fruit": "The orange is the fruit of various citrus species. Oranges are tasty.",
    "hs_doc_generic_tech": "Technology drives innovation in the modern world. Tech is evolving.",
    "hs_doc_unrelated_cat": "A fluffy cat sat on a mat quietly observing a mouse.",
}
hybrid_docs = [
    Document(page_content=content, metadata={"doc_id_key": key})
    for key, content in hybrid_docs_content.items()
]


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


# Helper to bridge the Main Test Loop and the Engine Background Loop
async def run_on_background(engine: PostgresEngine, coro: Coroutine) -> Any:
    """Runs a coroutine on the engine's background loop."""
    if engine._loop:
        return await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(coro, engine._loop)
        )
    return await coro


async def aexecute(
    engine: PostgresEngine,
    query: str,
) -> None:
    async def _impl():
        async with engine._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    await run_on_background(engine, _impl())


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
        await aexecute(engine, f"DROP TABLE IF EXISTS {DEFAULT_TABLE}")
        await aexecute(engine, f"DROP TABLE IF EXISTS {CUSTOM_TABLE}")
        await aexecute(engine, f"DROP TABLE IF EXISTS {CUSTOM_FILTER_TABLE}")
        await aexecute(engine, f"DROP TABLE IF EXISTS {HYBRID_SEARCH_TABLE1}")
        await aexecute(engine, f"DROP TABLE IF EXISTS {HYBRID_SEARCH_TABLE2}")
        await engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def vs(self, engine):
        await run_on_background(
            engine,
            engine._ainit_vectorstore_table(
                DEFAULT_TABLE, VECTOR_SIZE, store_metadata=False
            ),
        )
        vs = await run_on_background(
            engine,
            AsyncPostgresVectorStore.create(
                engine,
                embedding_service=embeddings_service,
                table_name=DEFAULT_TABLE,
            ),
        )
        await run_on_background(engine, vs.aadd_documents(docs, ids=ids))
        yield vs

    @pytest_asyncio.fixture(scope="class")
    async def vs_custom(self, engine):
        await run_on_background(
            engine,
            engine._ainit_vectorstore_table(
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
            ),
        )

        vs_custom = await run_on_background(
            engine,
            AsyncPostgresVectorStore.create(
                engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="mycontent",
                embedding_column="myembedding",
                index_query_options=HNSWQueryOptions(ef_search=1),
            ),
        )
        await run_on_background(engine, vs_custom.aadd_documents(docs, ids=ids))
        yield vs_custom

    @pytest_asyncio.fixture(scope="class")
    async def vs_custom_filter(self, engine):
        await run_on_background(
            engine,
            engine._ainit_vectorstore_table(
                CUSTOM_FILTER_TABLE,
                VECTOR_SIZE,
                metadata_columns=[
                    Column("name", "TEXT"),
                    Column("code", "TEXT"),
                    Column("price", "FLOAT"),
                    Column("is_available", "BOOLEAN"),
                    Column("tags", "TEXT[]"),
                    Column("inventory_location", "INTEGER[]"),
                    Column("available_quantity", "INTEGER", nullable=True),
                ],
                id_column="langchain_id",
                store_metadata=False,
            ),
        )

        vs_custom_filter = await run_on_background(
            engine,
            AsyncPostgresVectorStore.create(
                engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_FILTER_TABLE,
                metadata_columns=[
                    "name",
                    "code",
                    "price",
                    "is_available",
                    "tags",
                    "inventory_location",
                    "available_quantity",
                ],
                id_column="langchain_id",
            ),
        )
        await run_on_background(
            engine, vs_custom_filter.aadd_documents(filter_docs, ids=ids)
        )
        yield vs_custom_filter

    @pytest_asyncio.fixture(scope="class")
    async def vs_hybrid_search_with_tsv_column(self, engine):
        hybrid_search_config = HybridSearchConfig(
            tsv_column="mycontent_tsv",
            tsv_lang="pg_catalog.english",
            fts_query="my_fts_query",
            fusion_function=reciprocal_rank_fusion,
            fusion_function_parameters={
                "rrf_k": 60,
                "fetch_top_k": 10,
            },
        )
        await run_on_background(
            engine,
            engine._ainit_vectorstore_table(
                HYBRID_SEARCH_TABLE1,
                VECTOR_SIZE,
                id_column=Column("myid", "TEXT"),
                content_column="mycontent",
                embedding_column="myembedding",
                metadata_columns=[
                    Column("page", "TEXT"),
                    Column("source", "TEXT"),
                    Column("doc_id_key", "TEXT"),
                ],
                metadata_json_column="mymetadata",  # ignored
                store_metadata=False,
                hybrid_search_config=hybrid_search_config,
            ),
        )

        vs_custom = await run_on_background(
            engine,
            AsyncPostgresVectorStore.create(
                engine,
                embedding_service=embeddings_service,
                table_name=HYBRID_SEARCH_TABLE1,
                id_column="myid",
                content_column="mycontent",
                embedding_column="myembedding",
                metadata_json_column="mymetadata",
                metadata_columns=["doc_id_key"],
                index_query_options=HNSWQueryOptions(ef_search=1),
                hybrid_search_config=hybrid_search_config,
            ),
        )
        await run_on_background(engine, vs_custom.aadd_documents(hybrid_docs))
        yield vs_custom

    async def test_asimilarity_search(self, engine, vs):
        results = await run_on_background(engine, vs.asimilarity_search("foo", k=1))
        assert len(results) == 1
        assert results == [Document(page_content="foo", id=ids[0])]
        results = await run_on_background(
            engine, vs.asimilarity_search("foo", k=1, filter={"content": "bar"})
        )
        assert results == [Document(page_content="bar", id=ids[1])]

    async def test_asimilarity_search_score(self, engine, vs):
        results = await run_on_background(
            engine, vs.asimilarity_search_with_score("foo")
        )
        assert len(results) == 4
        assert results[0][0] == Document(page_content="foo", id=ids[0])
        assert results[0][1] == 0

    async def test_asimilarity_search_by_vector(self, engine, vs):
        embedding = embeddings_service.embed_query("foo")
        results = await run_on_background(
            engine, vs.asimilarity_search_by_vector(embedding)
        )
        assert len(results) == 4
        assert results[0] == Document(page_content="foo", id=ids[0])
        results = await run_on_background(
            engine, vs.asimilarity_search_with_score_by_vector(embedding)
        )
        assert results[0][0] == Document(page_content="foo", id=ids[0])
        assert results[0][1] == 0

    async def test_similarity_search_with_relevance_scores_threshold_cosine(
        self, engine, vs
    ):
        score_threshold = {"score_threshold": 0}
        results = await run_on_background(
            engine,
            vs.asimilarity_search_with_relevance_scores("foo", **score_threshold),
        )
        # Note: Since tests use FakeEmbeddings which are non-normalized vectors, results might have scores beyond the range [0,1].
        # For a normalized embedding service, a threshold of zero will yield all matched documents.
        assert len(results) == 2

        score_threshold = {"score_threshold": 0.02}
        results = await run_on_background(
            engine,
            vs.asimilarity_search_with_relevance_scores("foo", **score_threshold),
        )
        assert len(results) == 2

        score_threshold = {"score_threshold": 0.9}
        results = await run_on_background(
            engine,
            vs.asimilarity_search_with_relevance_scores("foo", **score_threshold),
        )
        assert len(results) == 1
        assert results[0][0] == Document(page_content="foo", id=ids[0])

        score_threshold = {"score_threshold": 0.02}
        vs.distance_strategy = DistanceStrategy.EUCLIDEAN
        results = await run_on_background(
            engine,
            vs.asimilarity_search_with_relevance_scores("foo", **score_threshold),
        )
        assert len(results) == 1

    async def test_similarity_search_with_relevance_scores_threshold_euclidean(
        self, engine
    ):
        vs = await run_on_background(
            engine,
            AsyncPostgresVectorStore.create(
                engine,
                embedding_service=embeddings_service,
                table_name=DEFAULT_TABLE,
                distance_strategy=DistanceStrategy.EUCLIDEAN,
            ),
        )

        score_threshold = {"score_threshold": 0.9}
        results = await run_on_background(
            engine,
            vs.asimilarity_search_with_relevance_scores("foo", **score_threshold),
        )
        assert len(results) == 1
        assert results[0][0] == Document(page_content="foo", id=ids[0])

    async def test_amax_marginal_relevance_search(self, engine, vs):
        results = await run_on_background(
            engine, vs.amax_marginal_relevance_search("bar")
        )
        assert results[0] == Document(page_content="bar", id=ids[1])
        results = await run_on_background(
            engine, vs.amax_marginal_relevance_search("bar", filter={"content": "boo"})
        )
        assert results[0] == Document(page_content="boo", id=ids[3])

    async def test_amax_marginal_relevance_search_vector(self, engine, vs):
        embedding = embeddings_service.embed_query("bar")
        results = await run_on_background(
            engine, vs.amax_marginal_relevance_search_by_vector(embedding)
        )
        assert results[0] == Document(page_content="bar", id=ids[1])

    async def test_amax_marginal_relevance_search_vector_score(self, engine, vs):
        embedding = embeddings_service.embed_query("bar")
        results = await run_on_background(
            engine, vs.amax_marginal_relevance_search_with_score_by_vector(embedding)
        )
        assert results[0][0] == Document(page_content="bar", id=ids[1])

        results = await run_on_background(
            engine,
            vs.amax_marginal_relevance_search_with_score_by_vector(
                embedding, lambda_mult=0.75, fetch_k=10
            ),
        )
        assert results[0][0] == Document(page_content="bar", id=ids[1])

    async def test_similarity_search(self, engine, vs_custom):
        results = await run_on_background(
            engine, vs_custom.asimilarity_search("foo", k=1)
        )
        assert len(results) == 1
        assert results == [Document(page_content="foo", id=ids[0])]
        results = await run_on_background(
            engine,
            vs_custom.asimilarity_search("foo", k=1, filter={"mycontent": "bar"}),
        )
        assert results == [Document(page_content="bar", id=ids[1])]

    async def test_similarity_search_score(self, engine, vs_custom):
        results = await run_on_background(
            engine, vs_custom.asimilarity_search_with_score("foo")
        )
        assert len(results) == 4
        assert results[0][0] == Document(page_content="foo", id=ids[0])
        assert results[0][1] == 0

    async def test_similarity_search_by_vector(self, engine, vs_custom):
        embedding = embeddings_service.embed_query("foo")
        results = await run_on_background(
            engine, vs_custom.asimilarity_search_by_vector(embedding)
        )
        assert len(results) == 4
        assert results[0] == Document(page_content="foo", id=ids[0])
        results = await run_on_background(
            engine, vs_custom.asimilarity_search_with_score_by_vector(embedding)
        )
        assert results[0][0] == Document(page_content="foo", id=ids[0])
        assert results[0][1] == 0

    async def test_max_marginal_relevance_search(self, engine, vs_custom):
        results = await run_on_background(
            engine, vs_custom.amax_marginal_relevance_search("bar")
        )
        assert results[0] == Document(page_content="bar", id=ids[1])
        results = await run_on_background(
            engine,
            vs_custom.amax_marginal_relevance_search(
                "bar", filter={"mycontent": "boo"}
            ),
        )
        assert results[0] == Document(page_content="boo", id=ids[3])

    async def test_max_marginal_relevance_search_vector(self, engine, vs_custom):
        embedding = embeddings_service.embed_query("bar")
        results = await run_on_background(
            engine, vs_custom.amax_marginal_relevance_search_by_vector(embedding)
        )
        assert results[0] == Document(page_content="bar", id=ids[1])

    async def test_max_marginal_relevance_search_vector_score(self, engine, vs_custom):
        embedding = embeddings_service.embed_query("bar")
        results = await run_on_background(
            engine,
            vs_custom.amax_marginal_relevance_search_with_score_by_vector(embedding),
        )
        assert results[0][0] == Document(page_content="bar", id=ids[1])

        results = await run_on_background(
            engine,
            vs_custom.amax_marginal_relevance_search_with_score_by_vector(
                embedding, lambda_mult=0.75, fetch_k=10
            ),
        )
        assert results[0][0] == Document(page_content="bar", id=ids[1])

    async def test_aget_by_ids(self, engine, vs):
        test_ids = [ids[0]]
        results = await run_on_background(engine, vs.aget_by_ids(ids=test_ids))

        assert results[0] == Document(page_content="foo", id=ids[0])

    async def test_aget_by_ids_custom_vs(self, engine, vs_custom):
        test_ids = [ids[0]]
        results = await run_on_background(engine, vs_custom.aget_by_ids(ids=test_ids))

        assert results[0] == Document(page_content="foo", id=ids[0])

    def test_get_by_ids(self, vs):
        test_ids = [ids[0]]
        with pytest.raises(Exception):
            vs.get_by_ids(ids=test_ids)

    @pytest.mark.parametrize("test_filter, expected_ids", FILTERING_TEST_CASES)
    async def test_vectorstore_with_metadata_filters(
        self,
        engine,
        vs_custom_filter,
        test_filter,
        expected_ids,
    ):
        """Test end to end construction and search."""
        docs = await run_on_background(
            engine, vs_custom_filter.asimilarity_search("meow", k=5, filter=test_filter)
        )
        assert [doc.metadata["code"] for doc in docs] == expected_ids, test_filter

    async def test_asimilarity_hybrid_search_rrk(self, engine, vs):
        results = await run_on_background(
            engine,
            vs.asimilarity_search(
                "foo",
                k=1,
                hybrid_search_config=HybridSearchConfig(
                    fusion_function=reciprocal_rank_fusion
                ),
            ),
        )
        assert len(results) == 1
        assert results == [Document(page_content="foo", id=ids[0])]

        results = await run_on_background(
            engine,
            vs.asimilarity_search(
                "bar",
                k=1,
                filter={"content": {"$ne": "baz"}},
                hybrid_search_config=HybridSearchConfig(
                    fusion_function=reciprocal_rank_fusion,
                    fusion_function_parameters={
                        "rrf_k": 100,
                        "fetch_top_k": 10,
                    },
                    primary_top_k=1,
                    secondary_top_k=1,
                ),
            ),
        )
        assert results == [Document(page_content="bar", id=ids[1])]

    async def test_hybrid_search_weighted_sum_default(
        self, engine, vs_hybrid_search_with_tsv_column
    ):
        """Test hybrid search with default weighted sum (0.5 vector, 0.5 FTS)."""
        query = "apple"  # Should match "apple" in FTS and vector

        # The vs_hybrid_search_with_tsv_column instance is already configured for hybrid search.
        # Default fusion is weighted_sum_ranking with 0.5/0.5 weights.
        # fts_query will default to the main query.
        results_with_scores = await run_on_background(
            engine,
            vs_hybrid_search_with_tsv_column.asimilarity_search_with_score(query, k=3),
        )

        assert len(results_with_scores) > 1
        result_ids = [doc.metadata["doc_id_key"] for doc, score in results_with_scores]

        # Expect "hs_doc_apple_fruit" and "hs_doc_apple_tech" to be highly ranked.
        assert "hs_doc_apple_fruit" in result_ids

        # Scores should be floats (fused scores)
        for doc, score in results_with_scores:
            assert isinstance(score, float)

        # Check if sorted by score (descending for weighted_sum_ranking with positive scores)
        assert results_with_scores[0][1] >= results_with_scores[1][1]

    async def test_hybrid_search_weighted_sum_vector_bias(
        self, engine, vs_hybrid_search_with_tsv_column
    ):
        """Test weighted sum with higher weight for vector results."""
        query = "Apple Inc technology"  # More specific for vector similarity

        config = HybridSearchConfig(
            tsv_column="mycontent_tsv",  # Must match table setup
            fusion_function_parameters={
                "primary_results_weight": 0.8,  # Vector bias
                "secondary_results_weight": 0.2,
            },
            # fts_query will default to main query
        )
        results = await run_on_background(
            engine,
            vs_hybrid_search_with_tsv_column.asimilarity_search(
                query, k=2, hybrid_search_config=config
            ),
        )
        result_ids = [doc.metadata["doc_id_key"] for doc in results]

        assert len(result_ids) > 0
        assert result_ids[0] == "hs_doc_generic_tech"

    async def test_hybrid_search_weighted_sum_fts_bias(
        self, engine, vs_hybrid_search_with_tsv_column
    ):
        """Test weighted sum with higher weight for FTS results."""
        query = "fruit common tasty"  # Strong FTS signal for fruit docs

        config = HybridSearchConfig(
            tsv_column="mycontent_tsv",
            fusion_function=weighted_sum_ranking,
            fusion_function_parameters={
                "primary_results_weight": 0.01,
                "secondary_results_weight": 0.99,  # FTS bias
            },
        )
        results = await run_on_background(
            engine,
            vs_hybrid_search_with_tsv_column.asimilarity_search(
                query, k=2, hybrid_search_config=config
            ),
        )
        result_ids = [doc.metadata["doc_id_key"] for doc in results]

        assert len(result_ids) == 2
        assert "hs_doc_apple_fruit" in result_ids

    async def test_hybrid_search_reciprocal_rank_fusion(
        self, engine, vs_hybrid_search_with_tsv_column
    ):
        """Test hybrid search with Reciprocal Rank Fusion."""
        query = "technology company"

        # Configure RRF. primary_top_k and secondary_top_k control inputs to fusion.
        # fusion_function_parameters.fetch_top_k controls output count from RRF.
        config = HybridSearchConfig(
            tsv_column="mycontent_tsv",
            fusion_function=reciprocal_rank_fusion,
            primary_top_k=3,  # How many dense results to consider
            secondary_top_k=3,  # How many sparse results to consider
            fusion_function_parameters={
                "rrf_k": 60,
                "fetch_top_k": 2,
            },  # RRF specific params
        )
        results = await run_on_background(
            engine,
            vs_hybrid_search_with_tsv_column.asimilarity_search(
                query, k=2, hybrid_search_config=config
            ),
        )
        result_ids = [doc.metadata["doc_id_key"] for doc in results]

        assert len(result_ids) == 2
        # "hs_doc_apple_tech" (FTS: technology, company; Vector: Apple Inc technology)
        # "hs_doc_generic_tech" (FTS: technology; Vector: Technology drives innovation)
        # RRF should combine these ranks. "hs_doc_apple_tech" is likely higher.
        assert "hs_doc_apple_tech" in result_ids
        assert result_ids[0] == "hs_doc_apple_tech"  # Stronger combined signal

    async def test_hybrid_search_explicit_fts_query(
        self, engine, vs_hybrid_search_with_tsv_column
    ):
        """Test hybrid search when fts_query in HybridSearchConfig is different from main query."""
        main_vector_query = "Apple Inc."  # For vector search
        fts_specific_query = "fruit"  # For FTS

        config = HybridSearchConfig(
            tsv_column="mycontent_tsv",
            fts_query=fts_specific_query,  # Override FTS query
            fusion_function_parameters={  # Using default weighted_sum_ranking
                "primary_results_weight": 0.5,
                "secondary_results_weight": 0.5,
            },
        )
        results = await run_on_background(
            engine,
            vs_hybrid_search_with_tsv_column.asimilarity_search(
                main_vector_query, k=2, hybrid_search_config=config
            ),
        )
        result_ids = [doc.metadata["doc_id_key"] for doc in results]

        # Vector search for "Apple Inc.": hs_doc_apple_tech
        # FTS search for "fruit": hs_doc_apple_fruit, hs_doc_orange_fruit
        # Combined: hs_doc_apple_fruit (strong FTS) and hs_doc_apple_tech (strong vector) are candidates.
        # "hs_doc_apple_fruit" might get a boost if "Apple Inc." vector has some similarity to "apple fruit" doc.
        assert len(result_ids) > 0
        assert (
            "hs_doc_apple_fruit" in result_ids
            or "hs_doc_apple_tech" in result_ids
            or "hs_doc_orange_fruit" in result_ids
        )

    async def test_hybrid_search_with_filter(
        self, engine, vs_hybrid_search_with_tsv_column
    ):
        """Test hybrid search with a metadata filter applied."""
        query = "apple"
        # Filter to only include "tech" related apple docs using metadata
        # Assuming metadata_columns=["doc_id_key"] was set up for vs_hybrid_search_with_tsv_column
        doc_filter = {"doc_id_key": {"$eq": "hs_doc_apple_tech"}}

        config = HybridSearchConfig(
            tsv_column="mycontent_tsv",
        )
        results = await run_on_background(
            engine,
            vs_hybrid_search_with_tsv_column.asimilarity_search(
                query, k=2, filter=doc_filter, hybrid_search_config=config
            ),
        )
        result_ids = [doc.metadata["doc_id_key"] for doc in results]

        assert len(results) == 1
        assert result_ids[0] == "hs_doc_apple_tech"

    async def test_hybrid_search_fts_empty_results(
        self, engine, vs_hybrid_search_with_tsv_column
    ):
        """Test when FTS query yields no results, should fall back to vector search."""
        vector_query = "apple"
        no_match_fts_query = "zzyyxx_gibberish_term_for_fts_nomatch"

        config = HybridSearchConfig(
            tsv_column="mycontent_tsv",
            fts_query=no_match_fts_query,
            fusion_function_parameters={
                "primary_results_weight": 0.6,
                "secondary_results_weight": 0.4,
            },
        )
        results = await run_on_background(
            engine,
            vs_hybrid_search_with_tsv_column.asimilarity_search(
                vector_query, k=2, hybrid_search_config=config
            ),
        )
        result_ids = [doc.metadata["doc_id_key"] for doc in results]

        # Expect results based purely on vector search for "apple"
        assert len(result_ids) > 0
        assert "hs_doc_apple_fruit" in result_ids or "hs_doc_apple_tech" in result_ids
        # The top result should be one of the apple documents based on vector search
        assert results[0].metadata["doc_id_key"].startswith("hs_doc_apple_fruit")

    async def test_hybrid_search_vector_empty_results_effectively(
        self, engine, vs_hybrid_search_with_tsv_column
    ):
        """Test when vector query is very dissimilar to docs, should rely on FTS."""
        # This is hard to guarantee with fake embeddings, but we try.
        # A better way might be to use a filter that excludes all docs for the vector part,
        # but filters are applied to both.
        vector_query_far_off = "supercalifragilisticexpialidocious_vector_nomatch"
        fts_query_match = "orange fruit"  # Should match hs_doc_orange_fruit

        config = HybridSearchConfig(
            tsv_column="mycontent_tsv",
            fts_query=fts_query_match,
            fusion_function_parameters={
                "primary_results_weight": 0.4,
                "secondary_results_weight": 0.6,
            },
        )
        results = await run_on_background(
            engine,
            vs_hybrid_search_with_tsv_column.asimilarity_search(
                vector_query_far_off, k=1, hybrid_search_config=config
            ),
        )
        result_ids = [doc.metadata["doc_id_key"] for doc in results]

        # Expect results based purely on FTS search for "orange fruit"
        assert len(result_ids) == 1
        assert result_ids[0] == "hs_doc_orange_fruit"

    async def test_hybrid_search_without_tsv_column(self, engine):
        """Test hybrid search without a TSV column."""
        # This is hard to guarantee with fake embeddings, but we try.
        # A better way might be to use a filter that excludes all docs for the vector part,
        # but filters are applied to both.
        vector_query_far_off = "apple iphone tech is better designed than macs"
        fts_query_match = "apple fruit"

        config = HybridSearchConfig(
            tsv_column="mycontent_tsv",
            fts_query=fts_query_match,
            fusion_function_parameters={
                "primary_results_weight": 0.1,
                "secondary_results_weight": 0.9,
            },
        )
        await run_on_background(
            engine,
            engine._ainit_vectorstore_table(
                HYBRID_SEARCH_TABLE2,
                VECTOR_SIZE,
                id_column=Column("myid", "TEXT"),
                content_column="mycontent",
                embedding_column="myembedding",
                metadata_columns=[
                    Column("page", "TEXT"),
                    Column("source", "TEXT"),
                    Column("doc_id_key", "TEXT"),
                ],
                store_metadata=False,
                hybrid_search_config=config,
            ),
        )

        vs_with_tsv_column = await run_on_background(
            engine,
            AsyncPostgresVectorStore.create(
                engine,
                embedding_service=embeddings_service,
                table_name=HYBRID_SEARCH_TABLE2,
                id_column="myid",
                content_column="mycontent",
                embedding_column="myembedding",
                metadata_columns=["doc_id_key"],
                index_query_options=HNSWQueryOptions(ef_search=1),
                hybrid_search_config=config,
            ),
        )
        await run_on_background(engine, vs_with_tsv_column.aadd_documents(hybrid_docs))

        config_no_tsv = HybridSearchConfig(
            tsv_column="",  # no TSV column
            fts_query=fts_query_match,
            fusion_function_parameters={
                "primary_results_weight": 0.9,
                "secondary_results_weight": 0.1,
            },
        )
        vs_without_tsv_column = await run_on_background(
            engine,
            AsyncPostgresVectorStore.create(
                engine,
                embedding_service=embeddings_service,
                table_name=HYBRID_SEARCH_TABLE2,
                id_column="myid",
                content_column="mycontent",
                embedding_column="myembedding",
                metadata_columns=["doc_id_key"],
                index_query_options=HNSWQueryOptions(ef_search=1),
                hybrid_search_config=config_no_tsv,
            ),
        )

        results_with_tsv_column = await run_on_background(
            engine,
            vs_with_tsv_column.asimilarity_search(
                vector_query_far_off, k=1, hybrid_search_config=config
            ),
        )
        results_without_tsv_column = await run_on_background(
            engine,
            vs_without_tsv_column.asimilarity_search(
                vector_query_far_off, k=1, hybrid_search_config=config
            ),
        )
        result_ids_with_tsv_column = [
            doc.metadata["doc_id_key"] for doc in results_with_tsv_column
        ]
        result_ids_without_tsv_column = [
            doc.metadata["doc_id_key"] for doc in results_without_tsv_column
        ]

        # Expect results based purely on FTS search for "orange fruit"
        assert len(result_ids_with_tsv_column) == 1
        assert len(result_ids_without_tsv_column) == 1
        assert result_ids_with_tsv_column[0] == "hs_doc_apple_fruit"
        assert result_ids_without_tsv_column[0] == "hs_doc_apple_fruit"
