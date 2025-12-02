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
from sqlalchemy import text

from langchain_google_cloud_sql_pg import (  # type: ignore
    HybridSearchConfig,
    PostgresEngine,
)
from langchain_google_cloud_sql_pg.async_vectorstore import AsyncPostgresVectorStore
from langchain_google_cloud_sql_pg.indexes import (
    DEFAULT_INDEX_NAME_SUFFIX,
    DistanceStrategy,
    HNSWIndex,
    IVFFlatIndex,
)

UUID_STR = str(uuid.uuid4()).replace("-", "_")
DEFAULT_TABLE = "table" + UUID_STR
SIMPLE_TABLE = "simple" + UUID_STR
DEFAULT_HYBRID_TABLE = "hybrid" + UUID_STR
DEFAULT_INDEX_NAME = DEFAULT_INDEX_NAME_SUFFIX + UUID_STR
VECTOR_SIZE = 768

embeddings_service = DeterministicFakeEmbedding(size=VECTOR_SIZE)

texts = ["foo", "bar", "baz"]
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


# Helper to bridge the Main Test Loop and the Engine Background Loop
async def run_on_background(engine: PostgresEngine, coro: Coroutine) -> Any:
    """Runs a coroutine on the engine's background loop."""
    if engine._loop:
        return await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(coro, engine._loop)
        )
    return await coro


async def aexecute(engine: PostgresEngine, query: str) -> None:
    async def _impl():
        async with engine._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    await run_on_background(engine, _impl())


@pytest.mark.asyncio(scope="class")
class TestIndex:
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
        await aexecute(engine, f"DROP TABLE IF EXISTS {DEFAULT_HYBRID_TABLE}")
        await aexecute(engine, f"DROP TABLE IF EXISTS {SIMPLE_TABLE}")
        await engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def vs(self, engine):
        await run_on_background(
            engine,
            engine._ainit_vectorstore_table(
                DEFAULT_TABLE, VECTOR_SIZE, overwrite_existing=True
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

        await run_on_background(engine, vs.aadd_texts(texts, ids=ids))
        await run_on_background(engine, vs.adrop_vector_index())
        yield vs

    async def test_apply_default_name_vector_index(self, engine):
        await run_on_background(
            engine,
            engine._ainit_vectorstore_table(
                SIMPLE_TABLE, VECTOR_SIZE, overwrite_existing=True
            ),
        )

        vs = await run_on_background(
            engine,
            AsyncPostgresVectorStore.create(
                engine,
                embedding_service=embeddings_service,
                table_name=SIMPLE_TABLE,
            ),
        )
        await run_on_background(engine, vs.aadd_texts(texts, ids=ids))
        await run_on_background(engine, vs.adrop_vector_index())

        index = HNSWIndex()
        await run_on_background(engine, vs.aapply_vector_index(index))
        assert await run_on_background(engine, vs.is_valid_index())
        await run_on_background(engine, vs.adrop_vector_index())

    async def test_aapply_vector_index(self, engine, vs):
        await run_on_background(engine, vs.adrop_vector_index(DEFAULT_INDEX_NAME))
        index = HNSWIndex(name=DEFAULT_INDEX_NAME)
        await run_on_background(engine, vs.aapply_vector_index(index))
        assert await run_on_background(engine, vs.is_valid_index(DEFAULT_INDEX_NAME))
        await run_on_background(engine, vs.adrop_vector_index())

    async def test_areindex(self, engine, vs):
        if not await run_on_background(engine, vs.is_valid_index(DEFAULT_INDEX_NAME)):
            index = HNSWIndex()
            await run_on_background(engine, vs.aapply_vector_index(index))
        await run_on_background(engine, vs.areindex(DEFAULT_INDEX_NAME))
        await run_on_background(engine, vs.areindex(DEFAULT_INDEX_NAME))
        assert await run_on_background(engine, vs.is_valid_index(DEFAULT_INDEX_NAME))
        await run_on_background(engine, vs.adrop_vector_index())

    async def test_dropindex(self, engine, vs):
        await run_on_background(engine, vs.adrop_vector_index(DEFAULT_INDEX_NAME))
        result = await run_on_background(engine, vs.is_valid_index(DEFAULT_INDEX_NAME))
        assert not result

    async def test_aapply_vector_index_ivfflat(self, engine, vs):
        await run_on_background(engine, vs.adrop_vector_index(DEFAULT_INDEX_NAME))
        index = IVFFlatIndex(
            name=DEFAULT_INDEX_NAME, distance_strategy=DistanceStrategy.EUCLIDEAN
        )
        await run_on_background(
            engine, vs.aapply_vector_index(index, concurrently=True)
        )
        assert await run_on_background(engine, vs.is_valid_index(DEFAULT_INDEX_NAME))
        index = IVFFlatIndex(
            name="secondindex",
            distance_strategy=DistanceStrategy.INNER_PRODUCT,
        )
        await run_on_background(engine, vs.aapply_vector_index(index))
        assert await run_on_background(engine, vs.is_valid_index("secondindex"))
        await run_on_background(engine, vs.adrop_vector_index("secondindex"))
        await run_on_background(engine, vs.adrop_vector_index(DEFAULT_INDEX_NAME))

    async def test_is_valid_index(self, engine, vs):
        is_valid = await run_on_background(engine, vs.is_valid_index("invalid_index"))
        assert is_valid == False

    async def test_aapply_hybrid_search_index_table_without_tsv_column(
        self, engine, vs
    ):
        # overwriting vs to get a hybrid vs
        tsv_index_name = "index_without_tsv_column_" + UUID_STR
        vs = await run_on_background(
            engine,
            AsyncPostgresVectorStore.create(
                engine,
                embedding_service=embeddings_service,
                table_name=DEFAULT_TABLE,
                hybrid_search_config=HybridSearchConfig(index_name=tsv_index_name),
            ),
        )
        is_valid_index = await run_on_background(
            engine, vs.is_valid_index(tsv_index_name)
        )
        assert is_valid_index == False
        await run_on_background(engine, vs.aapply_hybrid_search_index())
        assert await run_on_background(engine, vs.is_valid_index(tsv_index_name))
        await run_on_background(engine, vs.adrop_vector_index(tsv_index_name))
        is_valid_index = await run_on_background(
            engine, vs.is_valid_index(tsv_index_name)
        )
        assert is_valid_index == False

    async def test_aapply_hybrid_search_index_table_with_tsv_column(self, engine):
        tsv_index_name = "index_without_tsv_column_" + UUID_STR
        config = HybridSearchConfig(
            tsv_column="tsv_column",
            tsv_lang="pg_catalog.english",
            index_name=tsv_index_name,
        )
        await run_on_background(
            engine,
            engine._ainit_vectorstore_table(
                DEFAULT_HYBRID_TABLE,
                VECTOR_SIZE,
                hybrid_search_config=config,
            ),
        )
        vs = await run_on_background(
            engine,
            AsyncPostgresVectorStore.create(
                engine,
                embedding_service=embeddings_service,
                table_name=DEFAULT_HYBRID_TABLE,
                hybrid_search_config=config,
            ),
        )

        is_valid_index = await run_on_background(
            engine, vs.is_valid_index(tsv_index_name)
        )
        assert is_valid_index == False
        await run_on_background(engine, vs.aapply_hybrid_search_index())
        assert await run_on_background(engine, vs.is_valid_index(tsv_index_name))
        await run_on_background(engine, vs.areindex(tsv_index_name))
        assert await run_on_background(engine, vs.is_valid_index(tsv_index_name))
        await run_on_background(engine, vs.adrop_vector_index(tsv_index_name))
        is_valid_index = await run_on_background(
            engine, vs.is_valid_index(tsv_index_name)
        )
        assert is_valid_index == False
