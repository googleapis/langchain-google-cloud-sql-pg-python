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
from typing import Sequence

import pytest
import pytest_asyncio
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import text
from sqlalchemy.engine.row import RowMapping

from langchain_google_cloud_sql_pg import Column, PostgresEngine
from langchain_google_cloud_sql_pg.async_vectorstore import AsyncPostgresVectorStore

DEFAULT_TABLE = "test_table" + str(uuid.uuid4()).replace("-", "_")
DEFAULT_TABLE_SYNC = "test_table_sync" + str(uuid.uuid4()).replace("-", "_")
CUSTOM_TABLE = "test_table_custom" + str(uuid.uuid4()).replace("-", "_")
CUSTOM_TABLE_WITH_INT_ID = "test_table_custom_with_int_it" + str(uuid.uuid4()).replace(
    "-", "_"
)
VECTOR_SIZE = 768


embeddings_service = DeterministicFakeEmbedding(size=VECTOR_SIZE)

texts = ["foo", "bar", "baz"]
metadatas = [{"page": str(i), "source": "google.com"} for i in range(len(texts))]
docs = [
    Document(page_content=texts[i], metadata=metadatas[i]) for i in range(len(texts))
]

embeddings = [embeddings_service.embed_query(texts[i]) for i in range(len(texts))]


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


async def aexecute(engine: PostgresEngine, query: str) -> None:
    async with engine._pool.connect() as conn:
        await conn.execute(text(query))
        await conn.commit()


async def afetch(engine: PostgresEngine, query: str) -> Sequence[RowMapping]:
    async with engine._pool.connect() as conn:
        result = await conn.execute(text(query))
        result_map = result.mappings()
        result_fetch = result_map.fetchall()
    return result_fetch


@pytest.mark.asyncio
class TestVectorStoreFromMethods:
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
        return get_env_var("DATABASE_ID", "database name on cloud sql instance")

    @pytest_asyncio.fixture
    async def engine(self, db_project, db_region, db_instance, db_name):
        engine = await PostgresEngine.afrom_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            database=db_name,
        )
        await engine._ainit_vectorstore_table(DEFAULT_TABLE, VECTOR_SIZE)
        await engine._ainit_vectorstore_table(
            CUSTOM_TABLE,
            VECTOR_SIZE,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=[Column("page", "TEXT"), Column("source", "TEXT")],
            store_metadata=False,
        )
        await engine._ainit_vectorstore_table(
            CUSTOM_TABLE_WITH_INT_ID,
            VECTOR_SIZE,
            id_column=Column(name="integer_id", data_type="INTEGER", nullable="False"),
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=[Column("page", "TEXT"), Column("source", "TEXT")],
            store_metadata=False,
        )
        yield engine
        await aexecute(engine, f"DROP TABLE IF EXISTS {DEFAULT_TABLE}")
        await aexecute(engine, f"DROP TABLE IF EXISTS {CUSTOM_TABLE}")
        await aexecute(engine, f"DROP TABLE IF EXISTS {CUSTOM_TABLE_WITH_INT_ID}")
        await engine.close()

    async def test_afrom_texts(self, engine):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await AsyncPostgresVectorStore.afrom_texts(
            texts,
            embeddings_service,
            engine,
            DEFAULT_TABLE,
            metadatas=metadatas,
            ids=ids,
        )
        results = await afetch(engine, f"SELECT * FROM {DEFAULT_TABLE}")
        assert len(results) == 3
        await aexecute(engine, f"TRUNCATE TABLE {DEFAULT_TABLE}")

    async def test_afrom_docs(self, engine):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await AsyncPostgresVectorStore.afrom_documents(
            docs,
            embeddings_service,
            engine,
            DEFAULT_TABLE,
            ids=ids,
        )
        results = await afetch(engine, f"SELECT * FROM {DEFAULT_TABLE}")
        assert len(results) == 3
        await aexecute(engine, f"TRUNCATE TABLE {DEFAULT_TABLE}")

    async def test_afrom_texts_custom(self, engine):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await AsyncPostgresVectorStore.afrom_texts(
            texts,
            embeddings_service,
            engine,
            CUSTOM_TABLE,
            ids=ids,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=["page", "source"],
        )
        results = await afetch(engine, f"SELECT * FROM {CUSTOM_TABLE}")
        assert len(results) == 3
        assert results[0]["mycontent"] == "foo"
        assert results[0]["myembedding"]
        assert results[0]["page"] is None
        assert results[0]["source"] is None

    async def test_afrom_docs_custom(self, engine):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        docs = [
            Document(
                page_content=texts[i],
                metadata={"page": str(i), "source": "google.com"},
            )
            for i in range(len(texts))
        ]
        await AsyncPostgresVectorStore.afrom_documents(
            docs,
            embeddings_service,
            engine,
            CUSTOM_TABLE,
            ids=ids,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=["page", "source"],
        )

        results = await afetch(engine, f"SELECT * FROM {CUSTOM_TABLE}")
        assert len(results) == 3
        assert results[0]["mycontent"] == "foo"
        assert results[0]["myembedding"]
        assert results[0]["page"] == "0"
        assert results[0]["source"] == "google.com"
        await aexecute(engine, f"TRUNCATE TABLE {CUSTOM_TABLE}")

    async def test_afrom_docs_custom_with_int_id(self, engine):
        ids = [i for i in range(len(texts))]
        docs = [
            Document(
                page_content=texts[i],
                metadata={"page": str(i), "source": "google.com"},
            )
            for i in range(len(texts))
        ]
        await AsyncPostgresVectorStore.afrom_documents(
            docs,
            embeddings_service,
            engine,
            CUSTOM_TABLE_WITH_INT_ID,
            ids=ids,
            id_column="integer_id",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=["page", "source"],
        )

        results = await afetch(engine, f"SELECT * FROM {CUSTOM_TABLE_WITH_INT_ID}")
        assert len(results) == 3
        for row in results:
            assert isinstance(row["integer_id"], int)
        await aexecute(engine, f"TRUNCATE TABLE {CUSTOM_TABLE_WITH_INT_ID}")
