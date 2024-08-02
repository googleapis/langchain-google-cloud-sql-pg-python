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

import pytest
import pytest_asyncio
from google.cloud.sql.connector import Connector, create_async_connector
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool

from langchain_google_cloud_sql_pg import (
    Column,
    PostgresEngine,
    PostgresVectorStore,
)

DEFAULT_TABLE = "test_table" + str(uuid.uuid4())
DEFAULT_TABLE_SYNC = "test_table_sync" + str(uuid.uuid4())
CUSTOM_TABLE = "test-table-custom" + str(uuid.uuid4())
VECTOR_SIZE = 768

embeddings_service = DeterministicFakeEmbedding(size=VECTOR_SIZE)

texts = ["foo", "bar", "baz"]
metadatas = [
    {"page": str(i), "source": "google.com"} for i in range(len(texts))
]
docs = [
    Document(page_content=texts[i], metadata=metadatas[i])
    for i in range(len(texts))
]

embeddings = [
    embeddings_service.embed_query(texts[i]) for i in range(len(texts))
]


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


@pytest.mark.asyncio(scope="class")
class TestVectorStore:
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

    @pytest.fixture(scope="module")
    def user(self) -> str:
        return get_env_var("DB_USER", "database user for cloud sql")

    @pytest.fixture(scope="module")
    def password(self) -> str:
        return get_env_var("DB_PASSWORD", "database password for cloud sql")

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
    def vs_sync(self, engine_sync):
        engine_sync.init_vectorstore_table(DEFAULT_TABLE_SYNC, VECTOR_SIZE)

        vs = PostgresVectorStore.create_sync(
            engine_sync,
            embedding_service=embeddings_service,
            table_name=DEFAULT_TABLE_SYNC,
        )
        yield vs
        engine_sync._execute(f'DROP TABLE IF EXISTS "{DEFAULT_TABLE_SYNC}"')

    @pytest_asyncio.fixture(scope="class")
    async def vs(self, engine):
        await engine.ainit_vectorstore_table(DEFAULT_TABLE, VECTOR_SIZE)
        vs = await PostgresVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=DEFAULT_TABLE,
        )
        yield vs
        await engine._aexecute(f'DROP TABLE IF EXISTS "{DEFAULT_TABLE}"')

    @pytest_asyncio.fixture(scope="class")
    async def vs_custom(self, engine):
        await engine.ainit_vectorstore_table(
            CUSTOM_TABLE,
            VECTOR_SIZE,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=[Column("page", "TEXT"), Column("source", "TEXT")],
            metadata_json_column="mymeta",
        )
        vs = await PostgresVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=CUSTOM_TABLE,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=["page", "source"],
            metadata_json_column="mymeta",
        )
        yield vs
        await engine._aexecute(f'DROP TABLE IF EXISTS "{CUSTOM_TABLE}"')

    async def test_init_with_constructor(self, engine):
        with pytest.raises(Exception):
            PostgresVectorStore(
                engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="noname",
                embedding_column="myembedding",
                metadata_columns=["page", "source"],
                metadata_json_column="mymeta",
            )

    async def test_post_init(self, engine):
        with pytest.raises(ValueError):
            await PostgresVectorStore.create(
                engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="noname",
                embedding_column="myembedding",
                metadata_columns=["page", "source"],
                metadata_json_column="mymeta",
            )

    async def test_aadd_texts(self, engine, vs):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_texts(texts, ids=ids)
        results = await engine._afetch(f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3

        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_texts(texts, metadatas, ids)
        results = await engine._afetch(f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 6
        await engine._aexecute(f'TRUNCATE TABLE "{DEFAULT_TABLE}"')

    async def test_cross_env_add_texts(self, engine, vs):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        vs.add_texts(texts, ids=ids)
        results = await engine._afetch(f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3
        vs.delete(ids)

    async def test_aadd_texts_edge_cases(self, engine, vs):
        texts = ["Taylor's", '"Swift"', "best-friend"]
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_texts(texts, ids=ids)
        results = await engine._afetch(f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3
        await engine._aexecute(f'TRUNCATE TABLE "{DEFAULT_TABLE}"')

    async def test_aadd_docs(self, engine, vs):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_documents(docs, ids=ids)
        results = await engine._afetch(f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3
        await engine._aexecute(f'TRUNCATE TABLE "{DEFAULT_TABLE}"')

    async def test_aadd_embedding(self, engine, vs):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs._aadd_embeddings(texts, embeddings, metadatas, ids)
        results = await engine._afetch(f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3
        await engine._aexecute(f'TRUNCATE TABLE "{DEFAULT_TABLE}"')

    async def test_aadd_embedding_without_id(self, engine, vs):
        await vs._aadd_embeddings(texts, embeddings, metadatas)
        results = await engine._afetch(f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3
        assert results[0]["langchain_id"]
        await engine._aexecute(f'TRUNCATE TABLE "{DEFAULT_TABLE}"')

    async def test_adelete(self, engine, vs):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_texts(texts, ids=ids)
        results = await engine._afetch(f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3
        # delete an ID
        await vs.adelete([ids[0]])
        results = await engine._afetch(f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 2

    async def test_aadd_texts_custom(self, engine, vs_custom):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs_custom.aadd_texts(texts, ids=ids)
        results = await engine._afetch(f'SELECT * FROM "{CUSTOM_TABLE}"')
        assert len(results) == 3
        assert results[0]["mycontent"] == "foo"
        assert results[0]["myembedding"]
        assert results[0]["page"] is None
        assert results[0]["source"] is None

        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs_custom.aadd_texts(texts, metadatas, ids)
        results = await engine._afetch(f'SELECT * FROM "{CUSTOM_TABLE}"')
        assert len(results) == 6
        await engine._aexecute(f'TRUNCATE TABLE "{CUSTOM_TABLE}"')

    async def test_aadd_docs_custom(self, engine, vs_custom):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        docs = [
            Document(
                page_content=texts[i],
                metadata={"page": str(i), "source": "google.com"},
            )
            for i in range(len(texts))
        ]
        await vs_custom.aadd_documents(docs, ids=ids)

        results = await engine._afetch(f'SELECT * FROM "{CUSTOM_TABLE}"')
        assert len(results) == 3
        assert results[0]["mycontent"] == "foo"
        assert results[0]["myembedding"]
        assert results[0]["page"] == "0"
        assert results[0]["source"] == "google.com"
        await engine._aexecute(f'TRUNCATE TABLE "{CUSTOM_TABLE}"')

    async def test_aadd_embedding_custom(self, engine, vs_custom):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs_custom._aadd_embeddings(texts, embeddings, metadatas, ids)
        results = await engine._afetch(f'SELECT * FROM "{CUSTOM_TABLE}"')
        assert len(results) == 3
        await engine._aexecute(f'TRUNCATE TABLE "{CUSTOM_TABLE}"')

    async def test_adelete_custom(self, engine, vs_custom):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs_custom.aadd_texts(texts, ids=ids)
        results = await engine._afetch(f'SELECT * FROM "{CUSTOM_TABLE}"')
        content = [result["mycontent"] for result in results]
        assert len(results) == 3
        assert "foo" in content
        # delete an ID
        await vs_custom.adelete([ids[0]])
        results = await engine._afetch(f'SELECT * FROM "{CUSTOM_TABLE}"')
        content = [result["mycontent"] for result in results]
        assert len(results) == 2
        assert "foo" not in content

    async def test_add_docs(self, engine_sync, vs_sync):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        vs_sync.add_documents(docs, ids=ids)
        results = engine_sync._fetch(f'SELECT * FROM "{DEFAULT_TABLE_SYNC}"')
        assert len(results) == 3
        vs_sync.delete(ids)

    async def test_add_texts(self, engine_sync, vs_sync):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        vs_sync.add_texts(texts, ids=ids)
        results = engine_sync._fetch(f'SELECT * FROM "{DEFAULT_TABLE_SYNC}"')
        assert len(results) == 3
        # engine_sync._execute(f'TRUNCATE TABLE "{CUSTOM_TABLE}"')
        vs_sync.delete(ids)

    async def test_cross_env(self, engine_sync, vs_sync):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs_sync.aadd_texts(texts, ids=ids)
        results = engine_sync._fetch(f'SELECT * FROM "{DEFAULT_TABLE_SYNC}"')
        assert len(results) == 3
        await vs_sync.adelete(ids)

    async def test_ignore_metadata_columns(self, vs_custom):
        column_to_ignore = "source"
        vs = await PostgresVectorStore.create(
            vs_custom.engine,
            embedding_service=embeddings_service,
            table_name=CUSTOM_TABLE,
            ignore_metadata_columns=[column_to_ignore],
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_json_column="mymeta",
        )
        assert column_to_ignore not in vs.metadata_columns

    async def test_create_vectorstore_with_invalid_parameters(self, vs_custom):
        with pytest.raises(ValueError):
            await PostgresVectorStore.create(
                vs_custom.engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="mycontent",
                embedding_column="myembedding",
                metadata_columns=["random_column"],  # invalid metadata column
            )
        with pytest.raises(ValueError):
            await PostgresVectorStore.create(
                vs_custom.engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="langchain_id",  # invalid content column type
                embedding_column="myembedding",
                metadata_columns=["random_column"],
            )
        with pytest.raises(ValueError):
            await PostgresVectorStore.create(
                vs_custom.engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="mycontent",
                embedding_column="random_column",  # invalid embedding column
                metadata_columns=["random_column"],
            )
        with pytest.raises(ValueError):
            await PostgresVectorStore.create(
                vs_custom.engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="mycontent",
                embedding_column="langchain_id",  # invalid embedding column data type
                metadata_columns=["random_column"],
            )

    async def test_from_engine(
        self,
        db_project,
        db_region,
        db_instance,
        db_name,
        user,
        password,
    ):
        async def init_connection_pool(connector: Connector) -> AsyncEngine:
            async def getconn():
                conn = await connector.connect_async(
                    f"{db_project}:{db_region}:{db_instance}",
                    "asyncpg",
                    user=user,
                    password=password,
                    db=db_name,
                    enable_iam_auth=False,
                    ip_type="PUBLIC",
                )
                return conn

            pool = create_async_engine(
                "postgresql+asyncpg://",
                async_creator=getconn,
                poolclass=NullPool,
            )
            return pool

        connector = await create_async_connector()
        pool = await init_connection_pool(connector)
        engine = PostgresEngine.from_engine(pool)
        table_name = "from_engine_test" + str(uuid.uuid4())
        await engine.ainit_vectorstore_table(table_name, VECTOR_SIZE)

        vs = await PostgresVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=table_name,
        )
        await vs.aadd_texts(["foo"])
        r1 = await vs.asimilarity_search("foo")
        assert len(r1) == 1
        r2 = vs.similarity_search("foo")
        assert len(r2) == 1
        await engine._aexecute(f'DROP TABLE IF EXISTS "{table_name}"')
