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

import asyncpg  # type: ignore
import pytest
import pytest_asyncio
from google.cloud.sql.connector import Connector, IPTypes
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import VARCHAR
from sqlalchemy.ext.asyncio import create_async_engine

from langchain_google_cloud_sql_pg import Column, PostgresEngine

DEFAULT_TABLE = "test_table" + str(uuid.uuid4()).replace("-", "_")
CUSTOM_TABLE = "test_table_custom" + str(uuid.uuid4()).replace("-", "_")
VECTOR_SIZE = 768

embeddings_service = DeterministicFakeEmbedding(size=VECTOR_SIZE)


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


@pytest.mark.asyncio
class TestEngineAsync:
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

    @pytest.fixture(scope="module")
    def user(self) -> str:
        return get_env_var("DB_USER", "database user for cloud sql")

    @pytest.fixture(scope="module")
    def password(self) -> str:
        return get_env_var("DB_PASSWORD", "database password for cloud sql")

    @pytest.fixture(scope="module")
    def iam_account(self) -> str:
        return get_env_var("IAM_ACCOUNT", "Cloud SQL IAM account email")

    @pytest_asyncio.fixture
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

    async def test_execute(self, engine):
        await engine._aexecute("SELECT 1")

    async def test_cross_env_execute(self, engine):
        engine._execute("SELECT 1")

    async def test_init_table(self, engine):
        await engine.ainit_vectorstore_table(DEFAULT_TABLE, VECTOR_SIZE)
        id = str(uuid.uuid4())
        content = "coffee"
        embedding = await embeddings_service.aembed_query(content)
        stmt = f"INSERT INTO {DEFAULT_TABLE} (langchain_id, content, embedding) VALUES ('{id}', '{content}','{embedding}');"
        await engine._aexecute(stmt)

    async def test_fetch(self, engine):
        results = await engine._afetch(f"SELECT * FROM {DEFAULT_TABLE}")
        assert len(results) > 0
        await engine._aexecute(f"DROP TABLE {DEFAULT_TABLE}")

    async def test_init_table_custom(self, engine):
        await engine.ainit_vectorstore_table(
            CUSTOM_TABLE,
            VECTOR_SIZE,
            id_column="uuid",
            content_column="my-content",
            embedding_column="my_embedding",
            metadata_columns=[Column("page", "TEXT"), Column("source", "TEXT")],
            store_metadata=True,
        )
        stmt = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{CUSTOM_TABLE}';"
        results = await engine._afetch(stmt)
        expected = [
            {"column_name": "uuid", "data_type": "uuid"},
            {"column_name": "my_embedding", "data_type": "USER-DEFINED"},
            {"column_name": "langchain_metadata", "data_type": "json"},
            {"column_name": "my-content", "data_type": "text"},
            {"column_name": "page", "data_type": "text"},
            {"column_name": "source", "data_type": "text"},
        ]
        for row in results:
            assert row in expected

        await engine._aexecute(f"DROP TABLE {CUSTOM_TABLE}")

    async def test_password(
        self,
        db_project,
        db_region,
        db_instance,
        db_name,
        user,
        password,
    ):
        PostgresEngine._connector = None
        engine = await PostgresEngine.afrom_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            database=db_name,
            user=user,
            password=password,
        )
        assert engine
        await engine._aexecute("SELECT 1")
        PostgresEngine._connector = None

    async def test_from_engine(
        self,
        db_project,
        db_region,
        db_instance,
        db_name,
        user,
        password,
    ):
        async with Connector() as connector:

            async def getconn() -> asyncpg.Connection:
                conn = await connector.connect_async(  # type: ignore
                    f"{db_project}:{db_region}:{db_instance}",
                    "asyncpg",
                    user=user,
                    password=password,
                    db=db_name,
                    enable_iam_auth=False,
                    ip_type=IPTypes.PUBLIC,
                )
                return conn

            engine = create_async_engine(
                "postgresql+asyncpg://",
                async_creator=getconn,
            )

            engine = PostgresEngine.from_engine(engine)
            await engine._aexecute("SELECT 1")

    async def test_column(self, engine):
        with pytest.raises(ValueError):
            Column("test", VARCHAR)
        with pytest.raises(ValueError):
            Column(1, "INTEGER")

    async def test_iam_account_override(
        self,
        db_project,
        db_instance,
        db_region,
        db_name,
        iam_account,
    ):
        engine = await PostgresEngine.afrom_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            database=db_name,
            iam_account_email=iam_account,
        )
        assert engine
        await engine._aexecute("SELECT 1")
        await engine._connector.close_async()
        await engine._engine.dispose()


@pytest.mark.asyncio
class TestEngineSync:
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

    @pytest.fixture(scope="module")
    def user(self) -> str:
        return get_env_var("DB_USER", "database user for cloud sql")

    @pytest.fixture(scope="module")
    def password(self) -> str:
        return get_env_var("DB_PASSWORD", "database password for cloud sql")

    @pytest.fixture(scope="module")
    def iam_account(self) -> str:
        return get_env_var("IAM_ACCOUNT", "Cloud SQL IAM account email")

    @pytest_asyncio.fixture
    def engine(self, db_project, db_region, db_instance, db_name):
        engine = PostgresEngine.from_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            database=db_name,
        )
        yield engine
        engine._run_as_sync(engine._connector.close_async())
        engine._run_as_sync(engine._engine.dispose())

    def test_execute(self, engine):
        engine._execute("SELECT 1")

    async def test_cross_env_execute(self, engine):
        await engine._aexecute("SELECT 1")

    async def test_init_table(self, engine):
        engine.init_vectorstore_table(DEFAULT_TABLE, VECTOR_SIZE)
        id = str(uuid.uuid4())
        content = "coffee"
        embedding = await embeddings_service.aembed_query(content)
        stmt = f"INSERT INTO {DEFAULT_TABLE} (langchain_id, content, embedding) VALUES ('{id}', '{content}','{embedding}');"
        engine._execute(stmt)

    async def test_fetch(self, engine):
        results = engine._fetch(f"SELECT * FROM {DEFAULT_TABLE}")
        assert len(results) > 0
        engine._execute(f"DROP TABLE {DEFAULT_TABLE}")

    async def test_init_table_custom(self, engine):
        engine.init_vectorstore_table(
            CUSTOM_TABLE,
            VECTOR_SIZE,
            id_column="uuid",
            content_column="my-content",
            embedding_column="my_embedding",
            metadata_columns=[Column("page", "TEXT"), Column("source", "TEXT")],
            store_metadata=True,
        )
        stmt = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{CUSTOM_TABLE}';"
        results = engine._fetch(stmt)
        expected = [
            {"column_name": "uuid", "data_type": "uuid"},
            {"column_name": "my_embedding", "data_type": "USER-DEFINED"},
            {"column_name": "langchain_metadata", "data_type": "json"},
            {"column_name": "my-content", "data_type": "text"},
            {"column_name": "page", "data_type": "text"},
            {"column_name": "source", "data_type": "text"},
        ]
        for row in results:
            assert row in expected

        engine._execute(f"DROP TABLE {CUSTOM_TABLE}")

    async def test_password(
        self,
        db_project,
        db_region,
        db_instance,
        db_name,
        user,
        password,
    ):
        PostgresEngine._connector = None
        engine = PostgresEngine.from_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            database=db_name,
            user=user,
            password=password,
            quota_project=db_project,
        )
        assert engine
        engine._execute("SELECT 1")
        PostgresEngine._connector = None

    async def test_engine_constructor_key(
        self,
        engine,
    ):
        key = object()
        with pytest.raises(Exception):
            PostgresEngine(key, engine)

    def test_iam_account_override(
        self,
        db_project,
        db_instance,
        db_region,
        db_name,
        iam_account,
    ):
        engine = PostgresEngine.from_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            database=db_name,
            iam_account_email=iam_account,
        )
        assert engine
        engine._execute("SELECT 1")
        engine._connector.close()
        engine._engine.dispose()
