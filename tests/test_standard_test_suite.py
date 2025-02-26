# Copyright 2025 Google LLC
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
from langchain_tests.integration_tests import VectorStoreIntegrationTests
from langchain_tests.integration_tests.vectorstores import EMBEDDING_SIZE
from sqlalchemy import text

from langchain_google_cloud_sql_pg import Column, PostgresEngine, PostgresVectorStore

DEFAULT_TABLE = "test_table_standard_test_suite" + str(uuid.uuid4())
DEFAULT_TABLE_SYNC = "test_table_sync_standard_test_suite" + str(uuid.uuid4())


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


async def aexecute(
    engine: PostgresEngine,
    query: str,
) -> None:
    async def run(engine, query):
        async with engine._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    await engine._run_as_async(run(engine, query))


@pytest.mark.filterwarnings("ignore")
@pytest.mark.asyncio
class TestStandardSuiteSync(VectorStoreIntegrationTests):
    @pytest.fixture(scope="module")
    def db_project(self) -> str:
        return get_env_var("PROJECT_ID", "project id for google cloud")

    @pytest.fixture(scope="module")
    def db_region(self) -> str:
        return get_env_var("REGION", "region for Cloud SQL instance")

    @pytest.fixture(scope="module")
    def db_instance(self) -> str:
        return get_env_var("INSTANCE_ID", "instance for Cloud SQL")

    @pytest.fixture(scope="module")
    def db_name(self) -> str:
        return get_env_var("DATABASE_ID", "database name on Cloud SQL instance")

    @pytest.fixture(scope="module")
    def user(self) -> str:
        return get_env_var("DB_USER", "database user for Cloud SQL")

    @pytest.fixture(scope="module")
    def password(self) -> str:
        return get_env_var("DB_PASSWORD", "database password for Cloud SQL")

    @pytest_asyncio.fixture(loop_scope="function")
    async def sync_engine(self, db_project, db_region, db_instance, db_name):
        sync_engine = PostgresEngine.from_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            database=db_name,
        )
        yield sync_engine
        await aexecute(sync_engine, f'DROP TABLE IF EXISTS "{DEFAULT_TABLE_SYNC}"')
        await sync_engine.close()

    @pytest.fixture(scope="function")
    def vectorstore(self, sync_engine):
        """Get an empty vectorstore for unit tests."""
        sync_engine.init_vectorstore_table(
            DEFAULT_TABLE_SYNC,
            EMBEDDING_SIZE,
            id_column=Column(name="langchain_id", data_type="VARCHAR", nullable=False),
        )

        vs = PostgresVectorStore.create_sync(
            sync_engine,
            embedding_service=self.get_embeddings(),
            table_name=DEFAULT_TABLE_SYNC,
        )
        yield vs


@pytest.mark.filterwarnings("ignore")
@pytest.mark.asyncio
class TestStandardSuiteAsync(VectorStoreIntegrationTests):
    @pytest.fixture(scope="module")
    def db_project(self) -> str:
        return get_env_var("PROJECT_ID", "project id for google cloud")

    @pytest.fixture(scope="module")
    def db_region(self) -> str:
        return get_env_var("REGION", "region for Cloud SQL instance")

    @pytest.fixture(scope="module")
    def db_instance(self) -> str:
        return get_env_var("INSTANCE_ID", "instance for Cloud SQL")

    @pytest.fixture(scope="module")
    def db_name(self) -> str:
        return get_env_var("DATABASE_ID", "database name on Cloud SQL instance")

    @pytest.fixture(scope="module")
    def user(self) -> str:
        return get_env_var("DB_USER", "database user for Cloud SQL")

    @pytest.fixture(scope="module")
    def password(self) -> str:
        return get_env_var("DB_PASSWORD", "database password for Cloud SQL")

    @pytest_asyncio.fixture(loop_scope="function")
    async def async_engine(self, db_project, db_region, db_instance, db_name):
        async_engine = await PostgresEngine.afrom_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            database=db_name,
        )
        yield async_engine
        await aexecute(async_engine, f'DROP TABLE IF EXISTS "{DEFAULT_TABLE}"')
        await async_engine.close()

    @pytest_asyncio.fixture(loop_scope="function")
    async def vectorstore(self, async_engine):
        """Get an empty vectorstore for unit tests."""
        await async_engine.ainit_vectorstore_table(
            DEFAULT_TABLE,
            EMBEDDING_SIZE,
            id_column=Column(name="langchain_id", data_type="VARCHAR", nullable=False),
        )

        vs = await PostgresVectorStore.create(
            async_engine,
            embedding_service=self.get_embeddings(),
            table_name=DEFAULT_TABLE,
        )

        yield vs
