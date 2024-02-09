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

import json
import os
import uuid
from typing import List

import pytest
import pytest_asyncio
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.documents import Document
from langchain_google_vertexai import VertexAIEmbeddings

from langchain_google_cloud_sql_pg import Column, PostgreSQLEngine

PROJECT_ID = os.environ.get("PROJECT_ID")
INSTANCE = os.environ.get("INSTANCE_ID")
DATABASE = os.environ.get("DATABASE_ID")
REGION = os.environ.get("REGION")
DEFAULT_TABLE = "test_table" + str(uuid.uuid4()).replace("-", "_")
CUSTOM_TABLE = "test_table_custom" + str(uuid.uuid4()).replace("-", "_")
VECTOR_SIZE = 768


class FakeEmbeddingsWithDimension(FakeEmbeddings):
    """Fake embeddings functionality for testing."""

    size: int = VECTOR_SIZE

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings."""
        return [
            [float(1.0)] * (VECTOR_SIZE - 1) + [float(i)]
            for i in range(len(texts))
        ]

    def embed_query(self, text: str = "default") -> List[float]:
        """Return simple embeddings."""
        return [float(1.0)] * (VECTOR_SIZE - 1) + [float(0.0)]


embeddings_service = FakeEmbeddingsWithDimension()


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

    @pytest_asyncio.fixture
    async def engine(self, db_project, db_region, db_instance, db_name):
        engine = await PostgreSQLEngine.afrom_instance(
            project_id=PROJECT_ID,
            instance=INSTANCE,
            region=REGION,
            database=DATABASE,
        )
        yield engine

    async def test_execute(self, engine):
        await engine._aexecute("SELECT 1")

    async def test_init_table(self, engine):
        await engine.init_vectorstore_table(DEFAULT_TABLE, VECTOR_SIZE)
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
        await engine.init_vectorstore_table(
            CUSTOM_TABLE,
            VECTOR_SIZE,
            id_column="uuid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=[Column("page", "TEXT"), Column("source", "TEXT")],
            store_metadata=True,
        )
        stmt = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{CUSTOM_TABLE}';"
        results = await engine._afetch(stmt)
        expected = [
            {"column_name": "uuid", "data_type": "uuid"},
            {"column_name": "myembedding", "data_type": "USER-DEFINED"},
            {"column_name": "langchain_metadata", "data_type": "json"},
            {"column_name": "mycontent", "data_type": "text"},
            {"column_name": "page", "data_type": "text"},
            {"column_name": "source", "data_type": "text"},
        ]
        for row in results:
            assert row in expected

        await engine._aexecute(f"DROP TABLE {CUSTOM_TABLE}")

    def test_sync_engine(self):
        engine = PostgreSQLEngine.from_instance(
            project_id=PROJECT_ID,
            instance=INSTANCE,
            region=REGION,
            database=DATABASE,
        )
        assert engine
