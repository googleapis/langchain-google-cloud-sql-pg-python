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

from langchain_google_cloud_sql_pg import CloudSQLVectorStore, Column, PostgreSQLEngine

DEFAULT_TABLE = "test_table" + str(uuid.uuid4()).replace("-", "_")
CUSTOM_TABLE = "test_table_custom" + str(uuid.uuid4()).replace("-", "_")
VECTOR_SIZE = 768


class FakeEmbeddingsWithDimension(FakeEmbeddings):
    """Fake embeddings functionality for testing."""

    size: int = VECTOR_SIZE

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings."""
        return [
            [float(1.0)] * (VECTOR_SIZE - 1) + [float(i)] for i in range(len(texts))
        ]

    def embed_query(self, text: str = "default") -> List[float]:
        """Return simple embeddings."""
        return [float(1.0)] * (VECTOR_SIZE - 1) + [float(0.0)]


embeddings_service = FakeEmbeddingsWithDimension()

texts = ["foo", "bar", "baz"]
metadatas = [{"page": str(i), "source": "google.com"} for i in range(len(texts))]
docs = [
    Document(page_content=texts[i], metadata=metadatas[i]) for i in range(len(texts))
]

embeddings = [embeddings_service.embed_query() for i in range(len(texts))]


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


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
        return get_env_var("DATABASE_ID", "instance for cloud sql")

    @pytest_asyncio.fixture
    async def engine(self, db_project, db_region, db_instance, db_name):
        engine = await PostgreSQLEngine.afrom_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            database=db_name,
        )
        await engine.init_vectorstore_table(DEFAULT_TABLE, VECTOR_SIZE)
        await engine.init_vectorstore_table(
            CUSTOM_TABLE,
            VECTOR_SIZE,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=[Column("page", "TEXT"), Column("source", "TEXT")],
            store_metadata=False,
        )
        yield engine
        await engine._aexecute(f"DROP TABLE IF EXISTS {DEFAULT_TABLE}")
        await engine._aexecute(f"DROP TABLE IF EXISTS {CUSTOM_TABLE}")

    async def test_afrom_texts(self, engine):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await CloudSQLVectorStore.afrom_texts(
            texts,
            FakeEmbeddingsWithDimension(),
            engine,
            DEFAULT_TABLE,
            metadatas=metadatas,
            ids=ids,
        )
        results = await engine._afetch(f"SELECT * FROM {DEFAULT_TABLE}")
        assert len(results) == 3
        await engine._aexecute(f"TRUNCATE TABLE {DEFAULT_TABLE}")

    async def test_afrom_texts_sync(self, engine):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        CloudSQLVectorStore.from_texts(
            texts,
            FakeEmbeddingsWithDimension(),
            engine,
            DEFAULT_TABLE,
            metadatas=metadatas,
            ids=ids,
        )
        results = await engine._afetch(f"SELECT * FROM {DEFAULT_TABLE}")
        assert len(results) == 3
        await engine._aexecute(f"TRUNCATE TABLE {DEFAULT_TABLE}")

    async def test_afrom_docs(self, engine):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await CloudSQLVectorStore.afrom_documents(
            docs,
            FakeEmbeddingsWithDimension(),
            engine,
            DEFAULT_TABLE,
            ids=ids,
        )
        results = await engine._afetch(f"SELECT * FROM {DEFAULT_TABLE}")
        assert len(results) == 3
        await engine._aexecute(f"TRUNCATE TABLE {DEFAULT_TABLE}")

    async def test_afrom_docs_sync(self, engine):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        CloudSQLVectorStore.from_documents(
            docs,
            FakeEmbeddingsWithDimension(),
            engine,
            DEFAULT_TABLE,
            ids=ids,
        )
        results = await engine._afetch(f"SELECT * FROM {DEFAULT_TABLE}")
        assert len(results) == 3
        await engine._aexecute(f"TRUNCATE TABLE {DEFAULT_TABLE}")

    async def test_from_texts(self, engine):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await CloudSQLVectorStore.afrom_texts(
            texts,
            FakeEmbeddingsWithDimension(),
            engine,
            DEFAULT_TABLE,
            metadatas=metadatas,
            ids=ids,
        )
        results = await engine._afetch(f"SELECT * FROM {DEFAULT_TABLE}")
        assert len(results) == 3
        await CloudSQLVectorStore.afrom_texts(
            texts,
            FakeEmbeddingsWithDimension(),
            engine,
            DEFAULT_TABLE,
            metadatas=metadatas,
            ids=ids,
            overwrite_existing=True,
        )
        results = await engine._afetch(f"SELECT * FROM {DEFAULT_TABLE}")
        assert len(results) == 3

    async def test_from_docs(self, engine):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await CloudSQLVectorStore.afrom_documents(
            docs,
            FakeEmbeddingsWithDimension(),
            engine,
            DEFAULT_TABLE,
            ids=ids,
        )
        results = await engine._afetch(f"SELECT * FROM {DEFAULT_TABLE}")
        assert len(results) == 3
        await CloudSQLVectorStore.afrom_documents(
            docs,
            FakeEmbeddingsWithDimension(),
            engine,
            DEFAULT_TABLE,
            ids=ids,
            overwrite_existing=True,
        )
        results = await engine._afetch(f"SELECT * FROM {DEFAULT_TABLE}")
        assert len(results) == 3

    async def test_afrom_texts_custom(self, engine):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await CloudSQLVectorStore.afrom_texts(
            texts,
            FakeEmbeddingsWithDimension(),
            engine,
            CUSTOM_TABLE,
            ids=ids,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=["page", "source"],
        )
        results = await engine._afetch(f"SELECT * FROM {CUSTOM_TABLE}")
        assert len(results) == 3
        assert results[0]["mycontent"] == "foo"
        assert results[0]["myembedding"]
        assert results[0]["page"] is None
        assert results[0]["source"] is None

        ids = [str(uuid.uuid4()) for i in range(len(texts))]

    async def test_afrom_docs_custom(self, engine):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        docs = [
            Document(
                page_content=texts[i],
                metadata={"page": str(i), "source": "google.com"},
            )
            for i in range(len(texts))
        ]
        await CloudSQLVectorStore.afrom_documents(
            docs,
            FakeEmbeddingsWithDimension(),
            engine,
            CUSTOM_TABLE,
            ids=ids,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=["page", "source"],
        )

        results = await engine._afetch(f"SELECT * FROM {CUSTOM_TABLE}")
        assert len(results) == 3
        assert results[0]["mycontent"] == "foo"
        assert results[0]["myembedding"]
        assert results[0]["page"] == "0"
        assert results[0]["source"] == "google.com"
        await engine._aexecute(f"TRUNCATE TABLE {CUSTOM_TABLE}")
