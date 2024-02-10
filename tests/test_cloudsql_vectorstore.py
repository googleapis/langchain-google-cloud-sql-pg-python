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
        yield engine
        await engine._aexecute(f"DROP TABLE IF EXISTS {DEFAULT_TABLE}")

    @pytest_asyncio.fixture
    def vs(self, engine):
        vs = CloudSQLVectorStore(
            engine,
            embedding_service=FakeEmbeddingsWithDimension(),
            table_name=DEFAULT_TABLE,
        )
        yield vs

    @pytest_asyncio.fixture
    async def vs_custom(self, engine):
        await engine.init_vectorstore_table(
            CUSTOM_TABLE,
            VECTOR_SIZE,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=[Column("page", "TEXT"), Column("source", "TEXT")],
            metadata_json_column="mymeta",
        )
        vs = CloudSQLVectorStore(
            engine,
            embedding_service=FakeEmbeddingsWithDimension(),
            table_name=CUSTOM_TABLE,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=["page", "source"],
            metadata_json_column="mymeta",
        )
        yield vs
        await engine._aexecute(f"DROP TABLE IF EXISTS {CUSTOM_TABLE}")

    async def test_aadd_texts(self, engine, vs):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_texts(texts, ids=ids)
        results = await engine._afetch(f"SELECT * FROM {DEFAULT_TABLE}")
        assert len(results) == 3

        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_texts(texts, metadatas, ids)
        results = await engine._afetch(f"SELECT * FROM {DEFAULT_TABLE}")
        assert len(results) == 6
        await engine._aexecute(f"TRUNCATE TABLE {DEFAULT_TABLE}")

    async def test_aadd_docs(self, engine, vs):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_documents(docs, ids=ids)
        results = await engine._afetch(f"SELECT * FROM {DEFAULT_TABLE}")
        assert len(results) == 3
        await engine._aexecute(f"TRUNCATE TABLE {DEFAULT_TABLE}")

    async def test_aadd_embedding(self, engine, vs):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs._aadd_embeddings(texts, embeddings, metadatas, ids)
        results = await engine._afetch(f"SELECT * FROM {DEFAULT_TABLE}")
        assert len(results) == 3
        await engine._aexecute(f"TRUNCATE TABLE {DEFAULT_TABLE}")

    async def test_add_texts(self, engine, vs):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        vs.add_texts(texts, ids=ids)
        results = await engine._afetch(f"SELECT * FROM {DEFAULT_TABLE}")
        assert len(results) == 3
        await engine._aexecute(f"TRUNCATE TABLE {DEFAULT_TABLE}")
        vs = CloudSQLVectorStore(
            engine,
            embedding_service=FakeEmbeddingsWithDimension(),
            table_name=DEFAULT_TABLE,
            overwrite_existing=True,
        )
        results = await engine._afetch(f"SELECT * FROM {DEFAULT_TABLE}")
        assert len(results) == 0

    async def test_add_docs(self, engine, vs):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        vs.add_documents(docs, ids=ids)
        results = await engine._afetch(f"SELECT * FROM {DEFAULT_TABLE}")
        assert len(results) == 3

    async def test_aadd_texts_custom(self, engine, vs_custom):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs_custom.aadd_texts(texts, ids=ids)
        results = await engine._afetch(f"SELECT * FROM {CUSTOM_TABLE}")
        assert len(results) == 3
        assert results[0]["mycontent"] == "foo"
        assert results[0]["myembedding"]
        assert results[0]["page"] is None
        assert results[0]["source"] is None

        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs_custom.aadd_texts(texts, metadatas, ids)
        results = await engine._afetch(f"SELECT * FROM {CUSTOM_TABLE}")
        assert len(results) == 6
        await engine._aexecute(f"TRUNCATE TABLE {CUSTOM_TABLE}")

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

        results = await engine._afetch(f"SELECT * FROM {CUSTOM_TABLE}")
        assert len(results) == 3
        assert results[0]["mycontent"] == "foo"
        assert results[0]["myembedding"]
        assert results[0]["page"] == "0"
        assert results[0]["source"] == "google.com"
        await engine._aexecute(f"TRUNCATE TABLE {CUSTOM_TABLE}")

    async def test_aadd_embedding_custom(self, engine, vs_custom):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs_custom._aadd_embeddings(texts, embeddings, metadatas, ids)
        results = await engine._afetch(f"SELECT * FROM {CUSTOM_TABLE}")
        assert len(results) == 3
        await engine._aexecute(f"TRUNCATE TABLE {CUSTOM_TABLE}")

    async def test_add_texts_custom(self, engine, vs_custom):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        vs_custom.add_texts(texts, ids=ids)
        results = await engine._afetch(f"SELECT * FROM {CUSTOM_TABLE}")
        assert len(results) == 3
        vs_custom.delete(ids)

    async def test_add_docs_custom(self, engine, vs_custom):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        vs_custom.add_documents(docs, ids=ids)
        results = await engine._afetch(f"SELECT * FROM {CUSTOM_TABLE}")
        assert len(results) == 3

        results = await engine._afetch(f"SELECT * FROM {CUSTOM_TABLE}")
        assert len(results) == 3
        await vs_custom.adelete(ids)

    # Need tests for store metadata=False
