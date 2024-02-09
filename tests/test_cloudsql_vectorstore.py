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
from sqlalchemy import TEXT, VARCHAR, Column

from langchain_google_cloud_sql_pg import CloudSQLVectorStore, PostgreSQLEngine

PROJECT_ID = os.environ.get("PROJECT_ID")
INSTANCE = os.environ.get("INSTANCE_ID")
DATABASE = os.environ.get("DATABASE_ID")
REGION = os.environ.get("REGION")
DEFAULT_TABLE = "test_table_vs"
CUSTOM_TABLE = "test_table_custom_vs"
VECTOR_SIZE = 768


class FakeEmbeddingsDimension(FakeEmbeddings):
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


@pytest.mark.asyncio
class TestVectorStore:

    @pytest_asyncio.fixture  # (scope="function")
    async def engine(self):
        assert PROJECT_ID
        assert INSTANCE
        assert DATABASE
        assert REGION
        engine = await PostgreSQLEngine.afrom_instance(
            project_id=PROJECT_ID,
            instance=INSTANCE,
            region=REGION,
            database=DATABASE,
        )
        # await engine.ainit_vectorstore_table(DEFAULT_TABLE, VECTOR_SIZE)
        yield engine
        # await engine.aexecute(f"DROP TABLE {DEFAULT_TABLE}")

    async def test_init(self, engine):
        vs = CloudSQLVectorStore(
            engine,
            embedding_service=FakeEmbeddingsDimension(),
            table_name=DEFAULT_TABLE,
        )
        assert vs
