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

from langchain_google_cloud_sql_pg import PostgreSQLEngine

PROJECT_ID = os.environ.get("PROJECT_ID")
INSTANCE = os.environ.get("INSTANCE_ID")
DATABASE = os.environ.get("DATABASE_ID")
REGION = os.environ.get("REGION")
DEFAULT_TABLE = "test_table"
CUSTOM_TABLE = "test_table_custom"
VECTOR_SIZE = 768

embeddings_service = VertexAIEmbeddings()


@pytest.mark.asyncio
class TestEngineAsync:
    @pytest_asyncio.fixture
    async def engine(self) -> None:
        engine = await PostgreSQLEngine.afrom_instance(
            project_id=PROJECT_ID,
            instance=INSTANCE,
            region=REGION,
            database=DATABASE,
        )
        yield engine

    async def test_execute(self, engine):
        await engine.aexecute("SELECT 1")

    async def test_init_table(self, engine):
        await engine.ainit_vectorstore_table(DEFAULT_TABLE, VECTOR_SIZE)
        id = str(uuid.uuid4())
        content = "coffee"
        embedding = await embeddings_service.aembed_query(content)
        stmt = f"INSERT INTO {DEFAULT_TABLE} (langchain_id, content, embedding) VALUES ('{id}', '{content}','{embedding}');"
        await engine.aexecute(stmt)

    async def test_fetch(self, engine):
        results = await engine.afetch(f"SELECT * FROM {DEFAULT_TABLE}")
        assert len(results) > 0
        await engine.aexecute(f"DROP TABLE {DEFAULT_TABLE}")

    async def test_init_table_custom(self, engine):
        await engine.ainit_vectorstore_table(
            CUSTOM_TABLE,
            VECTOR_SIZE,
            id_column="uuid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=[Column("page", TEXT), Column("source", TEXT)],
            store_metadata=True,
        )
        id = str(uuid.uuid4())
        content = "coffee"
        embedding = await embeddings_service.aembed_query(content)
        meta = json.dumps({"field1": "value1"})
        stmt = f"""
        INSERT INTO {CUSTOM_TABLE} (uuid, mycontent, myembedding, page, source, langchain_metadata)
        VALUES ('{id}', '{content}','{embedding}', '1', 'google.com', '{meta}');
        """
        await engine.aexecute(stmt)
        results = await engine.afetch(f"SELECT * FROM {CUSTOM_TABLE}")
        assert len(results) > 0
        await engine.aexecute(f"DROP TABLE {CUSTOM_TABLE}")


class TestEngineSync:
    @pytest_asyncio.fixture
    def engine(self) -> None:
        engine = PostgreSQLEngine.from_instance(
            project_id=PROJECT_ID,
            instance=INSTANCE,
            region=REGION,
            database=DATABASE,
        )
        yield engine
