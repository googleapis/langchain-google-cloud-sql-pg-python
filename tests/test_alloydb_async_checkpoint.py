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

import pytest
import pytest_asyncio

from sqlalchemy import text


project_id = os.environ["PROJECT_ID"]
region = os.environ["REGION"]
cluster_id = os.environ["CLUSTER_ID"]
instance_id = os.environ["INSTANCE_ID"]
db_name = os.environ["DATABASE_ID"]
table_name = "message_store" + str(uuid.uuid4())
table_name_async = "message_store" + str(uuid.uuid4())

from langchain_google_alloydb_pg import AlloyDBEngine
from langchain_google_alloydb_pg.async_chat_message_history import (
    AsyncAlloyDBChatMessageHistory,
)

from ..src.langgraph_google_alloydb_pg.async_checkpoint import Engine
from ..src.langgraph_google_alloydb_pg import AsyncAlloyDBPostgresSaver

        
@pytest_asyncio.fixture
async def async_engine():
    async_engine = await Engine.afrom_instance(
        project_id=project_id,
        region=region,
        instance=instance_id,
        database=db_name,
    )
    
    await async_engine.setup()
    await async_engine.close()
    
@pytest.mark.asyncio
async def test_alloydb_checkpoint_async(
    async_engine: AlloyDBEngine
) -> None:
    pass

@pytest.mark.asyncio
async def test_alloydb_checkpoint_sync(
    async_engine: AlloyDBEngine
) -> None:
    pass