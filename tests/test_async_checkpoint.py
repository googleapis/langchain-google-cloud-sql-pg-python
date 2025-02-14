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
from typing import Sequence

import pytest
import pytest_asyncio
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
)
from sqlalchemy import text
from sqlalchemy.engine.row import RowMapping

from langchain_google_cloud_sql_pg.async_checkpoint import AsyncPostgresSaver
from langchain_google_cloud_sql_pg.engine import (
    CHECKPOINT_WRITES_TABLE,
    CHECKPOINTS_TABLE,
    PostgresEngine,
)

write_config: RunnableConfig = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
read_config: RunnableConfig = {"configurable": {"thread_id": "1"}}

project_id = os.environ["PROJECT_ID"]
region = os.environ["REGION"]
cluster_id = os.environ["CLUSTER_ID"]
instance_id = os.environ["INSTANCE_ID"]
db_name = os.environ["DATABASE_ID"]

checkpoint: Checkpoint = {
    "v": 1,
    "ts": "2024-07-31T20:14:19.804150+00:00",
    "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
    "channel_values": {"my_key": "meow", "node": "node"},
    "channel_versions": {"__start__": 2, "my_key": 3, "start:node": 3, "node": 3},
    "versions_seen": {
        "__input__": {},
        "__start__": {"__start__": 1},
        "node": {"start:node": 2},
    },
    "pending_sends": [],
}


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


@pytest_asyncio.fixture
async def async_engine():
    async_engine = await PostgresEngine.afrom_instance(
        project_id=project_id,
        region=region,
        cluster=cluster_id,
        instance=instance_id,
        database=db_name,
    )
    await async_engine._ainit_checkpoint_table()
    yield async_engine
    # use default table for AsyncPostgresSaver
    await aexecute(async_engine, f'DROP TABLE IF EXISTS "{CHECKPOINTS_TABLE}"')
    await aexecute(async_engine, f'DROP TABLE IF EXISTS"{CHECKPOINT_WRITES_TABLE}"')
    await async_engine.close()


@pytest.mark.asyncio
async def test_checkpoint_async(
    async_engine: PostgresEngine,
) -> None:
    checkpointer = await AsyncPostgresSaver.create(async_engine)
    test_config = {
        "configurable": {
            "thread_id": "1",
            "checkpoint_ns": "",
            "checkpoint_id": "1ef4f797-8335-6428-8001-8a1503f9b875",
        }
    }
    # Verify if updated configuration after storing the checkpoint is correct
    next_config = await checkpointer.aput(write_config, checkpoint, {}, {})
    assert dict(next_config) == test_config

    # Verify if the checkpoint is stored correctly in the database
    results = await afetch(async_engine, f"SELECT * FROM {CHECKPOINTS_TABLE}")
    assert len(results) == 1
    for row in results:
        assert isinstance(row["thread_id"], str)
    await aexecute(async_engine, f"TRUNCATE TABLE {CHECKPOINTS_TABLE}")
