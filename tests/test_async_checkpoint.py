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
from typing import Sequence, Any, Tuple

import pytest
import pytest_asyncio
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
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

# Configurations for writing and reading checkpoints
write_config: RunnableConfig = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
read_config: RunnableConfig = {"configurable": {"thread_id": "1"}}

# Environment variables for PostgreSQL instance
project_id = os.environ["PROJECT_ID"]
region = os.environ["REGION"]
instance_id = os.environ["INSTANCE_ID"]
db_name = os.environ["DATABASE_ID"]

# Sample checkpoint data
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
    """Execute an asynchronous SQL command."""
    async with engine._pool.connect() as conn:
        await conn.execute(text(query))
        await conn.commit()


async def afetch(engine: PostgresEngine, query: str) -> Sequence[RowMapping]:
    """Fetch data from the database asynchronously."""
    async with engine._pool.connect() as conn:
        result = await conn.execute(text(query))
        return result.mappings().fetchall()


@pytest_asyncio.fixture
async def async_engine():
    """Setup and teardown for PostgresEngine instance."""
    async_engine = await PostgresEngine.afrom_instance(
        project_id=project_id,
        region=region,
        instance=instance_id,
        database=db_name,
    )
    await async_engine._ainit_checkpoint_table()

    yield async_engine  # Provide the engine instance for testing

    # Cleanup: Drop checkpoint tables after tests
    await aexecute(async_engine, f'DROP TABLE IF EXISTS "{CHECKPOINTS_TABLE}"')
    await aexecute(async_engine, f'DROP TABLE IF EXISTS "{CHECKPOINT_WRITES_TABLE}"')
    await async_engine.close()


@pytest.mark.asyncio
async def test_checkpoint_async(async_engine: PostgresEngine) -> None:
    """Test inserting and retrieving a checkpoint asynchronously."""

    # Create an instance of AsyncPostgresSaver
    checkpointer = await AsyncPostgresSaver.create(async_engine)

    # Expected configuration after storing the checkpoint
    expected_config = {
        "configurable": {
            "thread_id": "1",
            "checkpoint_ns": "",
            "checkpoint_id": "1ef4f797-8335-6428-8001-8a1503f9b875",
        }
    }

    # Insert checkpoint and verify returned configuration
    next_config = await checkpointer.aput(write_config, checkpoint, {}, {})
    assert dict(next_config) == expected_config

    # Verify if the checkpoint is stored correctly in the database
    results = await afetch(async_engine, f'SELECT * FROM "{CHECKPOINTS_TABLE}"')
    assert len(results) == 1  # Only one checkpoint should be stored

    for row in results:
        assert isinstance(row["thread_id"], str)

    # Cleanup: Remove all checkpoints after the test
    await aexecute(async_engine, f'TRUNCATE TABLE "{CHECKPOINTS_TABLE}"')


async def test_checkpoint_aput_writes(
    async_engine: PostgresEngine,
) -> None:
    checkpointer = await AsyncPostgresSaver.create(async_engine)

    config: RunnableConfig = {
        "configurable": {
            "thread_id": "1",
            "checkpoint_ns": "",
            "checkpoint_id": "1ef4f797-8335-6428-8001-8a1503f9b875",
        }
    }

    # Verify if the checkpoint writes are stored correctly in the database
    writes: Sequence[Tuple[str, Any]] = [("test_channel1", {}), ("test_channel2", {})]
    await checkpointer.aput_writes(config, writes, task_id="1")

    results = await afetch(async_engine, f"SELECT * FROM {CHECKPOINT_WRITES_TABLE}")
    assert len(results) == 2
    for row in results:
        assert isinstance(row["task_id"], str)
    await aexecute(async_engine, f"TRUNCATE TABLE {CHECKPOINT_WRITES_TABLE}")
