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
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
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
        result_map = result.mappings()
        result_fetch = result_map.fetchall()
    return result_fetch


@pytest.fixture
def test_data():
    """Fixture providing test data for checkpoint tests."""
    config_0: RunnableConfig = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
    config_1: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-1",
            # for backwards compatibility testing
            "thread_ts": "1",
            "checkpoint_ns": "",
        }
    }
    config_2: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-2",
            "checkpoint_id": "2",
            "checkpoint_ns": "",
        }
    }
    config_3: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-2",
            "checkpoint_id": "2-inner",
            "checkpoint_ns": "inner",
        }
    }
    chkpnt_0: Checkpoint = {
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
    chkpnt_1: Checkpoint = empty_checkpoint()
    chkpnt_2: Checkpoint = create_checkpoint(chkpnt_1, {}, 1)
    chkpnt_3: Checkpoint = empty_checkpoint()

    metadata_1: CheckpointMetadata = {
        "source": "input",
        "step": 2,
        "writes": {},
        "parents": 1,
    }
    metadata_2: CheckpointMetadata = {
        "source": "loop",
        "step": 1,
        "writes": {"foo": "bar"},
        "parents": None,
    }
    metadata_3: CheckpointMetadata = {}

    return {
        "configs": [config_0, config_1, config_2, config_3],
        "checkpoints": [chkpnt_0, chkpnt_1, chkpnt_2, chkpnt_3],
        "metadata": [metadata_1, metadata_2, metadata_3],
    }


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
async def test_checkpoint_async(
    async_engine: PostgresEngine, test_data: dict[str, Any]
) -> None:
    """Test inserting and retrieving a checkpoint asynchronously."""

    # Create an instance of AsyncPostgresSaver
    checkpointer = await AsyncPostgresSaver.create(async_engine)

    configs = test_data["configs"]
    checkpoints = test_data["checkpoints"]
    metadata = test_data["metadata"]

    test_config = {
        "configurable": {
            "thread_id": "1",
            "checkpoint_ns": "",
            "checkpoint_id": "1ef4f797-8335-6428-8001-8a1503f9b875",
        }
    }

    # Verify if updated configuration after storing the checkpoint is correct
    next_config = await checkpointer.aput(configs[0], checkpoints[0], {}, {})
    assert dict(next_config) == test_config

    # Verify if the checkpoint is stored correctly in the database
    results = await afetch(async_engine, f"SELECT * FROM {CHECKPOINTS_TABLE}")
    assert len(results) == 1
    for row in results:
        assert isinstance(row["thread_id"], str)
    await aexecute(async_engine, f"TRUNCATE TABLE {CHECKPOINTS_TABLE}")

    writes: Sequence[Tuple[str, Any]] = [("test_channel1", {}), ("test_channel2", {})]

    await checkpointer.aput_writes(configs[0], writes, task_id="1")
    # Verify if the checkpoint writes are stored correctly in the database
    results = await afetch(async_engine, f"SELECT * FROM {CHECKPOINT_WRITES_TABLE}")
    assert len(results) == 2
    for row in results:
        assert isinstance(row["task_id"], str)
    await aexecute(async_engine, f"TRUNCATE TABLE {CHECKPOINT_WRITES_TABLE}")

    await checkpointer.aput(configs[1], checkpoints[1], metadata[0], {})
    await checkpointer.aput(configs[2], checkpoints[2], metadata[1], {})
    await checkpointer.aput(configs[3], checkpoints[3], metadata[2], {})

    # call method / assertions
    query_1 = {"source": "input"}  # search by 1 key
    query_2 = {
        "step": 1,
        "writes": {"foo": "bar"},
    }  # search by multiple keys
    query_3: dict[str, Any] = {}  # search by no keys, return all checkpoints
    query_4 = {"source": "update", "step": 1}  # no match

    search_results_1 = [c async for c in checkpointer.alist(None, filter=query_1)]
    assert len(search_results_1) == 1
    assert search_results_1[0].metadata == metadata[0]

    search_results_2 = [c async for c in checkpointer.alist(None, filter=query_2)]
    assert len(search_results_2) == 1
    assert search_results_2[0].metadata == metadata[1]

    search_results_3 = [c async for c in checkpointer.alist(None, filter=query_3)]
    assert len(search_results_3) == 3

    search_results_4 = [c async for c in checkpointer.alist(None, filter=query_4)]
    assert len(search_results_4) == 0

    # search by config (defaults to checkpoints across all namespaces)
    search_results_5 = [
        c async for c in checkpointer.alist({"configurable": {"thread_id": "thread-2"}})
    ]
    assert len(search_results_5) == 2
    assert {
        search_results_5[0].config["configurable"]["checkpoint_ns"],
        search_results_5[1].config["configurable"]["checkpoint_ns"],
    } == {"", "inner"}


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


@pytest.mark.asyncio
async def test_null_chars(
    async_engine: PostgresEngine,
    test_data: dict[str, Any],
) -> None:
    checkpointer = await AsyncPostgresSaver.create(async_engine)
    config = await checkpointer.aput(
        test_data["configs"][0],
        test_data["checkpoints"][0],
        {"my_key": "\x00abc"},  # type: ignore
        {},
    )
    # assert (await checkpointer.aget_tuple(config)).metadata["my_key"] == "abc"  # type: ignore
    assert [c async for c in checkpointer.alist(None, filter={"my_key": "abc"})][
        0
    ].metadata[
        "my_key"
    ] == "abc"  # type: ignore
