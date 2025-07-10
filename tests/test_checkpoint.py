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
from typing import Any, Sequence, Tuple

import pytest
import pytest_asyncio
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from sqlalchemy import text
from sqlalchemy.engine.row import RowMapping

from langchain_google_cloud_sql_pg.checkpoint import PostgresSaver
from langchain_google_cloud_sql_pg.engine import PostgresEngine

write_config: RunnableConfig = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
read_config: RunnableConfig = {"configurable": {"thread_id": "1"}}

project_id = os.environ["PROJECT_ID"]
region = os.environ["REGION"]
instance_id = os.environ["INSTANCE_ID"]
db_name = os.environ["DATABASE_ID"]
table_name = "checkpoint" + str(uuid.uuid4())
table_name_writes = table_name + "_writes"
table_name_async = "checkpoint" + str(uuid.uuid4())
table_name_writes_async = table_name_async + "_writes"

checkpoint: Checkpoint = {
    "v": 1,
    "ts": "2024-07-31T20:14:19.804150+00:00",
    "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
    "channel_values": {"my_key": "meow", "node": "node"},
    "channel_versions": {
        "__start__": 2,
        "my_key": 3,
        "start:node": 3,
        "node": 3,
    },
    "versions_seen": {
        "__input__": {},
        "__start__": {"__start__": 1},
        "node": {"start:node": 2},
    },
}


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


async def aexecute(engine: PostgresEngine, query: str) -> None:
    async def run(engine, query):
        async with engine._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    await engine._run_as_async(run(engine, query))


async def afetch(engine: PostgresEngine, query: str) -> Sequence[RowMapping]:
    async def run(engine, query):
        async with engine._pool.connect() as conn:
            result = await conn.execute(text(query))
            result_map = result.mappings()
            result_fetch = result_map.fetchall()
        return result_fetch

    return await engine._run_as_async(run(engine, query))


@pytest_asyncio.fixture
async def engine():
    engine = PostgresEngine.from_instance(
        project_id=project_id,
        region=region,
        instance=instance_id,
        database=db_name,
    )
    yield engine
    # use default table
    await aexecute(engine, f'DROP TABLE IF EXISTS "{table_name}"')
    await aexecute(engine, f'DROP TABLE IF EXISTS "{table_name_writes}"')
    await engine.close()


@pytest_asyncio.fixture
async def async_engine():
    async_engine = await PostgresEngine.afrom_instance(
        project_id=project_id,
        region=region,
        instance=instance_id,
        database=db_name,
    )
    yield async_engine

    await aexecute(async_engine, f'DROP TABLE IF EXISTS "{table_name_async}"')
    await aexecute(async_engine, f'DROP TABLE IF EXISTS "{table_name_writes_async}"')
    await async_engine.close()


@pytest_asyncio.fixture
def checkpointer(engine):
    engine.init_checkpoint_table(table_name=table_name)
    checkpointer = PostgresSaver.create_sync(engine, table_name)
    yield checkpointer


@pytest_asyncio.fixture
async def async_checkpointer(async_engine):
    await async_engine.ainit_checkpoint_table(table_name=table_name_async)
    async_checkpointer = await PostgresSaver.create(async_engine, table_name_async)
    yield async_checkpointer


@pytest.mark.asyncio
async def test_checkpoint_async(
    async_engine: PostgresEngine,
    async_checkpointer: PostgresSaver,
) -> None:
    test_config = {
        "configurable": {
            "thread_id": "1",
            "checkpoint_ns": "",
            "checkpoint_id": "1ef4f797-8335-6428-8001-8a1503f9b875",
        }
    }
    # Verify if updated configuration after storing the checkpoint is correct
    next_config = await async_checkpointer.aput(write_config, checkpoint, {}, {})
    assert dict(next_config) == test_config

    # Verify if the checkpoint is stored correctly in the database
    results = await afetch(async_engine, f'SELECT * FROM "{table_name_async}"')
    assert len(results) == 1
    for row in results:
        assert isinstance(row["thread_id"], str)
    await aexecute(async_engine, f'TRUNCATE TABLE "{table_name_async}"')


# Test put method for checkpoint
@pytest.mark.asyncio
async def test_checkpoint_sync(
    engine: PostgresEngine,
    checkpointer: PostgresSaver,
) -> None:
    test_config = {
        "configurable": {
            "thread_id": "1",
            "checkpoint_ns": "",
            "checkpoint_id": "1ef4f797-8335-6428-8001-8a1503f9b875",
        }
    }
    # Verify if updated configuration after storing the checkpoint is correct
    next_config = checkpointer.put(write_config, checkpoint, {}, {})
    assert dict(next_config) == test_config

    # Verify if the checkpoint is stored correctly in the database
    results = await afetch(engine, f'SELECT * FROM "{table_name}"')
    assert len(results) == 1
    for row in results:
        assert isinstance(row["thread_id"], str)
    await aexecute(engine, f'TRUNCATE TABLE "{table_name}"')


@pytest.mark.asyncio
async def test_chat_table_async(async_engine):
    with pytest.raises(ValueError):
        await PostgresSaver.create(engine=async_engine, table_name="doesnotexist")


def test_checkpoint_table(engine: Any) -> None:
    with pytest.raises(ValueError):
        PostgresSaver.create_sync(engine=engine, table_name="doesnotexist")


@pytest.fixture
def test_data() -> dict[str, Any]:
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
        "channel_versions": {
            "__start__": 2,
            "my_key": 3,
            "start:node": 3,
            "node": 3,
        },
        "versions_seen": {
            "__input__": {},
            "__start__": {"__start__": 1},
            "node": {"start:node": 2},
        },
    }
    chkpnt_1: Checkpoint = empty_checkpoint()
    chkpnt_2: Checkpoint = create_checkpoint(chkpnt_1, {}, 1)
    chkpnt_3: Checkpoint = empty_checkpoint()

    metadata_1: CheckpointMetadata = {
        "source": "input",
        "step": 2,
        "writes": {},
        "parents": 1,  # type: ignore[typeddict-item]
    }
    metadata_2: CheckpointMetadata = {
        "source": "loop",
        "step": 1,
        "writes": {"foo": "bar"},
        "parents": None,  # type: ignore[typeddict-item]
    }
    metadata_3: CheckpointMetadata = {}

    return {
        "configs": [config_0, config_1, config_2, config_3],
        "checkpoints": [chkpnt_0, chkpnt_1, chkpnt_2, chkpnt_3],
        "metadata": [metadata_1, metadata_2, metadata_3],
    }


@pytest.mark.asyncio
async def test_checkpoint_put_writes(
    engine: PostgresEngine,
    checkpointer: PostgresSaver,
) -> None:
    config: RunnableConfig = {
        "configurable": {
            "thread_id": "1",
            "checkpoint_ns": "",
            "checkpoint_id": "1ef4f797-8335-6428-8001-8a1503f9b875",
        }
    }

    # Verify if the checkpoint writes are stored correctly in the database
    writes: Sequence[Tuple[str, Any]] = [
        ("test_channel1", {}),
        ("test_channel2", {}),
    ]
    checkpointer.put_writes(config, writes, task_id="1")

    results = await afetch(engine, f'SELECT * FROM "{table_name_writes}"')
    assert len(results) == 2
    for row in results:
        assert isinstance(row["task_id"], str)
    await aexecute(engine, f'TRUNCATE TABLE "{table_name_writes}"')


def test_checkpoint_list(
    checkpointer: PostgresSaver,
    test_data: dict[str, Any],
) -> None:
    configs = test_data["configs"]
    checkpoints = test_data["checkpoints"]
    metadata = test_data["metadata"]

    checkpointer.put(configs[1], checkpoints[1], metadata[0], {})
    checkpointer.put(configs[2], checkpoints[2], metadata[1], {})
    checkpointer.put(configs[3], checkpoints[3], metadata[2], {})

    # call method / assertions
    query_1 = {"source": "input"}  # search by 1 key
    query_2 = {
        "step": 1,
        "writes": {"foo": "bar"},
    }  # search by multiple keys
    query_3: dict[str, Any] = {}  # search by no keys, return all checkpoints
    query_4 = {"source": "update", "step": 1}  # no match

    search_results_1 = list(checkpointer.list(None, filter=query_1))
    assert len(search_results_1) == 1
    assert search_results_1[0].metadata == metadata[0]
    search_results_2 = list(checkpointer.list(None, filter=query_2))
    assert len(search_results_2) == 1
    assert search_results_2[0].metadata == metadata[1]

    search_results_3 = list(checkpointer.list(None, filter=query_3))
    assert len(search_results_3) == 3

    search_results_4 = list(checkpointer.list(None, filter=query_4))
    assert len(search_results_4) == 0

    # search by config (defaults to checkpoints across all namespaces)
    search_results_5 = list(
        checkpointer.list({"configurable": {"thread_id": "thread-2"}})
    )
    assert len(search_results_5) == 2
    assert {
        search_results_5[0].config["configurable"]["checkpoint_ns"],
        search_results_5[1].config["configurable"]["checkpoint_ns"],
    } == {"", "inner"}


def test_checkpoint_get_tuple(
    checkpointer: PostgresSaver,
    test_data: dict[str, Any],
) -> None:
    configs = test_data["configs"]
    checkpoints = test_data["checkpoints"]
    metadata = test_data["metadata"]

    new_config = checkpointer.put(configs[1], checkpoints[1], metadata[0], {})

    # Matching checkpoint
    search_results_1 = checkpointer.get_tuple(new_config)
    assert search_results_1.metadata == metadata[0]  # type: ignore

    # No matching checkpoint
    assert checkpointer.get_tuple(configs[0]) is None
