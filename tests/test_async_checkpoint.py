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
import re
import uuid
from typing import Any, List, Literal, Optional, Sequence, Tuple, Union

import pytest
import pytest_asyncio
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.prebuilt import (  # type: ignore[import-not-found]
    ToolNode,
    ValidationNode,
    create_react_agent,
    tools_condition,
)
from sqlalchemy import text
from sqlalchemy.engine.row import RowMapping

from langchain_google_cloud_sql_pg.async_checkpoint import AsyncPostgresSaver
from langchain_google_cloud_sql_pg.engine import PostgresEngine

write_config: RunnableConfig = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
read_config: RunnableConfig = {"configurable": {"thread_id": "1"}}
thread_agent_config: RunnableConfig = {"configurable": {"thread_id": "123"}}

project_id = os.environ["PROJECT_ID"]
region = os.environ["REGION"]
instance_id = os.environ["INSTANCE_ID"]
db_name = os.environ["DATABASE_ID"]
table_name = "checkpoint" + str(uuid.uuid4())
table_name_writes = table_name + "_writes"

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


class AnyStr(str):
    def __init__(self, prefix: Union[str, re.Pattern] = "") -> None:
        super().__init__()
        self.prefix = prefix

    def __eq__(self, other: object) -> bool:
        return isinstance(other, str) and (
            (
                other.startswith(self.prefix)
                if isinstance(self.prefix, str)
                else bool(self.prefix.match(other))
            )
        )

    def __hash__(self) -> int:
        return hash((str(self), self.prefix))


def _AnyIdToolMessage(**kwargs: Any) -> ToolMessage:
    """Create a tool message with an any id field."""
    message = ToolMessage(**kwargs)
    message.id = AnyStr()
    return message


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
        instance=instance_id,
        database=db_name,
    )

    yield async_engine

    await aexecute(async_engine, f'DROP TABLE IF EXISTS "{table_name}"')
    await aexecute(async_engine, f'DROP TABLE IF EXISTS "{table_name_writes}"')
    await async_engine.close()


@pytest_asyncio.fixture
async def checkpointer(async_engine):
    await async_engine._ainit_checkpoint_table(table_name=table_name)
    checkpointer = await AsyncPostgresSaver.create(
        async_engine,
        table_name,  # serde=JsonPlusSerializer
    )
    yield checkpointer


@pytest.mark.asyncio
async def test_checkpoint_async(
    async_engine: PostgresEngine,
    checkpointer: AsyncPostgresSaver,
) -> None:
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
    results = await afetch(async_engine, f'SELECT * FROM "{table_name}"')
    assert len(results) == 1
    for row in results:
        assert isinstance(row["thread_id"], str)
    await aexecute(async_engine, f'TRUNCATE TABLE "{table_name}"')


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
async def test_checkpoint_aput_writes(
    async_engine: PostgresEngine,
    checkpointer: AsyncPostgresSaver,
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
    await checkpointer.aput_writes(config, writes, task_id="1")

    results = await afetch(async_engine, f'SELECT * FROM "{table_name_writes}"')
    assert len(results) == 2
    for row in results:
        assert isinstance(row["task_id"], str)
    await aexecute(async_engine, f'TRUNCATE TABLE "{table_name_writes}"')


@pytest.mark.asyncio
async def test_checkpoint_alist(
    async_engine: PostgresEngine,
    checkpointer: AsyncPostgresSaver,
    test_data: dict[str, Any],
) -> None:
    configs = test_data["configs"]
    checkpoints = test_data["checkpoints"]
    metadata = test_data["metadata"]

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
    print(metadata[0])
    print(search_results_1[0].metadata)
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


class FakeToolCallingModel(BaseChatModel):
    tool_calls: Optional[list[list[ToolCall]]] = None
    index: int = 0
    tool_style: Literal["openai", "anthropic"] = "openai"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call"""
        messages_string = "-".join(
            [str(m.content) for m in messages if isinstance(m.content, str)]
        )
        tool_calls = (
            self.tool_calls[self.index % len(self.tool_calls)]
            if self.tool_calls
            else []
        )
        message = AIMessage(
            content=messages_string,
            id=str(self.index),
            tool_calls=tool_calls.copy(),
        )
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "fake-tool-call-model"


@pytest.mark.asyncio
async def test_checkpoint_with_agent(
    checkpointer: AsyncPostgresSaver,
) -> None:
    # from the tests in https://github.com/langchain-ai/langgraph/blob/909190cede6a80bb94a2d4cfe7dedc49ef0d4127/libs/langgraph/tests/test_prebuilt.py
    model = FakeToolCallingModel()

    agent = create_react_agent(model, [], checkpointer=checkpointer)
    inputs = [HumanMessage("hi?")]
    response = await agent.ainvoke(
        {"messages": inputs}, config=thread_agent_config, debug=True
    )
    expected_response = {"messages": inputs + [AIMessage(content="hi?", id="0")]}
    assert response == expected_response

    def _AnyIdHumanMessage(**kwargs: Any) -> HumanMessage:
        """Create a human message with an any id field."""
        message = HumanMessage(**kwargs)
        message.id = AnyStr()
        return message

    saved = await checkpointer.aget_tuple(thread_agent_config)
    assert saved is not None
    assert (
        _AnyIdHumanMessage(content="hi?")
        in saved.checkpoint["channel_values"]["messages"]
    )
    assert (
        AIMessage(content="hi?", id="0")
        in saved.checkpoint["channel_values"]["messages"]
    )
    assert saved.metadata == {
        "parents": {},
        "source": "loop",
        "step": 1,
    }
    assert saved.pending_writes == []


@pytest.mark.asyncio
async def test_checkpoint_aget_tuple(
    checkpointer: AsyncPostgresSaver,
    test_data: dict[str, Any],
) -> None:
    configs = test_data["configs"]
    checkpoints = test_data["checkpoints"]
    metadata = test_data["metadata"]

    new_config = await checkpointer.aput(configs[1], checkpoints[1], metadata[0], {})

    # Matching checkpoint
    search_results_1 = await checkpointer.aget_tuple(new_config)
    assert search_results_1.metadata == metadata[0]  # type: ignore

    # No matching checkpoint
    assert await checkpointer.aget_tuple(configs[0]) is None


@pytest.mark.asyncio
async def test_metadata(
    checkpointer: AsyncPostgresSaver,
    test_data: dict[str, Any],
) -> None:
    config = await checkpointer.aput(
        test_data["configs"][0],
        test_data["checkpoints"][0],
        {"my_key": "abc"},  # type: ignore
        {},
    )
    assert (await checkpointer.aget_tuple(config)).metadata["my_key"] == "abc"  # type: ignore
    assert [c async for c in checkpointer.alist(None, filter={"my_key": "abc"})][
        0
    ].metadata[
        "my_key"  # type: ignore
    ] == "abc"  # type: ignore
