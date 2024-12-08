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
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from sqlalchemy import text

from langchain_google_cloud_sql_pg import PostgresEngine
from langchain_google_cloud_sql_pg.async_chat_message_history import (
    AsyncPostgresChatMessageHistory,
)

project_id = os.environ["PROJECT_ID"]
region = os.environ["REGION"]
instance_id = os.environ["INSTANCE_ID"]
db_name = os.environ["DATABASE_ID"]
table_name = "message_store" + str(uuid.uuid4())
table_name_async = "message_store" + str(uuid.uuid4())


async def aexecute(engine: PostgresEngine, query: str) -> None:
    async with engine._pool.connect() as conn:
        await conn.execute(text(query))
        await conn.commit()


@pytest_asyncio.fixture
async def async_engine():
    async_engine = await PostgresEngine.afrom_instance(
        project_id=project_id,
        region=region,
        instance=instance_id,
        database=db_name,
    )
    await async_engine._ainit_chat_history_table(table_name=table_name_async)
    yield async_engine
    # use default table for AsyncPostgresChatMessageHistory
    query = f'DROP TABLE IF EXISTS "{table_name_async}"'
    await aexecute(async_engine, query)
    await async_engine.close()


@pytest.mark.asyncio
async def test_chat_message_history_async(
    async_engine: PostgresEngine,
) -> None:
    history = await AsyncPostgresChatMessageHistory.create(
        engine=async_engine, session_id="test", table_name=table_name_async
    )
    msg1 = HumanMessage(content="hi!")
    msg2 = AIMessage(content="whats up?")
    await history.aadd_message(msg1)
    await history.aadd_message(msg2)
    messages = await history._aget_messages()

    # verify messages are correct
    assert messages[0].content == "hi!"
    assert type(messages[0]) is HumanMessage
    assert messages[1].content == "whats up?"
    assert type(messages[1]) is AIMessage

    # verify clear() clears message history
    await history.aclear()
    assert len(await history._aget_messages()) == 0


@pytest.mark.asyncio
async def test_chat_message_history_sync_messages(
    async_engine: PostgresEngine,
) -> None:
    history1 = await AsyncPostgresChatMessageHistory.create(
        engine=async_engine, session_id="test", table_name=table_name_async
    )
    history2 = await AsyncPostgresChatMessageHistory.create(
        engine=async_engine, session_id="test", table_name=table_name_async
    )
    msg1 = HumanMessage(content="hi!")
    msg2 = AIMessage(content="whats up?")
    await history1.aadd_message(msg1)
    await history2.aadd_message(msg2)

    assert len(await history1._aget_messages()) == 2
    assert len(await history2._aget_messages()) == 2

    # verify clear() clears message history
    await history2.aclear()
    assert len(await history2._aget_messages()) == 0


@pytest.mark.asyncio
async def test_chat_table_async(async_engine):
    with pytest.raises(ValueError):
        await AsyncPostgresChatMessageHistory.create(
            engine=async_engine, session_id="test", table_name="doesnotexist"
        )


@pytest.mark.asyncio
async def test_chat_schema_async(async_engine):
    table_name = "test_table" + str(uuid.uuid4())
    await async_engine._ainit_document_table(table_name=table_name)
    with pytest.raises(IndexError):
        await AsyncPostgresChatMessageHistory.create(
            engine=async_engine, session_id="test", table_name=table_name
        )

    query = f'DROP TABLE IF EXISTS "{table_name}"'
    await aexecute(async_engine, query)
