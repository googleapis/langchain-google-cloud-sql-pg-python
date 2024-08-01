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
from typing import Any, Generator

import pytest
import pytest_asyncio
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage

from langchain_google_cloud_sql_pg import PostgresChatMessageHistory, PostgresEngine

table_name = "message_store" + str(uuid.uuid4())
table_name_async = "message_store" + str(uuid.uuid4())


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


@pytest.mark.asyncio
class TestEngineSync:
    @pytest.fixture(scope="module")
    def project_id(self) -> str:
        return get_env_var("PROJECT_ID", "project id for google cloud")

    @pytest.fixture(scope="module")
    def region(self) -> str:
        return get_env_var("REGION", "region for cloud sql instance")

    @pytest.fixture(scope="module")
    def instance_id(self) -> str:
        return get_env_var("INSTANCE_ID", "instance for cloud sql")

    @pytest.fixture(scope="module")
    def db_name(self) -> str:
        return get_env_var("DATABASE_ID", "instance for cloud sql")

    @pytest.fixture(scope="module")
    def user(self) -> str:
        return get_env_var("DB_USER", "database user for cloud sql")

    @pytest.fixture(scope="module")
    def password(self) -> str:
        return get_env_var("DB_PASSWORD", "database password for cloud sql")

    @pytest_asyncio.fixture
    def memory_engine(self, project_id, region, instance_id, db_name) -> Generator:
        engine = PostgresEngine.from_instance(
            project_id=project_id,
            region=region,
            instance=instance_id,
            database=db_name,
        )
        engine.init_chat_history_table(table_name=table_name)
        yield engine
        # use default table for PostgresChatMessageHistory
        query = f'DROP TABLE IF EXISTS "{table_name}"'
        engine._execute(query)
        engine._run_as_sync(engine._connector.close_async())
        engine._run_as_sync(engine._engine.dispose())

    def test_chat_message_history(self, memory_engine: PostgresEngine) -> None:
        history = PostgresChatMessageHistory.create_sync(
            engine=memory_engine, session_id="test", table_name=table_name
        )
        history.add_user_message("hi!")
        history.add_ai_message("whats up?")
        messages = history.messages

        # verify messages are correct
        assert messages[0].content == "hi!"
        assert type(messages[0]) is HumanMessage
        assert messages[1].content == "whats up?"
        assert type(messages[1]) is AIMessage

        # verify clear() clears message history
        history.clear()
        assert len(history.messages) == 0

    @pytest.mark.asyncio
    async def test_cross_env_chat_message_history(
        self, memory_engine: PostgresEngine
    ) -> None:
        history = PostgresChatMessageHistory.create_sync(
            engine=memory_engine, session_id="test", table_name=table_name
        )
        await history.aadd_message(HumanMessage(content="hi!"))
        messages = history.messages
        assert messages[0].content == "hi!"
        history.clear()

    def test_chat_table(self, memory_engine: PostgresEngine) -> None:
        with pytest.raises(ValueError):
            PostgresChatMessageHistory.create_sync(
                engine=memory_engine,
                session_id="test",
                table_name="doesnotexist",
            )

    def test_chat_schema(self, memory_engine: PostgresEngine) -> None:
        doc_table_name = "test_table" + str(uuid.uuid4())
        memory_engine.init_document_table(table_name=doc_table_name)
        with pytest.raises(IndexError):
            PostgresChatMessageHistory.create_sync(
                engine=memory_engine,
                session_id="test",
                table_name=doc_table_name,
            )

        memory_engine._execute(f'DROP TABLE IF EXISTS "{doc_table_name}"')

    @pytest.mark.asyncio
    async def test_cross_env_sync_engine(self, memory_engine: PostgresEngine) -> None:
        history = PostgresChatMessageHistory.create_sync(
            engine=memory_engine, session_id="test", table_name=table_name
        )
        await history.aadd_message(HumanMessage(content="hi!"))
        messages = history.messages
        assert messages[0].content == "hi!"
        await history.aclear()

        history.add_message(HumanMessage(content="hi!"))
        messages = history.messages
        assert messages[0].content == "hi!"
        history.clear()


@pytest.mark.asyncio
class TestEngineAsync:
    @pytest.fixture(scope="module")
    def project_id(self) -> str:
        return get_env_var("PROJECT_ID", "project id for google cloud")

    @pytest.fixture(scope="module")
    def region(self) -> str:
        return get_env_var("REGION", "region for cloud sql instance")

    @pytest.fixture(scope="module")
    def instance_id(self) -> str:
        return get_env_var("INSTANCE_ID", "instance for cloud sql")

    @pytest.fixture(scope="module")
    def db_name(self) -> str:
        return get_env_var("DATABASE_ID", "instance for cloud sql")

    @pytest.fixture(scope="module")
    def user(self) -> str:
        return get_env_var("DB_USER", "database user for cloud sql")

    @pytest.fixture(scope="module")
    def password(self) -> str:
        return get_env_var("DB_PASSWORD", "database password for cloud sql")

    @pytest_asyncio.fixture
    async def async_engine(self, project_id, region, instance_id, db_name):
        engine = await PostgresEngine.afrom_instance(
            project_id=project_id,
            region=region,
            instance=instance_id,
            database=db_name,
        )
        await engine.ainit_chat_history_table(table_name=table_name_async)
        yield engine

        await engine._aexecute(f'DROP TABLE IF EXISTS "{table_name_async}"')
        await engine._connector.close_async()
        await engine._engine.dispose()

    @pytest.mark.asyncio
    async def test_cross_env_engine(self, async_engine: PostgresEngine) -> None:
        history = PostgresChatMessageHistory.create_sync(
            engine=async_engine, session_id="test", table_name=table_name_async
        )
        await history.aadd_message(HumanMessage(content="hi!"))
        messages = history.messages
        assert messages[0].content == "hi!"
        await history.aclear()

        history.add_message(HumanMessage(content="hi!"))
        messages = history.messages
        assert messages[0].content == "hi!"
        history.clear()

    @pytest.mark.asyncio
    async def test_chat_message_history_async(
        self,
        async_engine: PostgresEngine,
    ) -> None:
        history = await PostgresChatMessageHistory.create(
            engine=async_engine, session_id="test", table_name=table_name_async
        )
        msg1 = HumanMessage(content="hi!")
        msg2 = AIMessage(content="whats up?")
        await history.aadd_message(msg1)
        await history.aadd_message(msg2)
        messages = history.messages

        # verify messages are correct
        assert messages[0].content == "hi!"
        assert type(messages[0]) is HumanMessage
        assert messages[1].content == "whats up?"
        assert type(messages[1]) is AIMessage

        # verify clear() clears message history
        await history.aclear()
        assert len(history.messages) == 0

    @pytest.mark.asyncio
    async def test_chat_message_history_sync_messages(
        self,
        async_engine: PostgresEngine,
    ) -> None:
        history1 = await PostgresChatMessageHistory.create(
            engine=async_engine, session_id="test", table_name=table_name_async
        )
        history2 = await PostgresChatMessageHistory.create(
            engine=async_engine, session_id="test", table_name=table_name_async
        )
        msg1 = HumanMessage(content="hi!")
        msg2 = AIMessage(content="whats up?")
        await history1.aadd_message(msg1)
        await history2.aadd_message(msg2)

        assert len(history1.messages) == 1
        assert len(history2.messages) == 2

        await history1.async_messages()
        assert len(history1.messages) == 2

        # verify clear() clears message history
        await history2.aclear()
        assert len(history2.messages) == 0

    @pytest.mark.asyncio
    async def test_cross_env(
        self,
        async_engine: PostgresEngine,
    ) -> None:
        history = await PostgresChatMessageHistory.create(
            engine=async_engine, session_id="test", table_name=table_name_async
        )
        history.add_message(HumanMessage(content="hi!"))
        messages = history.messages
        assert messages[0].content == "hi!"
        history.clear()

    @pytest.mark.asyncio
    async def test_chat_table_async(self, async_engine: PostgresEngine) -> None:
        with pytest.raises(ValueError):
            await PostgresChatMessageHistory.create(
                engine=async_engine,
                session_id="test",
                table_name="doesnotexist",
            )

    @pytest.mark.asyncio
    async def test_chat_schema_async(self, async_engine: PostgresEngine) -> None:
        table_name = "test_table" + str(uuid.uuid4())
        await async_engine.ainit_document_table(table_name=table_name)
        with pytest.raises(IndexError):
            await PostgresChatMessageHistory.create(
                engine=async_engine, session_id="test", table_name=table_name
            )

        await async_engine._aexecute(f'DROP TABLE IF EXISTS "{table_name}"')
