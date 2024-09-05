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

from __future__ import annotations

import json
from typing import List, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict

from .engine import PostgresEngine


async def _aget_messages(
    engine: PostgresEngine,
    session_id: str,
    table_name: str,
    schema_name: str = "public",
) -> List[BaseMessage]:
    """Retrieve the messages from PostgreSQL."""
    query = f"""SELECT data, type FROM "{schema_name}"."{table_name}" WHERE session_id = :session_id ORDER BY id;"""
    results = await engine._afetch(query, {"session_id": session_id})
    if not results:
        return []

    items = [{"data": result["data"], "type": result["type"]} for result in results]
    messages = messages_from_dict(items)
    return messages


class PostgresChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in an Cloud SQL for PostgreSQL database."""

    __create_key = object()

    def __init__(
        self,
        key: object,
        engine: PostgresEngine,
        session_id: str,
        table_name: str,
        messages: List[BaseMessage],
        schema_name: str = "public",
    ):
        """PostgresChatMessageHistory constructor.

        Args:
            key (object): Key to prevent direct constructor usage.
            engine (PostgresEngine): Database connection pool.
            session_id (str): Retrieve the table content with this session ID.
            table_name (str): Table name that stores the chat message history.
            messages (List[BaseMessage]): Messages to store.
            schema_name (str, optional): Database schema name of the chat message history table. Defaults to "public".

        Raises:
            Exception: If constructor is directly called by the user.
        """
        if key != PostgresChatMessageHistory.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync' methods!"
            )
        self.engine = engine
        self.session_id = session_id
        self.table_name = table_name
        self.messages = messages
        self.schema_name = schema_name

    @classmethod
    async def create(
        cls,
        engine: PostgresEngine,
        session_id: str,
        table_name: str,
        schema_name: str = "public",
    ) -> PostgresChatMessageHistory:
        """Create a new PostgresChatMessageHistory instance.

        Args:
            engine (PostgresEngine): Postgres engine to use.
            session_id (str): Retrieve the table content with this session ID.
            table_name (str): Table name that stores the chat message history.
            schema_name (str, optional): Schema name for the chat message history table. Defaults to "public".

        Raises:
            IndexError: If the table provided does not contain required schema.

        Returns:
            PostgresChatMessageHistory: A newly created instance of PostgresChatMessageHistory.
        """
        table_schema = await engine._aload_table_schema(table_name, schema_name)
        column_names = table_schema.columns.keys()

        required_columns = ["id", "session_id", "data", "type"]

        if not (all(x in column_names for x in required_columns)):
            raise IndexError(
                f"Table '{schema_name}'.'{table_name}' has incorrect schema. Got "
                f"column names '{column_names}' but required column names "
                f"'{required_columns}'.\nPlease create table with following schema:"
                f"\nCREATE TABLE {schema_name}.{table_name} ("
                "\n    id INT AUTO_INCREMENT PRIMARY KEY,"
                "\n    session_id TEXT NOT NULL,"
                "\n    data JSON NOT NULL,"
                "\n    type TEXT NOT NULL"
                "\n);"
            )
        messages = await _aget_messages(engine, session_id, table_name, schema_name)
        return cls(
            cls.__create_key, engine, session_id, table_name, messages, schema_name
        )

    @classmethod
    def create_sync(
        cls,
        engine: PostgresEngine,
        session_id: str,
        table_name: str,
        schema_name: str = "public",
    ) -> PostgresChatMessageHistory:
        """Create a new PostgresChatMessageHistory instance.

        Args:
            engine (PostgresEngine): Postgres engine to use.
            session_id (str): Retrieve the table content with this session ID.
            table_name (str): Table name that stores the chat message history.
            schema_name (str, optional): Database schema name for the chat message history table. Defaults to "public".

        Raises:
            IndexError: If the table provided does not contain required schema.

        Returns:
            PostgresChatMessageHistory: A newly created instance of PostgresChatMessageHistory.
        """
        coro = cls.create(engine, session_id, table_name, schema_name)
        return engine._run_as_sync(coro)

    async def aadd_message(self, message: BaseMessage) -> None:
        """Append the message to the record in PostgreSQL"""
        query = f"""INSERT INTO "{self.schema_name}"."{self.table_name}"(session_id, data, type)
                    VALUES (:session_id, :data, :type);
                """
        await self.engine._aexecute(
            query,
            {
                "session_id": self.session_id,
                "data": json.dumps(message.dict()),
                "type": message.type,
            },
        )
        self.messages = await _aget_messages(
            self.engine, self.session_id, self.table_name, self.schema_name
        )

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in PostgreSQL"""
        self.engine._run_as_sync(self.aadd_message(message))

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Append a list of messages to the record in PostgreSQL"""
        for message in messages:
            await self.aadd_message(message)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Append a list of messages to the record in PostgreSQL"""
        self.engine._run_as_sync(self.aadd_messages(messages))

    async def aclear(self) -> None:
        """Clear session memory from PostgreSQL"""
        query = f"""DELETE FROM "{self.schema_name}"."{self.table_name}" WHERE session_id = :session_id;"""
        await self.engine._aexecute(query, {"session_id": self.session_id})
        self.messages = []

    def clear(self) -> None:
        """Clear session memory from PostgreSQL"""
        self.engine._run_as_sync(self.aclear())

    async def async_messages(self) -> None:
        """Retrieve the messages from Postgres."""
        self.messages = await _aget_messages(
            self.engine, self.session_id, self.table_name, self.schema_name
        )

    def sync_messages(self) -> None:
        """Retrieve the messages from Postgres."""
        self.engine._run_as_sync(self.async_messages())
