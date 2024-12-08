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
from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from .engine import PostgresEngine


class AsyncPostgresChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in an Cloud SQL for PostgreSQL database."""

    __create_key = object()

    def __init__(
        self,
        key: object,
        pool: AsyncEngine,
        session_id: str,
        table_name: str,
        schema_name: str = "public",
    ):
        """AsyncPostgresChatMessageHistory constructor.

        Args:
            key (object): Key to prevent direct constructor usage.
            engine (PostgresEngine): Database connection pool.
            session_id (str): Retrieve the table content with this session ID.
            table_name (str): Table name that stores the chat message history.
            schema_name (str, optional): Database schema name of the chat message history table. Defaults to "public".

        Raises:
            Exception: If constructor is directly called by the user.
        """
        if key != AsyncPostgresChatMessageHistory.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync' methods!"
            )
        self.pool = pool
        self.session_id = session_id
        self.table_name = table_name
        self.schema_name = schema_name

    @classmethod
    async def create(
        cls,
        engine: PostgresEngine,
        session_id: str,
        table_name: str,
        schema_name: str = "public",
    ) -> AsyncPostgresChatMessageHistory:
        """Create a new AsyncPostgresChatMessageHistory instance.

        Args:
            engine (PostgresEngine): Postgres engine to use.
            session_id (str): Retrieve the table content with this session ID.
            table_name (str): Table name that stores the chat message history.
            schema_name (str, optional): Database schema name for the chat message history table. Defaults to "public".

        Raises:
            IndexError: If the table provided does not contain required schema.

        Returns:
            AsyncPostgresChatMessageHistory: A newly created instance of AsyncPostgresChatMessageHistory.
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
        return cls(cls.__create_key, engine._pool, session_id, table_name)

    async def aadd_message(self, message: BaseMessage) -> None:
        """Append the message to the record in PostgreSQL"""
        query = f"""INSERT INTO "{self.schema_name}"."{self.table_name}"(session_id, data, type)
                    VALUES (:session_id, :data, :type);
                """
        async with self.pool.connect() as conn:
            await conn.execute(
                text(query),
                {
                    "session_id": self.session_id,
                    "data": json.dumps(message.dict()),
                    "type": message.type,
                },
            )
            await conn.commit()

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Append a list of messages to the record in PostgreSQL"""
        for message in messages:
            await self.aadd_message(message)

    async def aclear(self) -> None:
        """Clear session memory from PostgreSQL"""
        query = f"""DELETE FROM "{self.schema_name}"."{self.table_name}" WHERE session_id = :session_id;"""
        async with self.pool.connect() as conn:
            await conn.execute(text(query), {"session_id": self.session_id})
            await conn.commit()

    async def _aget_messages(self) -> list[BaseMessage]:
        """Retrieve the messages from PostgreSQL."""
        query = f"""SELECT data, type FROM "{self.schema_name}"."{self.table_name}" WHERE session_id = :session_id ORDER BY id;"""
        async with self.pool.connect() as conn:
            result = await conn.execute(text(query), {"session_id": self.session_id})
            result_map = result.mappings()
            results = result_map.fetchall()
        if not results:
            return []

        items = [{"data": result["data"], "type": result["type"]} for result in results]
        messages = messages_from_dict(items)
        return messages

    def clear(self) -> None:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresChatMessageHistory. Use PostgresChatMessageHistory interface instead."
        )
