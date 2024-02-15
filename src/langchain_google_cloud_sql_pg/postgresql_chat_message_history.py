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
from typing import List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

from langchain_google_cloud_sql_pg.postgresql_engine import PostgreSQLEngine


class PostgreSQLChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in a Postgres database."""

    def __init__(self, engine: PostgreSQLEngine, session_id: str, table_name: str):
        self.engine = engine
        self.session_id = session_id
        self.table_name = table_name

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve the messages from PostgreSQL"""
        query = f"""SELECT data, type FROM "{self.table_name}" WHERE session_id = :session_id ORDER BY id;"""
        results = self.engine.run_as_sync(
            self.engine._afetch(query, {"session_id": self.session_id})
        )
        if not results:
            return []

        items = [{"data": result["data"], "type": result["type"]} for result in results]
        messages = messages_from_dict(items)
        return messages

    async def aadd_message(self, message: BaseMessage) -> None:
        """Append the message to the record in PostgreSQL"""
        query = f"""INSERT INTO "{self.table_name}"(session_id, data, type)
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

    def add_message(self, message: BaseMessage) -> None:
        self.engine.run_as_sync(self.aadd_message(message))

    async def aclear(self) -> None:
        """Clear session memory from PostgreSQL"""
        query = f"""DELETE FROM "{self.table_name}" WHERE session_id = :session_id;"""
        await self.engine._aexecute(query, {"session_id": self.session_id})

    def clear(self) -> None:
        self.engine.run_as_sync(self.aclear())
