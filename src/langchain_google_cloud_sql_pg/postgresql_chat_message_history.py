from __future__ import annotations
import json
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_google_cloud_sql_pg_python.postgresql_engine import PostgreSQLEngine
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)

class PostgreSQLChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in a Postgres database."""

    def __init__(
        self,
        engine:CloudSQLEngine,
        session_id: str,
        table_name: str
    ):

        self.engine = engine
        self.session_id = session_id
        self.table_name = table_name
        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self) -> None:
        create_table_query = f"""CREATE TABLE IF NOT EXISTS {self.table_name} (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            data JSONB NOT NULL,
            type TEXT NOT NULL,
            content TEXT NOT NULL,
            example TEXT NOT NULL,
            additional_kwargs JSONB NOT NULL

        );"""
        
        asyncio.run_coroutine_threadsafe(self.engine._aexecute(create_table_query), self._loop).result()
        
    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve the messages from PostgreSQL"""
        query = (
            f"SELECT data, type FROM {self.table_name} WHERE session_id = '{self.session_id}' ORDER BY id;"
        )
        results = asyncio.run_coroutine_threadsafe(self.engine._afetch(query), self._loop).result()
        if not results:
            return []
    
        items = [{"data":result['data'], "type":result['type']} for result in results]
        messages = messages_from_dict(items)
        return messages

    async def aadd_message(self, message: BaseMessage) -> None:
        """Append the message to the record in PostgreSQL"""

        data = json.dumps(message.dict())
        content = message.dict()['content']
        example = message.dict()['example']
        additional_kwargs = message.dict()['additional_kwargs']
        query = f"INSERT INTO {self.table_name} (session_id, data, type, content, example, additional_kwargs) VALUES ('{self.session_id}','{data}','{message.type}','{content}','{example}','{additional_kwargs}');"
        # asyncio.run_coroutine_threadsafe(self.engine._aexecute(query), self._loop).result()
        await self.engine._aexecute(query)
    
    def add_message(self, message: BaseMessage) -> None:
        asyncio.run_coroutine_threadsafe(self.aadd_message(message), self._loop).result()

    async def aclear(self) -> None:
        """Clear session memory from PostgreSQL"""
        query = f"DELETE FROM {self.table_name} WHERE session_id = '{self.session_id}';"
        # asyncio.run_coroutine_threadsafe(self.engine._aexecute(query), self._loop).result()
        await self.engine._aexecute(query)
    
    def clear(self) -> None:
        asyncio.run_coroutine_threadsafe(self.aclear(), self._loop).result()
