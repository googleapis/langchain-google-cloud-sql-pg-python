import os
from typing import Generator

import pytest
import sqlalchemy
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage

from langchain_google_cloud_sql_pg import PostgreSQLChatMessageHistory, PostgreSQLEngine

project_id = os.environ["PROJECT_ID"]
region = os.environ["REGION"]
instance_id = os.environ["INSTANCE_ID"]
db_name = os.environ["DB_NAME"]


@pytest.fixture(name="memory_engine")
def setup() -> Generator:
    engine = PostgreSQLEngine.from_instance(
        project_id=project_id, region=region, instance=instance_id, database=db_name
    )

    yield engine
    # use default table for PostgreSQLChatMessageHistory
    table_name = "message_store"
    query = f"DROP TABLE IF EXISTS `{table_name}`"
    engine._aexecute(query)

def test_chat_message_history(memory_engine: PostgreSQLEngine) -> None:
    history = PostgreSQLChatMessageHistory(engine=memory_engine, session_id="test")
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


def test_chat_message_history_custom_table_name(memory_engine: PostgreSQLEngine) -> None:
    """Test PostgreSQLChatMessageHistory with custom table name"""
    history = PostgreSQLChatMessageHistory(
        engine=memory_engine, session_id="test", table_name="message-store"
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
