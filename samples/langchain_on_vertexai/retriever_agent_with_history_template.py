# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from typing import Optional

import vertexai  # type: ignore
from config import (
    CHAT_TABLE_NAME,
    DATABASE,
    INSTANCE,
    PASSWORD,
    PROJECT_ID,
    REGION,
    STAGING_BUCKET,
    TABLE_NAME,
    USER,
)
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from vertexai.preview import reasoning_engines  # type: ignore

from langchain_google_cloud_sql_pg import (
    PostgresChatMessageHistory,
    PostgresEngine,
    PostgresVectorStore,
)

# This sample requires a vector store table and a chat message table
# Create these tables using `PostgresEngine` methods
# `init_vectorstore_table()` and `init_chat_history_table()`
# or create and load the tables using `create_embeddings.py`


class PostgresAgent(reasoning_engines.Queryable):
    def __init__(
        self,
        model: str,
        project: str,
        region: str,
        instance: str,
        database: str,
        table: str,
        chat_table: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
        tool_name: str = "search_movies",
        tool_description: str = "Searches and returns movies.",
    ):
        self.model_name = model
        self.project = project
        self.region = region
        self.instance = instance
        self.database = database
        self.table = table
        self.chat_table = chat_table
        self.user = user
        self.password = password
        self.tool_name = tool_name
        self.tool_description = tool_description

    def set_up(self):
        """All unpickle-able logic should go here.
        In general, add any logic that requires a network or database
        connection.
        """
        # Initialize the vector store
        engine = PostgresEngine.from_instance(
            self.project,
            self.region,
            self.instance,
            self.database,
            user=self.user,
            password=self.password,
            quota_project=self.project,
        )
        vector_store = PostgresVectorStore.create_sync(
            engine,
            table_name=self.table,
            embedding_service=VertexAIEmbeddings(
                model_name="textembedding-gecko@latest", project=self.project
            ),
        )
        retriever = vector_store.as_retriever()

        # Create a tool to do retrieval of documents
        tool = create_retriever_tool(
            retriever,
            self.tool_name,
            self.tool_description,
        )
        tools = [tool]

        # Initialize the LLM and prompt
        llm = ChatVertexAI(model_name=self.model_name, project=self.project)
        base_prompt = hub.pull("langchain-ai/react-agent-template")
        instructions = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
        )
        prompt = base_prompt.partial(instructions=instructions)

        # Create an agent
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, tools=tools, handle_parsing_errors=True
        )

        # Initialize a Runnable that manages chat message history for the agent
        self.agent = RunnableWithMessageHistory(
            agent_executor,
            lambda session_id: PostgresChatMessageHistory.create_sync(
                engine=engine, session_id=session_id, table_name=self.chat_table
            ),
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    def query(self, input: str, session_id: str) -> str:
        """Query the application.

        Args:
            input: The user query.
            session_id: The user's session id.

        Returns:
            The LLM response dictionary.
        """
        response = self.agent.invoke(
            {"input": input},
            config={"configurable": {"session_id": session_id}},
        )
        return response["output"]


# Uncomment to test locally
# app = PostgresAgent(
#     model="gemini-2.0-flash-001",
#     project=PROJECT_ID,
#     region=REGION,
#     instance=INSTANCE,
#     database=DATABASE,
#     table=TABLE_NAME,
#     chat_table=CHAT_TABLE_NAME,
#     user=USER,
#     password=PASSWORD
# )

# app.set_up()
# print(app.query(input="What movies are about engineers?", session_id="abc123"))

# Initialize VertexAI
vertexai.init(project=PROJECT_ID, location="us-central1", staging_bucket=STAGING_BUCKET)

# Deploy to VertexAI
DISPLAY_NAME = os.getenv("DISPLAY_NAME") or "PostgresAgent"

remote_app = reasoning_engines.ReasoningEngine.create(
    PostgresAgent(
        model="gemini-2.0-flash-001",
        project=PROJECT_ID,
        region=REGION,
        instance=INSTANCE,
        database=DATABASE,
        table=TABLE_NAME,
        chat_table=CHAT_TABLE_NAME,
        # To use IAM authentication, remove user and password and ensure
        # the Reasoning Engine Agent service account is a database user
        # with access to vector store and chat tables
        user=USER,
        password=PASSWORD,
    ),
    requirements="requirements.txt",
    display_name=DISPLAY_NAME,
    sys_version="3.11",
    extra_packages=["config.py"],
)

print(remote_app.query(input="movies about engineers", session_id="abc123"))
