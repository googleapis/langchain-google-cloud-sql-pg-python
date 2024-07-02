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
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from vertexai.preview import reasoning_engines  # type: ignore

from langchain_google_cloud_sql_pg import PostgresEngine, PostgresVectorStore

# This sample requires a vector store table
# Create this table using `PostgresEngine` method: init_vectorstore_table()

# Replace the following variables with your values
PROJECT_ID = os.getenv("PROJECT_ID") or "my-project-id"
STAGING_BUCKET = os.getenv("STAGING_BUCKET") or "gs://my-bucket"
REGION = os.getenv("REGION") or "us-central1"
INSTANCE = os.getenv("INSTANCE") or "my-instance"
DATABASE = os.getenv("DATABASE") or "my_database"
TABLE_NAME = os.getenv("TABLE_NAME") or "my_test_table"
USER = os.getenv("DB_USER") or "postgres"
PASSWORD = os.getenv("DB_PASSWORD") or "password"


class PostgresRetriever(reasoning_engines.Queryable):
    def __init__(
        self,
        model: str,
        project: str,
        region: str,
        instance: str,
        database: str,
        table: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.model_name = model
        self.project = project
        self.region = region
        self.instance = instance
        self.database = database
        self.table = table
        self.user = user
        self.password = password

    def set_up(self):
        """All unpickle-able logic should go here.
        In general, add any logic that requires a network or database
        connection.
        """
        # Create a chain to handle the processing of relevant documents
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        llm = VertexAI(model_name=self.model_name, project=self.project)
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)

        # Initialize the vector store and retriever
        engine = PostgresEngine.from_instance(
            self.project,
            self.region,
            self.instance,
            self.database,
            user=self.user,
            password=self.password,
            quota_project=self.project,  # Set quota project to ensure use of project's credentials
        )
        vector_store = PostgresVectorStore.create_sync(
            engine,
            table_name=self.table,
            embedding_service=VertexAIEmbeddings(
                model_name="textembedding-gecko@latest", project=self.project
            ),
        )
        retriever = vector_store.as_retriever()

        # Create a retrieval chain to fetch relevant documents and pass them to
        # an LLM to generate a response
        self.chain = create_retrieval_chain(retriever, combine_docs_chain)

    def query(self, input: str) -> str:
        """Query the application.

        Args:
            input: The user query.

        Returns:
            The LLM response dictionary.
        """
        # Define the runtime logic that serves user queries
        response = self.chain.invoke({"input": input})
        return response["answer"]


# Uncomment to test locally

# app = PostgresRetriever(
#     model="gemini-1.0-pro",
#     project=PROJECT_ID,
#     region=REGION,
#     instance=INSTANCE,
#     database=DATABASE,
#     table=TABLE,
#     user=USER,
#     password=PASSWORD,
# )
# app.set_up()
# print(app.query(input="movies about engineers"))

# Initialize VertexAI
vertexai.init(project=PROJECT_ID, location="us-central1", staging_bucket=STAGING_BUCKET)

# Deploy to VertexAI
DISPLAY_NAME = os.getenv("DISPLAY_NAME") or "PostgresRetriever"

remote_app = reasoning_engines.ReasoningEngine.create(
    PostgresRetriever(
        model="gemini-1.0-pro",
        project=PROJECT_ID,
        region=REGION,
        instance=INSTANCE,
        database=DATABASE,
        table=TABLE_NAME,
        # To use IAM authentication, remove user and password and ensure
        # the Reasoning Engine Agent service account is a database user
        # with access to the vector store table
        user=USER,
        password=PASSWORD,
    ),
    requirements=[
        "google-cloud-aiplatform[reasoningengine,langchain]==1.57.0",
        "langchain-google-cloud-sql-pg==0.6.1",
        "langchain-google-vertexai==1.0.4",
    ],
    display_name=DISPLAY_NAME,
    sys_version="3.11",
)

print(remote_app.query(input="movies about engineers"))
