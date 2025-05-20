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

import vertexai  # type: ignore
from config import (
    DATABASE,
    INSTANCE,
    PASSWORD,
    PROJECT_ID,
    REGION,
    STAGING_BUCKET,
    TABLE_NAME,
    USER,
)
from langchain_core.documents import Document
from langchain_google_vertexai import VertexAIEmbeddings
from vertexai.preview import reasoning_engines  # type: ignore

from langchain_google_cloud_sql_pg import PostgresEngine, PostgresVectorStore

# This sample requires a vector store table
# Create this table using `PostgresEngine` method `init_vectorstore_table()`
# or create and load the table using `create_embeddings.py`

engine = None  # Use global variable to share connection pooling


def similarity_search(query: str) -> list[Document]:
    """Searches and returns movies.

    Args:
      query: The user query to search for related items

    Returns:
      list[Document]: A list of Documents
    """
    global engine
    if not engine:  # Reuse connection pool
        engine = PostgresEngine.from_instance(
            PROJECT_ID,
            REGION,
            INSTANCE,
            DATABASE,
            quota_project=PROJECT_ID,
            # To use IAM authentication, remove user and password and ensure
            # the Reasoning Engine Agent service account is a database user
            # with access to the vector store table
            user=USER,
            password=PASSWORD,
        )

    vector_store = PostgresVectorStore.create_sync(
        engine,
        table_name=TABLE_NAME,
        embedding_service=VertexAIEmbeddings(
            model_name="textembedding-gecko@latest", project=PROJECT_ID
        ),
    )
    retriever = vector_store.as_retriever()
    return retriever.invoke(query)


# Uncomment to test locally

# app = reasoning_engines.LangchainAgent(
#     model="gemini-2.0-flash-001",
#     tools=[similarity_search],
#     model_kwargs={
#         "temperature": 0.1,
#     },
# )
# app.set_up()
# print(app.query(input="movies about engineers"))

# Initialize VertexAI
vertexai.init(project=PROJECT_ID, location="us-central1", staging_bucket=STAGING_BUCKET)

# Deploy to VertexAI
DISPLAY_NAME = os.getenv("DISPLAY_NAME") or "PrebuiltAgent"

remote_app = reasoning_engines.ReasoningEngine.create(
    reasoning_engines.LangchainAgent(
        model="gemini-2.0-flash-001",
        tools=[similarity_search],
        model_kwargs={
            "temperature": 0.1,
        },
    ),
    requirements="requirements.txt",
    display_name=DISPLAY_NAME,
    sys_version="3.11",
    extra_packages=["config.py"],
)

print(remote_app.query(input="movies about engineers"))
