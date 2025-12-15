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
import asyncio
import uuid

from config import (
    CHAT_TABLE_NAME,
    DATABASE,
    INSTANCE,
    PASSWORD,
    PROJECT_ID,
    REGION,
    TABLE_NAME,
    USER,
)
from google.cloud import resourcemanager_v3  # type: ignore
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_google_vertexai import VertexAIEmbeddings
from sqlalchemy import text

from langchain_google_cloud_sql_pg import PostgresEngine, PostgresVectorStore


def create_databases():
    engine = PostgresEngine.from_instance(
        PROJECT_ID,
        REGION,
        INSTANCE,
        database="postgres",
        user=USER,
        password=PASSWORD,
    )
    async def _create_logic():
        async with engine._pool.connect() as conn:
            await conn.execute(text("COMMIT"))
            await conn.execute(text(f'DROP DATABASE IF EXISTS "{DATABASE}"'))
            await conn.execute(text(f'CREATE DATABASE "{DATABASE}"'))
    engine._run_as_sync(_create_logic())
    engine.close()


def create_vectorstore():
    engine = PostgresEngine.from_instance(
        PROJECT_ID,
        REGION,
        INSTANCE,
        DATABASE,
        user=USER,
        password=PASSWORD,
    )

    engine.init_vectorstore_table(
        table_name=TABLE_NAME, vector_size=768, overwrite_existing=True
    )

    engine.init_chat_history_table(table_name=CHAT_TABLE_NAME)

    rm = resourcemanager_v3.ProjectsClient()
    res = rm.get_project(
        request=resourcemanager_v3.GetProjectRequest(name=f"projects/{PROJECT_ID}")
    )
    project_number = res.name.split("/")[1]
    IAM_USER = f"service-{project_number}@gcp-sa-aiplatform-re.iam"

    async def grant_select(engine):
        async with engine._pool.connect() as conn:
            await conn.execute(text(f'GRANT SELECT ON {TABLE_NAME} TO "{IAM_USER}";'))
            await conn.commit()

    engine._run_as_sync(grant_select(engine))

    metadata = [
        "show_id",
        "type",
        "country",
        "date_added",
        "release_year",
        "rating",
        "duration",
        "listed_in",
    ]
    loader = CSVLoader(file_path="./movies.csv", metadata_columns=metadata)
    docs = loader.load()

    vector_store = PostgresVectorStore.create_sync(
        engine,
        table_name=TABLE_NAME,
        embedding_service=VertexAIEmbeddings(
            model_name="textembedding-gecko@latest", project=PROJECT_ID
        ),
    )

    ids = [str(uuid.uuid4()) for i in range(len(docs))]
    vector_store.add_documents(docs, ids=ids)


def main():
    create_databases()
    create_vectorstore()

if __name__ == "__main__":
    main()
