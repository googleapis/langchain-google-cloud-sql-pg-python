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
import os

from vertexai.preview import reasoning_engines  # type: ignore

from langchain_google_cloud_sql_pg import PostgresEngine

PROJECT_ID = os.getenv("PROJECT_ID") or "my-project-id"
REGION = os.getenv("REGION") or "us-central1"
INSTANCE = os.getenv("INSTANCE") or "my-primary"
DATABASE = os.getenv("DATABASE") or "my_database"
TABLE_NAME = os.getenv("TABLE_NAME") or "my_test_table"
CHAT_TABLE_NAME = os.getenv("CHAT_TABLE_NAME") or "my_chat_table"
USER = os.getenv("DB_USER") or "postgres"
PASSWORD = os.getenv("DB_PASSWORD") or "password"
TEST_NAME = os.getenv("DISPLAY_NAME")


async def delete_tables():
    engine = await PostgresEngine.afrom_instance(
        PROJECT_ID,
        REGION,
        INSTANCE,
        database=DATABASE,
        user=USER,
        password=PASSWORD,
    )

    await engine._aexecute_outside_tx(f"DROP TABLE IF EXISTS {TABLE_NAME}")
    await engine._aexecute_outside_tx(f"DROP TABLE IF EXISTS {CHAT_TABLE_NAME}")


def delete_engines():
    apps = reasoning_engines.ReasoningEngine.list(filter=f'display_name="{TEST_NAME}"')
    for app in apps:
        app.delete()


async def main():
    await delete_tables()
    delete_engines()


asyncio.run(main())
