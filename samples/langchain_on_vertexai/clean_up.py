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
from sqlalchemy import text
from vertexai.preview import reasoning_engines  # type: ignore

from langchain_google_cloud_sql_pg import PostgresEngine

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

    async with engine._pool.connect() as conn:
        await conn.execute(text("COMMIT"))
        await conn.execute(text(f"DROP TABLE IF EXISTS {TABLE_NAME}"))
        await conn.execute(text(f"DROP TABLE IF EXISTS {CHAT_TABLE_NAME}"))
    await engine.close()
    await engine._connector.close_async()


def delete_engines():
    apps = reasoning_engines.ReasoningEngine.list(filter=f'display_name="{TEST_NAME}"')
    for app in apps:
        app.delete()


async def main():
    await delete_tables()
    delete_engines()


asyncio.run(main())
