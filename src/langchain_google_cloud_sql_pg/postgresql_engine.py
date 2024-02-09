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

import asyncio
from threading import Thread
from typing import TYPE_CHECKING, Awaitable, Dict, List, Optional, TypeVar

import aiohttp
import google.auth  # type: ignore
import google.auth.transport.requests  # type: ignore
from google.cloud.sql.connector import Connector, create_async_connector
from sqlalchemy import Column, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

if TYPE_CHECKING:
    import asyncpg  # type: ignore
    import google.auth.credentials  # type: ignore

T = TypeVar("T")


async def _get_iam_principal_email(
    credentials: google.auth.credentials.Credentials,
) -> str:
    """Get email address associated with current authenticated IAM principal.

    Email will be used for automatic IAM database authentication to Cloud SQL.

    Args:
        credentials (google.auth.credentials.Credentials):
            The credentials object to use in finding the associated IAM
            principal email address.

    Returns:
        email (str):
            The email address associated with the current authenticated IAM
            principal.
    """
    # refresh credentials if they are not valid
    if not credentials.valid:
        request = google.auth.transport.requests.Request()
        credentials.refresh(request)
    # call OAuth2 api to get IAM principal email associated with OAuth2 token
    url = f"https://oauth2.googleapis.com/tokeninfo?access_token={credentials.token}"
    async with aiohttp.ClientSession() as client:
        response = await client.get(url, raise_for_status=True)
        response_json: Dict = await response.json()
        email = response_json.get("email")
    if email is None:
        raise ValueError(
            "Failed to automatically obtain authenticated IAM princpal's "
            "email address using environment's ADC credentials!"
        )
    return email


class PostgreSQLEngine:
    """A class for managing connections to a Cloud SQL for Postgres database."""

    _connector: Optional[Connector] = None

    def __init__(
        self,
        engine: AsyncEngine,
        loop: Optional[asyncio.AbstractEventLoop],
        thread: Optional[Thread],
    ):
        self._engine = engine
        self._loop = loop
        self._thread = thread

    @classmethod
    def from_instance(
        cls,
        project_id: str,
        region: str,
        instance: str,
        database: str,
    ) -> PostgreSQLEngine:
        # Running a loop in a background thread allows us to support 
        # async methods from non-async enviroments
        loop = asyncio.new_event_loop()
        thread = Thread(target=loop.run_forever, daemon=True)
        thread.start()
        coro = cls.afrom_instance(project_id, region, instance, database)
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    @classmethod
    async def _create(
        cls,
        project_id: str,
        region: str,
        instance: str,
        database: str,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        thread: Optional[Thread] = None,
    ) -> PostgreSQLEngine:
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/userinfo.email"]
        )
        iam_database_user = await _get_iam_principal_email(credentials)
        if cls._connector is None:
            cls._connector = await create_async_connector()

        # anonymous function to be used for SQLAlchemy 'creator' argument
        def getconn() -> asyncpg.Connection:
            conn = cls._connector.connect_async(  # type: ignore
                f"{project_id}:{region}:{instance}",
                "asyncpg",
                user=iam_database_user,
                db=database,
                enable_iam_auth=True,
            )
            return conn

        engine = create_async_engine(
            "postgresql+asyncpg://",
            async_creator=getconn,
        )
        return cls(engine, loop, thread)

    @classmethod
    async def afrom_instance(
        cls,
        project_id: str,
        region: str,
        instance: str,
        database: str,
    ) -> PostgreSQLEngine:
        return await cls._create(project_id, region, instance, database)

    async def aexecute(self, query: str):
        async with self._engine.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    async def _afetch(self, query: str):
        async with self._engine.connect() as conn:
            """ Fetch results from a SQL query."""
            result = await conn.execute(text(query))
            result_map = result.mappings()
            result_fetch = result_map.fetchall()

        return result_fetch

    def run_as_sync(self, coro: Awaitable[T]):  # TODO: add return type
        if not self._loop:
            raise Exception("Engine was initialized async.")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    async def ainit_vectorstore_table(
        self,
        table_name: str,
        vector_size: int,
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: List[Column] = [],
        id_column: str = "langchain_id",
        overwrite_existing: bool = False,
        store_metadata: bool = True,
    ) -> None:
        await self.aexecute("CREATE EXTENSION IF NOT EXISTS vector")

        if overwrite_existing:
            await self.aexecute(f"DROP TABLE {table_name}")

        query = f"""CREATE TABLE IF NOT EXISTS {table_name}(
            {id_column} UUID PRIMARY KEY,
            {content_column} TEXT NOT NULL,
            {embedding_column} vector({vector_size}) NOT NULL"""
        for column in metadata_columns:
            query += f",\n{column.name} {column.type}" + (
                "NOT NULL" if not column.nullable else ""
            )
        if store_metadata:
            query += ",\nlangchain_metadata JSON"
        query += "\n);"

        await self.aexecute(query)
