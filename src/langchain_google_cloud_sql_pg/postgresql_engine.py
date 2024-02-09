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

# TODO: Remove below import when minimum supported Python version is 3.10
from __future__ import annotations

# import requests
# import sqlalchemy
import asyncio
from threading import Thread
from typing import TYPE_CHECKING, Dict, List, Optional, Type

import aiohttp
import google.auth
import google.auth.transport.requests
import nest_asyncio
from google.cloud.sql.connector import Connector, create_async_connector

# from pgvector.asyncpg import register_vector
from sqlalchemy import Column, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

# nest_asyncio.apply()

if TYPE_CHECKING:
    import asyncpg
    import google.auth.credentials


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

    __create_key = object()

    def __init__(
        self,
        key,
        project_id=None,
        region=None,
        instance=None,
        database=None,
        engine=None,
    ):
        if key != PostgreSQLEngine.__create_key:
            raise Exception(
                "Only create class through from_instance and from_engine methods!"
            )
        self.project_id = project_id
        self.region = region
        self.instance = instance
        self.database = database
        self.engine = engine
        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._pool = asyncio.run_coroutine_threadsafe(
            self._engine(), self._loop
        ).result()

    @classmethod
    def from_instance(
        cls,
        region: str,
        instance: str,
        database: str,
        project_id: str = None,
    ) -> PostgreSQLEngine:
        """Create PostgreSQLEngine connection to the postgres database in the CloudSQL instance.
        Args:
            region (str): CloudSQL instance region.
            instance (str): CloudSQL instance name.
            database (str): CloudSQL instance database name.
            project_id (str): GCP project ID. Defaults to None
        Returns:
            PostgreSQLEngine containing the asyncpg connection pool.
        """
        return cls(
            project_id=project_id,
            region=region,
            instance=instance,
            database=database,
            key=PostgreSQLEngine.__create_key,
        )

    @classmethod
    def from_engine(cls, engine: AsyncEngine) -> PostgreSQLEngine:
        return cls(engine=engine, key=PostgreSQLEngine.__create_key)

    async def _engine(self) -> AsyncEngine:
        if self.engine is not None:
            return self.engine

        credentials, _ = google.auth.default(
            scopes=["email", "https://www.googleapis.com/auth/cloud-platform"]
        )

        if self.project_id is None:
            self.project_id = _

        async def get_conn():
            async with Connector(loop=asyncio.get_running_loop()) as connector:
                conn = await connector.connect_async(
                    f"{self.project_id}:{self.region}:{self.instance}",
                    "asyncpg",
                    # user=await _get_iam_principal_email(credentials),
                    user="postgres",
                    password="my-pg-pass",
                    enable_iam_auth=True,
                    db=self.database,
                )

            return conn

        pool = create_async_engine(
            "postgresql+asyncpg://",
            # poolclass=NullPool,
            async_creator=get_conn,
        )

        return pool

    async def _aexecute_fetch(self, query):
        async with self._pool.connect() as conn:
            result = await conn.execute(text(query))
            result_map = result.mappings()
            result_fetch = result_map.fetchall()

        return result_fetch

    async def _aexecute_update(self, query, additional=None) -> None:
        async with self._pool.connect() as conn:
            await conn.execute(text(query), additional)
            await conn.commit()

    async def init_vectorstore_table(
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
        # async with self.engine.connect() as conn:
        # Enable pgvector
        # await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await self._aexecute_update("CREATE EXTENSION IF NOT EXISTS vector")
        # Register the vector type
        # await register_vector(conn)

        if overwrite_existing:
            await self._aexecute_update(f"DROP TABLE {table_name}")
            # await conn.execute(
            #     text(f"TRUNCATE TABLE {table_name} RESET IDENTITY")
            # )  # TODO?

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

        await self._aexecute_update(query)
