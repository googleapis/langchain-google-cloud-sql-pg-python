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
from dataclasses import dataclass
from threading import Thread
from typing import TYPE_CHECKING, Awaitable, Dict, List, Optional, TypeVar

import aiohttp
import google.auth  # type: ignore
import google.auth.transport.requests  # type: ignore
from google.cloud.sql.connector import Connector, create_async_connector
from sqlalchemy import text  # Column,
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
    if not credentials.valid:
        request = google.auth.transport.requests.Request()
        credentials.refresh(request)
    if hasattr(credentials, "_service_account_email"):
        email = credentials._service_account_email

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
    return email.replace(".gserviceaccount.com", "")



@dataclass
class Column:
    name: str
    data_type: str
    nullable: bool = True


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
        user: Optional[str] = None,
        password: Optional[str] = None,
    ) -> PostgreSQLEngine:
        # Running a loop in a background thread allows us to support
        # async methods from non-async environments
        loop = asyncio.new_event_loop()
        thread = Thread(target=loop.run_forever, daemon=True)
        thread.start()
        coro = cls._create(
            project_id,
            region,
            instance,
            database,
            user,
            password,
            loop=loop,
            thread=thread,
        )
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    @classmethod
    async def _create(
        cls,
        project_id: str,
        region: str,
        instance: str,
        database: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        thread: Optional[Thread] = None,
    ) -> PostgreSQLEngine:
        if bool(user) ^ bool(password):
            raise ValueError(
                "Only one of 'user' or 'password' were specified. Either "
                "both should be specified to use basic user/password "
                "authentication or neither for IAM DB authentication."
            )
        if cls._connector is None:
            cls._connector = await create_async_connector()
        # if user and password are given, use basic auth
        if user and password:
            enable_iam_auth = False
            db_user = user
        # otherwise use automatic IAM database authentication
        else:
            # get application default credentials
            credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/userinfo.email"]
            )
            db_user = await _get_iam_principal_email(credentials)
            enable_iam_auth = True

        # anonymous function to be used for SQLAlchemy 'creator' argument
        async def getconn() -> asyncpg.Connection:
            conn = await cls._connector.connect_async(  # type: ignore
                f"{project_id}:{region}:{instance}",
                "asyncpg",
                user=db_user,
                password=password,
                db=database,
                enable_iam_auth=enable_iam_auth,
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
        user: Optional[str] = None,
        password: Optional[str] = None,
    ) -> PostgreSQLEngine:
        return await cls._create(
            project_id,
            region,
            instance,
            database,
            user,
            password,
        )

    async def _aexecute(self, query: str):
        """Execute a SQL query."""
        async with self._engine.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    async def _aexecute_outside_tx(self, query: str):
        """Execute a SQL query."""
        async with self._engine.connect() as conn:
            await conn.execute(text("COMMIT"))
            await conn.execute(text(query))

    async def _afetch(self, query: str):
        async with self._engine.connect() as conn:
            """Fetch results from a SQL query."""
            result = await conn.execute(text(query))
            result_map = result.mappings()
            result_fetch = result_map.fetchall()

        return result_fetch

    def run_as_sync(self, coro: Awaitable[T]) -> T:
        if not self._loop:
            raise Exception("Engine was initialized async.")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    async def init_vectorstore_table(
        self,
        table_name: str,
        vector_size: int,
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: List[Column] = [],
        metadata_json_column: str = "langchain_metadata",
        id_column: str = "langchain_id",
        overwrite_existing: bool = False,
        store_metadata: bool = True,
    ) -> None:
        await self._aexecute("CREATE EXTENSION IF NOT EXISTS vector")

        if overwrite_existing:
            await self._aexecute(f"DROP TABLE IF EXISTS {table_name}")

        query = f"""CREATE TABLE {table_name}(
            {id_column} UUID PRIMARY KEY,
            {content_column} TEXT NOT NULL,
            {embedding_column} vector({vector_size}) NOT NULL"""
        for column in metadata_columns:
            query += f",\n{column.name} {column.data_type}" + (
                "NOT NULL" if not column.nullable else ""
            )
        if store_metadata:
            query += f",\n{metadata_json_column} JSON"
        query += "\n);"

        await self._aexecute(query)

    async def init_document_table(
        self,
        table_name: str,
        content_column: str = "page_content",
        metadata_columns: List[Column] = [],
        metadata_json_columns: str = "langchain_metadata",
        store_metadata: bool = True,
    ) -> None:
        """
        Create a table for saving of langchain documents.

        Args:
            table_name (str): The PgSQL database table name.
            metadata_columns (List[sqlalchemy.Column]): A list of SQLAlchemy Columns
                to create for custom metadata. Optional.
            store_metadata (bool): Whether to store extra metadata in a metadata column
                if not described in 'metadata' field list (Default: True).
        """

        query = f"""CREATE TABLE {table_name}(
            {content_column} TEXT NOT NULL
            """
        for column in metadata_columns:
            query += f",\n{column.name} {column.data_type}" + (
                "NOT NULL" if not column.nullable else ""
            )
        if store_metadata:
            query += f",\n{metadata_json_columns} JSON"
        query += "\n);"

        await self._aexecute(query)

    async def _load_document_table(self, table_name: str) -> sqlalchemy.Table:
        """
        Load table schema from existing table in PgSQL database.

        Args:
            table_name (str): The PgSQL database table name.

        Returns:
            (sqlalchemy.Table): The loaded table.
        """
        metadata = sqlalchemy.MetaData()
        #self._pool.connect().run_sync(metadata.reflect, only=[table_name])
        async with self._engine.connect() as conn:
            await conn.run_sync(metadata.reflect, only=[table_name])

        table = Table(table_name, metadata)
        # Extract the schema information
        schema = []
        for column in table.columns:
            schema.append({
                "name": column.name,
                "type": column.type.python_type,
                "max_length": getattr(column.type, "length", None),
                "nullable": not column.nullable
            })

        return metadata.tables[table_name]
