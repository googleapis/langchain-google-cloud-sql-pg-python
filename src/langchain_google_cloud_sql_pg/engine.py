# Copyright 2025 Google LLC
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
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Thread
from typing import TYPE_CHECKING, Any, Awaitable, Mapping, Optional, TypeVar, Union

import aiohttp
import google.auth  # type: ignore
import google.auth.transport.requests  # type: ignore
from google.cloud.sql.connector import Connector, IPTypes, RefreshStrategy
from sqlalchemy import MetaData, Table, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from .version import __version__

if TYPE_CHECKING:
    import asyncpg  # type: ignore
    import google.auth.credentials  # type: ignore

T = TypeVar("T")

USER_AGENT = "langchain-google-cloud-sql-pg-python/" + __version__

CHECKPOINTS_TABLE = "checkpoints"


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
        return credentials._service_account_email.replace(".gserviceaccount.com", "")
    # call OAuth2 api to get IAM principal email associated with OAuth2 token
    url = f"https://oauth2.googleapis.com/tokeninfo?access_token={credentials.token}"
    async with aiohttp.ClientSession() as client:
        response = await client.get(url, raise_for_status=True)
        response_json: dict = await response.json()
        email = response_json.get("email")
    if email is None:
        raise ValueError(
            "Failed to automatically obtain authenticated IAM principal's "
            "email address using environment's ADC credentials!"
        )
    return email.replace(".gserviceaccount.com", "")


@dataclass
class Column:
    name: str
    data_type: str
    nullable: bool = True

    def __post_init__(self):
        """Check if initialization parameters are valid.

        Raises:
            ValueError: Raises error if Column name is not string.
            ValueError: Raises error if data_type is not type string.
        """
        if not isinstance(self.name, str):
            raise ValueError("Column name must be type string")
        if not isinstance(self.data_type, str):
            raise ValueError("Column data_type must be type string")


class PostgresEngine:
    """A class for managing connections to a Cloud SQL for Postgres database."""

    _connector: Optional[Connector] = None
    _default_loop: Optional[asyncio.AbstractEventLoop] = None
    _default_thread: Optional[Thread] = None
    __create_key = object()

    def __init__(
        self,
        key: object,
        pool: AsyncEngine,
        loop: Optional[asyncio.AbstractEventLoop],
        thread: Optional[Thread],
    ):
        """PostgresEngine constructor.

        Args:
            key (object): Prevent direct constructor usage.
            pool (AsyncEngine): Async engine connection pool.
            loop (Optional[asyncio.AbstractEventLoop]): Async event loop used to create the engine.
            thread (Optional[Thread]): Thread used to create the engine async.

        Raises:
            Exception: If the constructor is called directly by the user.
        """
        if key != PostgresEngine.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync' methods!"
            )
        self._pool = pool
        self._loop = loop
        self._thread = thread

    @classmethod
    async def _create(
        cls,
        project_id: str,
        region: str,
        instance: str,
        database: str,
        ip_type: Union[str, IPTypes],
        user: Optional[str] = None,
        password: Optional[str] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        thread: Optional[Thread] = None,
        quota_project: Optional[str] = None,
        iam_account_email: Optional[str] = None,
        engine_args: Mapping = {},
    ) -> PostgresEngine:
        """Create a PostgresEngine instance.

        Args:
            project_id (str): GCP project ID.
            region (str): Postgres instance region.
            instance (str): Postgres instance name.
            database (str): Database name.
            ip_type (Union[str, IPTypes]): IP address type. Defaults to IPTypes.PUBLIC.
            user (Optional[str]): Postgres user name. Defaults to None.
            password (Optional[str]): Postgres user password. Defaults to None.
            loop (Optional[asyncio.AbstractEventLoop]): Async event loop used to create the engine.
            thread (Optional[Thread]): Thread used to create the engine async.
            quota_project (Optional[str]): Project that provides quota for API calls.
            iam_account_email (Optional[str]): IAM service account email. Defaults to None.
            engine_args (Mapping): Additional arguments that are passed directly to
                :func:`~sqlalchemy.ext.asyncio.mymodule.MyClass.create_async_engine`. This can be
                used to specify additional parameters to the underlying pool during it's creation.

        Raises:
            ValueError: If only one of `user` and `password` is specified.

        Returns:
            PostgresEngine
        """
        if bool(user) ^ bool(password):
            raise ValueError(
                "Only one of 'user' or 'password' were specified. Either "
                "both should be specified to use basic user/password "
                "authentication or neither for IAM DB authentication."
            )
        if cls._connector is None:
            cls._connector = Connector(
                loop=loop,
                user_agent=USER_AGENT,
                quota_project=quota_project,
                refresh_strategy=RefreshStrategy.LAZY,
            )

        # if user and password are given, use basic auth
        if user and password:
            enable_iam_auth = False
            db_user = user
        # otherwise use automatic IAM database authentication
        else:
            enable_iam_auth = True
            if iam_account_email:
                db_user = iam_account_email
            else:
                # get application default credentials
                credentials, _ = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/userinfo.email"]
                )
                db_user = await _get_iam_principal_email(credentials)

        # anonymous function to be used for SQLAlchemy 'creator' argument
        async def getconn() -> asyncpg.Connection:
            conn = await cls._connector.connect_async(  # type: ignore
                f"{project_id}:{region}:{instance}",
                "asyncpg",
                user=db_user,
                password=password,
                db=database,
                enable_iam_auth=enable_iam_auth,
                ip_type=ip_type,
            )
            return conn

        engine = create_async_engine(
            "postgresql+asyncpg://",
            async_creator=getconn,
            **engine_args,
        )
        return cls(cls.__create_key, engine, loop, thread)

    @classmethod
    def __start_background_loop(
        cls,
        project_id: str,
        region: str,
        instance: str,
        database: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
        ip_type: Union[str, IPTypes] = IPTypes.PUBLIC,
        quota_project: Optional[str] = None,
        iam_account_email: Optional[str] = None,
        engine_args: Mapping = {},
    ) -> Future:
        # Running a loop in a background thread allows us to support
        # async methods from non-async environments
        if cls._default_loop is None:
            cls._default_loop = asyncio.new_event_loop()
            cls._default_thread = Thread(
                target=cls._default_loop.run_forever, daemon=True
            )
            cls._default_thread.start()
        coro = cls._create(
            project_id,
            region,
            instance,
            database,
            ip_type,
            user,
            password,
            loop=cls._default_loop,
            thread=cls._default_thread,
            quota_project=quota_project,
            iam_account_email=iam_account_email,
            engine_args=engine_args,
        )
        return asyncio.run_coroutine_threadsafe(coro, cls._default_loop)

    @classmethod
    def from_instance(
        cls,
        project_id: str,
        region: str,
        instance: str,
        database: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
        ip_type: Union[str, IPTypes] = IPTypes.PUBLIC,
        quota_project: Optional[str] = None,
        iam_account_email: Optional[str] = None,
        engine_args: Mapping = {},
    ) -> PostgresEngine:
        """Create a PostgresEngine from a Postgres instance.

        Args:
            project_id (str): GCP project ID.
            region (str): Postgres instance region.
            instance (str): Postgres instance name.
            database (str): Database name.
            user (Optional[str], optional): Postgres user name. Defaults to None.
            password (Optional[str], optional): Postgres user password. Defaults to None.
            ip_type (Union[str, IPTypes], optional): IP address type. Defaults to IPTypes.PUBLIC.
            quota_project (Optional[str]): Project that provides quota for API calls.
            iam_account_email (Optional[str], optional): IAM service account email. Defaults to None.
            engine_args (Mapping): Additional arguments that are passed directly to
                :func:`~sqlalchemy.ext.asyncio.mymodule.MyClass.create_async_engine`. This can be
                used to specify additional parameters to the underlying pool during it's creation.

        Returns:
            PostgresEngine: A newly created PostgresEngine instance.
        """
        future = cls.__start_background_loop(
            project_id,
            region,
            instance,
            database,
            user,
            password,
            ip_type,
            quota_project=quota_project,
            iam_account_email=iam_account_email,
            engine_args=engine_args,
        )
        return future.result()

    @classmethod
    async def afrom_instance(
        cls,
        project_id: str,
        region: str,
        instance: str,
        database: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
        ip_type: Union[str, IPTypes] = IPTypes.PUBLIC,
        quota_project: Optional[str] = None,
        iam_account_email: Optional[str] = None,
        engine_args: Mapping = {},
    ) -> PostgresEngine:
        """Create a PostgresEngine from a Postgres instance.

        Args:
            project_id (str): GCP project ID.
            region (str): Postgres instance region.
            instance (str): Postgres instance name.
            database (str): Database name.
            user (Optional[str], optional): Postgres user name. Defaults to None.
            password (Optional[str], optional): Postgres user password. Defaults to None.
            ip_type (Union[str, IPTypes], optional): IP address type. Defaults to IPTypes.PUBLIC.
            quota_project (Optional[str]): Project that provides quota for API calls.
            iam_account_email (Optional[str], optional): IAM service account email. Defaults to None.
            engine_args (Mapping): Additional arguments that are passed directly to
                :func:`~sqlalchemy.ext.asyncio.mymodule.MyClass.create_async_engine`. This can be
                used to specify additional parameters to the underlying pool during it's creation.

        Returns:
            PostgresEngine: A newly created PostgresEngine instance.
        """
        future = cls.__start_background_loop(
            project_id,
            region,
            instance,
            database,
            user,
            password,
            ip_type,
            quota_project=quota_project,
            iam_account_email=iam_account_email,
            engine_args=engine_args,
        )
        return await asyncio.wrap_future(future)

    @classmethod
    def from_engine(
        cls,
        engine: AsyncEngine,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> PostgresEngine:
        """Create an PostgresEngine instance from an AsyncEngine."""
        return cls(cls.__create_key, engine, loop, None)

    @classmethod
    def from_engine_args(
        cls,
        url: str | URL,
        **kwargs: Any,
    ) -> PostgresEngine:
        """Create an PostgresEngine instance from arguments. These parameters are pass directly into sqlalchemy's create_async_engine function.

        Args:
            url (Union[str | URL]): the URL used to connect to a database
            **kwargs (Any, optional): sqlalchemy `create_async_engine` arguments

        Raises:
            ValueError: If `postgresql+asyncpg` is not specified as the PG driver

        Returns:
            PostgresEngine
        """
        # Running a loop in a background thread allows us to support
        # async methods from non-async environments
        if cls._default_loop is None:
            cls._default_loop = asyncio.new_event_loop()
            cls._default_thread = Thread(
                target=cls._default_loop.run_forever, daemon=True
            )
            cls._default_thread.start()

        driver = "postgresql+asyncpg"
        if (isinstance(url, str) and not url.startswith(driver)) or (
            isinstance(url, URL) and url.drivername != driver
        ):
            raise ValueError("Driver must be type 'postgresql+asyncpg'")

        engine = create_async_engine(url, **kwargs)
        return cls(cls.__create_key, engine, cls._default_loop, cls._default_thread)

    async def _run_as_async(self, coro: Awaitable[T]) -> T:
        """Run an async coroutine asynchronously"""
        # If a loop has not been provided, attempt to run in current thread
        if not self._loop:
            return await coro
        # Otherwise, run in the background thread
        return await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        )

    def _run_as_sync(self, coro: Awaitable[T]) -> T:
        """Run an async coroutine synchronously"""
        if not self._loop:
            raise Exception(
                "Engine was initialized without a background loop and cannot call sync methods."
            )
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    async def close(self) -> None:
        """Dispose of connection pool"""
        await self._run_as_async(self._pool.dispose())

    async def _ainit_vectorstore_table(
        self,
        table_name: str,
        vector_size: int,
        schema_name: str = "public",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: list[Column] = [],
        metadata_json_column: str = "langchain_metadata",
        id_column: Union[str, Column] = "langchain_id",
        overwrite_existing: bool = False,
        store_metadata: bool = True,
    ) -> None:
        """
        Create a table for saving of vectors to be used with PostgresVectorStore.

        Args:
            table_name (str): The Postgres database table name.
            vector_size (int): Vector size for the embedding model to be used.
            schema_name (str): The schema name to store Postgres database table.
                Default: "public".
            content_column (str): Name of the column to store document content.
                Default: "page_content".
            embedding_column (str) : Name of the column to store vector embeddings.
                Default: "embedding".
            metadata_columns (list[Column]): A list of Columns to create for custom
                metadata. Default: []. Optional.
            metadata_json_column (str): The column to store extra metadata in JSON format.
                Default: "langchain_metadata". Optional.
            id_column (Union[str, Column]) : Column to store ids.
                Default: "langchain_id" column name with data type UUID. Optional.
            overwrite_existing (bool): Whether to drop existing table. Default: False.
            store_metadata (bool): Whether to store metadata in the table.
                Default: True.
        Raises:
            :class:`DuplicateTableError <asyncpg.exceptions.DuplicateTableError>`: if table already exists and overwrite flag is not set.
            :class:`UndefinedObjectError <asyncpg.exceptions.UndefinedObjectError>`: if the data type of the id column is not a postgreSQL data type.
        """
        async with self._pool.connect() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.commit()

        if overwrite_existing:
            async with self._pool.connect() as conn:
                await conn.execute(
                    text(f'DROP TABLE IF EXISTS "{schema_name}"."{table_name}"')
                )
                await conn.commit()

        id_data_type = "UUID" if isinstance(id_column, str) else id_column.data_type
        id_column_name = id_column if isinstance(id_column, str) else id_column.name

        query = f"""CREATE TABLE "{schema_name}"."{table_name}"(
            "{id_column_name}" {id_data_type} PRIMARY KEY,
            "{content_column}" TEXT NOT NULL,
            "{embedding_column}" vector({vector_size}) NOT NULL"""
        for column in metadata_columns:
            nullable = "NOT NULL" if not column.nullable else ""
            query += f',\n"{column.name}" {column.data_type} {nullable}'
        if store_metadata:
            query += f""",\n"{metadata_json_column}" JSON"""
        query += "\n);"

        async with self._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    async def ainit_vectorstore_table(
        self,
        table_name: str,
        vector_size: int,
        schema_name: str = "public",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: list[Column] = [],
        metadata_json_column: str = "langchain_metadata",
        id_column: Union[str, Column] = "langchain_id",
        overwrite_existing: bool = False,
        store_metadata: bool = True,
    ) -> None:
        """
        Create a table for saving of vectors to be used with PostgresVectorStore.

        Args:
            table_name (str): The Postgres database table name.
            vector_size (int): Vector size for the embedding model to be used.
            schema_name (str): The schema name to store Postgres database table.
                Default: "public".
            content_column (str): Name of the column to store document content.
                Default: "page_content".
            embedding_column (str) : Name of the column to store vector embeddings.
                Default: "embedding".
            metadata_columns (list[Column]): A list of Columns to create for custom
                metadata. Default: []. Optional.
            metadata_json_column (str): The column to store extra metadata in JSON format.
                Default: "langchain_metadata". Optional.
            id_column (Union[str, Column]) : Column to store ids.
                Default: "langchain_id" column name with data type UUID. Optional.
            overwrite_existing (bool): Whether to drop existing table. Default: False.
            store_metadata (bool): Whether to store metadata in the table.
                Default: True.
        """
        await self._run_as_async(
            self._ainit_vectorstore_table(
                table_name,
                vector_size,
                schema_name,
                content_column,
                embedding_column,
                metadata_columns,
                metadata_json_column,
                id_column,
                overwrite_existing,
                store_metadata,
            )
        )

    def init_vectorstore_table(
        self,
        table_name: str,
        vector_size: int,
        schema_name: str = "public",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: list[Column] = [],
        metadata_json_column: str = "langchain_metadata",
        id_column: Union[str, Column] = "langchain_id",
        overwrite_existing: bool = False,
        store_metadata: bool = True,
    ) -> None:
        """
        Create a table for saving of vectors to be used with PostgresVectorStore.

        Args:
            table_name (str): The Postgres database table name.
            vector_size (int): Vector size for the embedding model to be used.
            schema_name (str): The schema name to store Postgres database table.
                Default: "public".
            content_column (str): Name of the column to store document content.
                Default: "page_content".
            embedding_column (str) : Name of the column to store vector embeddings.
                Default: "embedding".
            metadata_columns (list[Column]): A list of Columns to create for custom
                metadata. Default: []. Optional.
            metadata_json_column (str): The column to store extra metadata in JSON format.
                Default: "langchain_metadata". Optional.
            id_column (Union[str, Column]) : Column to store ids.
                Default: "langchain_id" column name with data type UUID. Optional.
            overwrite_existing (bool): Whether to drop existing table. Default: False.
            store_metadata (bool): Whether to store metadata in the table.
                Default: True.
        Raises:
            :class:`UndefinedObjectError <asyncpg.exceptions.UndefinedObjectError>`: if the `ids` data type does not match that of the `id_column`.
        """
        self._run_as_sync(
            self._ainit_vectorstore_table(
                table_name,
                vector_size,
                schema_name,
                content_column,
                embedding_column,
                metadata_columns,
                metadata_json_column,
                id_column,
                overwrite_existing,
                store_metadata,
            )
        )

    async def _ainit_chat_history_table(
        self, table_name: str, schema_name: str = "public"
    ) -> None:
        """Create a Cloud SQL table to store chat history.

        Args:
            table_name (str): Table name to store chat history.
            schema_name (str): Schema name to store chat history table.
                Default: "public".

        Returns:
            None
        """
        create_table_query = f"""CREATE TABLE IF NOT EXISTS "{schema_name}"."{table_name}"(
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            data JSONB NOT NULL,
            type TEXT NOT NULL
        );"""
        async with self._pool.connect() as conn:
            await conn.execute(text(create_table_query))
            await conn.commit()

    async def ainit_chat_history_table(
        self, table_name: str, schema_name: str = "public"
    ) -> None:
        """Create a Cloud SQL table to store chat history.

        Args:
            table_name (str): Table name to store chat history.

        Returns:
            None
        """
        await self._run_as_async(
            self._ainit_chat_history_table(table_name, schema_name)
        )

    def init_chat_history_table(
        self, table_name: str, schema_name: str = "public"
    ) -> None:
        """Create a Cloud SQL table to store chat history.

        Args:
            table_name (str): Table name to store chat history.
            schema_name (str): Schema name to store chat history table.
                Default: "public".

        Returns:
            None
        """
        self._run_as_sync(
            self._ainit_chat_history_table(
                table_name,
                schema_name,
            )
        )

    async def _ainit_document_table(
        self,
        table_name: str,
        schema_name: str = "public",
        content_column: str = "page_content",
        metadata_columns: list[Column] = [],
        metadata_json_column: str = "langchain_metadata",
        store_metadata: bool = True,
    ) -> None:
        query = f"""CREATE TABLE "{schema_name}"."{table_name}"(
            {content_column} TEXT NOT NULL
            """
        for column in metadata_columns:
            nullable = "NOT NULL" if not column.nullable else ""
            query += f',\n"{column.name}" {column.data_type} {nullable}'
        metadata_json_column = metadata_json_column or "langchain_metadata"
        if store_metadata:
            query += f',\n"{metadata_json_column}" JSON'
        query += "\n);"

        async with self._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    async def ainit_document_table(
        self,
        table_name: str,
        schema_name: str = "public",
        content_column: str = "page_content",
        metadata_columns: list[Column] = [],
        metadata_json_column: str = "langchain_metadata",
        store_metadata: bool = True,
    ) -> None:
        """
        Create a table for saving of langchain documents.

        Args:
            table_name (str): The PgSQL database table name.
            schema_name (str): The schema name to store PgSQL database table.
                Default: "public".
            content_column (str): Name of the column to store document content.
                Default: "page_content".
            metadata_columns (list[sqlalchemy.Column]): A list of SQLAlchemy Columns
                to create for custom metadata. Optional.
            metadata_json_column (str): The column to store extra metadata in JSON format.
                Default: "langchain_metadata". Optional.
            store_metadata (bool): Whether to store extra metadata in a metadata column
                if not described in 'metadata' field list (Default: True).

        Raises:
            :class:`DuplicateTableError <asyncpg.exceptions.DuplicateTableError>`: if table already exists.
        """
        await self._run_as_async(
            self._ainit_document_table(
                table_name,
                schema_name,
                content_column,
                metadata_columns,
                metadata_json_column,
                store_metadata,
            )
        )

    def init_document_table(
        self,
        table_name: str,
        schema_name: str = "public",
        content_column: str = "page_content",
        metadata_columns: list[Column] = [],
        metadata_json_column: str = "langchain_metadata",
        store_metadata: bool = True,
    ) -> None:
        """
        Create a table for saving of langchain documents.

        Args:
            table_name (str): The PgSQL database table name.
            schema_name (str): The schema name to store PgSQL database table.
                Default: "public".
            content_column (str): Name of the column to store document content.
                Default: "page_content".
            metadata_columns (list[sqlalchemy.Column]): A list of SQLAlchemy Columns
                to create for custom metadata. Optional.
            metadata_json_column (str): The column to store extra metadata in JSON format.
                Default: "langchain_metadata". Optional.
            store_metadata (bool): Whether to store extra metadata in a metadata column
                if not described in 'metadata' field list (Default: True).

        Raises:
            :class:`DuplicateTableError <asyncpg.exceptions.DuplicateTableError>`: if table already exists.
        """
        self._run_as_sync(
            self._ainit_document_table(
                table_name,
                schema_name,
                content_column,
                metadata_columns,
                metadata_json_column,
                store_metadata,
            )
        )

    async def _ainit_checkpoint_table(
        self, table_name: str = CHECKPOINTS_TABLE, schema_name: str = "public"
    ) -> None:
        """
        Create PgSQL tables to save checkpoints.

        Args:
            schema_name (str): The schema name to store the checkpoint tables.
                Default: "public".
            table_name (str): The PgSQL database table name.
                Default: "checkpoints".

        Returns:
            None
        """
        create_checkpoints_table = f"""CREATE TABLE "{schema_name}"."{table_name}"(
            thread_id TEXT NOT NULL,
            checkpoint_ns TEXT NOT NULL DEFAULT '',
            checkpoint_id TEXT NOT NULL,
            parent_checkpoint_id TEXT,
            type TEXT,
            checkpoint BYTEA,
            metadata BYTEA,
            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
        );"""

        create_checkpoint_writes_table = f"""CREATE TABLE "{schema_name}"."{table_name + "_writes"}"(
            thread_id TEXT NOT NULL,
            checkpoint_ns TEXT NOT NULL DEFAULT '',
            checkpoint_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            idx INTEGER NOT NULL,
            channel TEXT NOT NULL,
            type TEXT,
            blob BYTEA NOT NULL,
            task_path TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
        );"""

        async with self._pool.connect() as conn:
            await conn.execute(text(create_checkpoints_table))
            await conn.execute(text(create_checkpoint_writes_table))
            await conn.commit()

    async def ainit_checkpoint_table(
        self, table_name: str = CHECKPOINTS_TABLE, schema_name: str = "public"
    ) -> None:
        """Create an PgSQL table to save checkpoint messages.

        Args:
            schema_name (str): The schema name to store checkpoint tables.
                Default: "public".
            table_name (str): The PgSQL database table name.
                Default: "checkpoints".

        Returns:
            None
        """
        await self._run_as_async(
            self._ainit_checkpoint_table(
                table_name,
                schema_name,
            )
        )

    def init_checkpoint_table(
        self, table_name: str = CHECKPOINTS_TABLE, schema_name: str = "public"
    ) -> None:
        """Create Cloud SQL tables to store checkpoints.

        Args:
            schema_name (str): The schema name to store checkpoint tables.
                Default: "public".
            table_name (str): The PgSQL database table name.
                Default: "checkpoints".

        Returns:
            None
        """
        self._run_as_sync(self._ainit_checkpoint_table(table_name, schema_name))

    async def _aload_table_schema(
        self,
        table_name: str,
        schema_name: str = "public",
    ) -> Table:
        """
        Load table schema from existing table in PgSQL database.
        Returns:
            (sqlalchemy.Table): The loaded table.
        """
        metadata = MetaData()
        async with self._pool.connect() as conn:
            try:
                await conn.run_sync(
                    metadata.reflect, schema=schema_name, only=[table_name]
                )
            except InvalidRequestError as e:
                raise ValueError(
                    f'Table, "{schema_name}"."{table_name}", does not exist: ' + str(e)
                )

        table = Table(table_name, metadata, schema=schema_name)
        # Extract the schema information
        schema = []
        for column in table.columns:
            schema.append(
                {
                    "name": column.name,
                    "type": column.type.python_type,
                    "max_length": getattr(column.type, "length", None),
                    "nullable": not column.nullable,
                }
            )

        return metadata.tables[f"{schema_name}.{table_name}"]
