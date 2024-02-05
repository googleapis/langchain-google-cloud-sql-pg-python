from __future__ import annotations

import asyncio
import json
import time
from threading import Thread
from typing import AnyStr

import aiohttp
import google.auth
from google.cloud.sql.connector import Connector
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from pgvector.asyncpg import register_vector

# import sqlalchemy
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


async def _get_IAM_user(
    credentials: google.auth.credentials.Credentials,
) -> str:
    """Get user/service account name"""
    request = google.auth.transport.requests.Request()
    credentials.refresh(request)

    url = f"https://oauth2.googleapis.com/tokeninfo?access_token={credentials.token}"
    async with aiohttp.ClientSession() as client:
        response = await client.get(url)
        response = json.loads(await response.text())
        email = response["email"]
        if ".gserviceaccount.com" in email:
            email = email.replace(".gserviceaccount.com", "")

        return email


class PostgreSQLEngine:
    """Creating a connection to the CloudSQL instance
    To use, you need the following packages installed:
        cloud-sql-python-connector[asyncpg]
    """

    def __init__(
        self,
        project_id=None,
        region=None,
        instance=None,
        database=None,
        engine=None,
    ):
        self.project_id = project_id
        self.region = region
        self.instance = instance
        self.database = database
        self.engine = engine
        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        pool_object = asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(self.async_func(), self._loop),
            loop=self._loop,
        )
        time.sleep(1)
        self._pool = pool_object.result()

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
        )

    @classmethod
    def from_engine(cls, engine: AsyncEngine) -> PostgreSQLEngine:
        return cls(engine=engine)

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
                    user=await _get_IAM_user(credentials),
                    enable_iam_auth=True,
                    db=self.database,
                )

            await register_vector(conn)
            return conn

        pool = create_async_engine(
            "postgresql+asyncpg://",
            async_creator=get_conn,
        )

        return pool

    async def _aexecute_fetch(self, query) -> Any:
        async with self._pool.connect() as conn:
            result = await conn.execute(text(query))
            result_map = result.mappings()
            result_fetch = result_map.fetchall()

        return result_fetch

    async def _aexecute_update(self, query, additional=None) -> None:
        async with self._pool.connect() as conn:
            result = await conn.execute(text(query), additional)
            result = result.mappings()
            await conn.commit()
