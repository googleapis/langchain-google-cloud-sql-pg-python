from __future__ import annotations

import asyncio
import json
from threading import Thread

import aiohttp
import google.auth
from google.cloud.sql.connector import Connector
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from typing import Any


async def _get_IAM_user(credentials):
    """Get user/service account name"""
    request = google.auth.transport.requests.Request()
    credentials.refresh(request)

    url = f"https://oauth2.googleapis.com/tokeninfo?access_token={credentials.token}"
    async with aiohttp.ClientSession() as client:
        response = await client.get(url)
        response = await response.text()
        response = json.loads(response)
        email = response['email']
        if ".gserviceaccount.com" in email:
            email = email.replace(".gserviceaccount.com", "")

        return email


class PgSQLEngine:
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
            engine=None
    ):
        self.project_id = project_id
        self.region = region
        self.instance = instance
        self.database = database
        self.engine = engine

        self._loop = asyncio.new_event_loop()
        thread = Thread(target=self._loop.run_forever, daemon=True)
        thread.start()
        pool_object = asyncio.run_coroutine_threadsafe(self._engine(), self._loop)
        self._pool = pool_object.result()

    @classmethod
    def from_instance(
            cls,
            region: str,
            instance: str,
            database: str,
            project_id: str = None,
    ):

        """Create PgSQLEngine connection to the postgres database in the CloudSQL instance.

        Args:
            region (str): CloudSQL instance region.
            instance (str): CloudSQL instance name.
            database (str): CloudSQL instance database name.
            project_id (str): GCP project ID. Defaults to None

        Returns:
            PgSQLEngine containing the asyncpg connection pool.
        """
        return cls(project_id=project_id, region=region, instance=instance, database=database)

    @classmethod
    def from_engine(
            cls,
            engine: AsyncEngine
    ):

        return cls(engine=engine)

    async def _engine(self) -> AsyncEngine:

        if self.engine is not None:
            return self.engine

        credentials, _ = google.auth.default(scopes=['email', 'https://www.googleapis.com/auth/cloud-platform'])

        if self.project_id is None:
            self.project_id = _

        # noinspection PyCompatibility
        async def get_conn():
            async with Connector(loop=asyncio.get_running_loop()) as connector:
                conn = await connector.connect_async(
                    f"{self.project_id}:{self.region}:{self.instance}",
                    "asyncpg",
                    user="postgres",
                    password="test",
                    enable_iam_auth=False,
                    db=self.database,
                )
                conn.transaction(readonly=True)

            return conn

        pool = create_async_engine(
            "postgresql+asyncpg://",
            async_creator=get_conn,
        )

        return pool

    async def _aexecute_fetch(
            self,
            query
    ) -> Any:

        async with self._pool.connect() as conn:
            result = (await conn.execute(text(query)))
            result_map = result.mappings()
            result_fetch = result_map.fetchall()

        return result_fetch

    async def _aexecute_update(
            self,
            query,
            additional=None
    ) -> None:

        async with self._pool.connect() as conn:
            (await conn.execute(text(query), additional)).mappings()
            await conn.commit()