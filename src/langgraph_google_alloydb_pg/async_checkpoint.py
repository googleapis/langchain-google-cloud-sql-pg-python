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

import asyncio
from contextlib import asynccontextmanager

import json
from typing import List, Sequence, Any, AsyncIterator, Iterator, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.checkpoint.postgres.base import BasePostgresSaver
from langgraph.checkpoint.serde.base import SerializerProtocol

from langchain_google_alloydb_pg import AlloyDBEngine

class Engine(AlloyDBEngine, BasePostgresSaver):
    """A class for managing connections to an AlloyDB for Postgres database."""
    __create_key = object()
    def __init__(
        self,
        key: object,
        pool: AsyncEngine
    ) -> None:
        AlloyDBEngine.__init__(self, key, pool)
        
        if key != AlloyDBEngine.__create_key:
            raise Exception(
                "Only create class through 'setup' method!"
            )
    async def setup(self) -> None:
        """Set up the checkpoint database asynchronously.

        This method creates the necessary tables in the Postgres database if they don't
        already exist and runs database migrations. It MUST be called directly by the user
        the first time checkpointer is used.
        """
        async with self._pool.connect() as conn:
            create_table_query = self.MIGRATIONS[0]
            await conn.execute(text(create_table_query))
            row = await conn.fetchrow(
                "SELECT v FROM checkpoint_migrations ORDER BY v DESC LIMIT 1"
            )
            if row is None:
                version = -1
            else:
                version = row["v"]
            for v, migration in zip(
                range(version + 1, len(self.MIGRATIONS)),
                self.MIGRATIONS[version + 1:]
            ):
                await conn.execute(migration)
                query = f"INSERT INTO checkpoint_migrations (v) VALUES ({v})"
                await conn.execute(text(query))
            await conn.commit()


class AsyncAlloyDBPostgresSaver(BasePostgresSaver):
    lock: asyncio.Lock
    """Checkpoint stored in an AlloyDB for PostgreSQL database."""
    
    __create_key = object()
    
    def __init__(
        self,
        key: object,
        serde: Optional[SerializerProtocol] = None
    ) -> None:
        super().__init__(serde=serde)
        if key != AsyncAlloyDBPostgresSaver.__create_key:
            raise Exception(
                "only create class through 'create' or 'create_sync' methods"
            )
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()
        
    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from the database asynchronously.

        This method retrieves a list of checkpoint tuples from the Postgres database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config (Optional[RunnableConfig]): Base configuration for filtering checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Yields:
            AsyncIterator[CheckpointTuple]: An asynchronous iterator of matching checkpoint tuples.
        """
        pass
    
    async def aget_tuple(self):
        pass
    
    async def aput(self):
        pass
    
    async def aput_writes(self):
        pass
    
    def list(self):
        pass
    
    def get_tuple(self):
        pass
    
    def put(self):
        pass
    
    def put_writes(self):
        pass
    