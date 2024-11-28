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


class AsyncAlloyDBPostgresSaver(BasePostgresSaver):
    lock: asyncio.Lock
    """Checkpoint stored in an AlloyDB for PostgreSQL database."""
    
    __create_key = object()
    
    def __init__(
        self,
        key: object,
        pool: AlloyDBEngine,
        serde: Optional[SerializerProtocol] = None
    ) -> None:
        if key != AsyncAlloyDBPostgresSaver.__create_key:
            raise Exception(
                "only create class through 'create' or 'create_sync' methods"
            )
        self.pool = pool
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()
    
    async def alist(self):
        pass
    
    async def aget_tuple(self):
        pass
    
    async def aput(self):
        pass
    
    async def aput_writes(self):
        pass
    async def setup(self):
        pass
    
    def list(self):
        pass
    
    def get_tuple(self):
        pass
    
    def put(self):
        pass
    
    def put_writes(self):
        pass
    