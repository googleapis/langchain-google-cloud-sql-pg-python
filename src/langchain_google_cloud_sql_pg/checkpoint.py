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

from typing import Any, AsyncIterator, Iterator, Optional, Sequence, Tuple

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.serde.base import SerializerProtocol

from .async_checkpoint import AsyncPostgresSaver
from .engine import CHECKPOINTS_TABLE, PostgresEngine


class PostgresSaver(BaseCheckpointSaver[str]):
    """Checkpoint stored in PgSQL"""

    __create_key = object()

    def __init__(
        self,
        key: object,
        engine: PostgresEngine,
        checkpoint: AsyncPostgresSaver,
        table_name: str = CHECKPOINTS_TABLE,
        schema_name: str = "public",
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        super().__init__(serde=serde)
        if key != PostgresSaver.__create_key:
            raise Exception(
                "only create class through 'create' or 'create_sync' methods"
            )
        self._engine = engine
        self.__checkpoint = checkpoint

    @classmethod
    async def create(
        cls,
        engine: PostgresEngine,
        table_name: str = CHECKPOINTS_TABLE,
        schema_name: str = "public",
        serde: Optional[SerializerProtocol] = None,
    ) -> "PostgresSaver":
        """Create a new PostgresSaver instance.
        Args:
            engine (PostgresEngine): PgSQL engine to use.
            table_name (str): Table name that stores the checkpoints (default: "checkpoints").
            schema_name (str): The schema name where the table is located (default: "public").
            serde (SerializerProtocol): Serializer for encoding/decoding checkpoints (default: None).
        Raises:
            IndexError: If the table provided does not contain required schema.
        Returns:
            PostgresSaver: A newly created instance of PostgresSaver.
        """
        coro = AsyncPostgresSaver.create(engine, table_name, schema_name, serde)
        checkpoint = await engine._run_as_async(coro)
        return cls(cls.__create_key, engine, checkpoint)

    @classmethod
    def create_sync(
        cls,
        engine: PostgresEngine,
        table_name: str = CHECKPOINTS_TABLE,
        schema_name: str = "public",
        serde: Optional[SerializerProtocol] = None,
    ) -> "PostgresSaver":
        """Create a new PostgresSaver instance.
        Args:
            engine (PostgresEngine): PgSQL engine to use.
            table_name (str): Table name that stores the checkpoints (default: "checkpoints").
            schema_name (str): The schema name where the table is located (default: "public").
            serde (SerializerProtocol): Serializer for encoding/decoding checkpoints (default: None).
        Raises:
            IndexError: If the table provided does not contain required schema.
        Returns:
            PostgresSaver: A newly created instance of PostgresSaver.
        """
        coro = AsyncPostgresSaver.create(engine, table_name, schema_name, serde)
        checkpoint = engine._run_as_sync(coro)
        return cls(cls.__create_key, engine, checkpoint)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Asynchronously list checkpoints that match the given criteria
        Args:
            config (Optional[RunnableConfig]): Base configuration for filtering checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata.
            before (Optional[RunnableConfig]): List checkpoints created before this configuration.
            limit (Optional[int]): Maximum number of checkpoints to return.
        Returns:
            AsyncIterator[CheckpointTuple]: Async iterator of matching checkpoint tuples.
        """
        iterator = self.__checkpoint.alist(
            config=config, filter=filter, before=before, limit=limit
        )
        while True:
            try:
                result = await self._engine._run_as_async(iterator.__anext__())
                yield result
            except StopAsyncIteration:
                break

    def list(
        self,
        config: Optional[RunnableConfig],
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from PgSQL
        Args:
            config (RunnableConfig): The config to use for listing the checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata. Defaults to None.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit (Optional[int]): The maximum number of checkpoints to return. Defaults to None.
        Yields:
            Iterator[CheckpointTuple]: An iterator of checkpoint tuples.
        """

        iterator: AsyncIterator[CheckpointTuple] = self.__checkpoint.alist(
            config=config, filter=filter, before=before, limit=limit
        )
        while True:
            try:
                result = self._engine._run_as_sync(iterator.__anext__())
                yield result
            except StopAsyncIteration:
                break

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Asynchronously fetch a checkpoint tuple using the given configuration.
        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.
        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        return await self._engine._run_as_async(self.__checkpoint.aget_tuple(config))

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from PgSQL.
        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.
        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        return self._engine._run_as_sync(self.__checkpoint.aget_tuple(config))

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Asynchronously store a checkpoint with its configuration and metadata.
        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.
        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        return await self._engine._run_as_async(
            self.__checkpoint.aput(config, checkpoint, metadata, new_versions)
        )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database.
        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.
        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        return self._engine._run_as_sync(
            self.__checkpoint.aput(config, checkpoint, metadata, new_versions)
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Asynchronously store intermediate writes linked to a checkpoint.
        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (List[Tuple[str, Any]]): List of writes to store.
            task_id (str): Identifier for the task creating the writes.
            task_path (str): Path of the task creating the writes.
        Returns:
                None
        """
        await self._engine._run_as_async(
            self.__checkpoint.aput_writes(config, writes, task_id, task_path)
        )

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint.
        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (List[Tuple[str, Any]]): List of writes to store.
            task_id (str): Identifier for the task creating the writes.
            task_path (str): Path of the task creating the writes.
        Returns:
            None
        """
        self._engine._run_as_sync(
            self.__checkpoint.aput_writes(config, writes, task_id, task_path)
        )
