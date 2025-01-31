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

import json
from contextlib import asynccontextmanager
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from .engine import CHECKPOINT_WRITES_TABLE, CHECKPOINTS_TABLE, PostgresEngine


class AsyncPostgresSaver(BaseCheckpointSaver[str]):
    """Checkpoint storage for a PostgreSQL database."""

    __create_key = object()

    jsonplus_serde = JsonPlusSerializer()

    def __init__(
        self,
        key: object,
        pool: AsyncEngine,
        schema_name: str = "public",
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        """
        Initializes an AsyncPostgresSaver instance.

        Args:
            key (object): Internal key to restrict instantiation.
            pool (AsyncEngine): The database connection pool.
            schema_name (str, optional): The schema where the checkpoint tables reside. Defaults to "public".
            serde (Optional[SerializerProtocol], optional): Serializer for encoding/decoding checkpoints. Defaults to None.
        """
        super().__init__(serde=serde)
        if key != AsyncPostgresSaver.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync' methods"
            )
        self.pool = pool
        self.schema_name = schema_name

    @classmethod
    async def create(
        cls,
        engine: PostgresEngine,
        schema_name: str = "public",
        serde: Optional[SerializerProtocol] = None,
    ) -> "AsyncPostgresSaver":
        """
        Creates a new AsyncPostgresSaver instance.

        Args:
            engine (PostgresEngine): The PostgreSQL engine to use.
            schema_name (str, optional): The schema name where the table is located. Defaults to "public".
            serde (Optional[SerializerProtocol], optional): Serializer for encoding/decoding checkpoints. Defaults to None.

        Raises:
            IndexError: If the table does not contain the required schema.

        Returns:
            AsyncPostgresSaver: A newly created instance.
        """

        checkpoints_table_schema = await engine._aload_table_schema(
            CHECKPOINTS_TABLE, schema_name
        )
        checkpoints_column_names = checkpoints_table_schema.columns.keys()

        checkpoints_required_columns = [
            "thread_id",
            "checkpoint_ns",
            "checkpoint_id",
            "parent_checkpoint_id",
            "type",
            "checkpoint",
            "metadata",
        ]

        if not (
            all(x in checkpoints_column_names for x in checkpoints_required_columns)
        ):
            raise IndexError(
                f"Table checkpoints.'{schema_name}' has incorrect schema. Got "
                f"column names '{checkpoints_column_names}' but required column names "
                f"'{checkpoints_required_columns}'.\nPlease create table with the following schema:"
                f"\nCREATE TABLE {schema_name}.checkpoints ("
                "\n    thread_id TEXT NOT NULL,"
                "\n    checkpoint_ns TEXT NOT NULL,"
                "\n    checkpoint_id TEXT NOT NULL,"
                "\n    parent_checkpoint_id TEXT,"
                "\n    type TEXT,"
                "\n    checkpoint JSONB NOT NULL,"
                "\n    metadata JSONB NOT NULL"
                "\n);"
            )

        checkpoint_writes_table_schema = await engine._aload_table_schema(CHECKPOINT_WRITES_TABLE, schema_name)
        checkpoint_writes_column_names = checkpoint_writes_table_schema.columns.keys()

        checkpoint_writes_columns = [
            "thread_id",
            "checkpoint_ns",
            "checkpoint_id",
            "task_id",
            "idx",
            "channel",
            "type",
            "blob",
        ]

        if not (
                all(x in checkpoint_writes_column_names for x in checkpoint_writes_columns)
        ):
            raise IndexError(
                f"Table checkpoint_writes.'{schema_name}' has incorrect schema. Got "
                f"column names '{checkpoint_writes_column_names}' but required column names "
                f"'{checkpoint_writes_columns}'.\nPlease create table with following schema:"
                f"\nCREATE TABLE {schema_name}.checkpoint_writes ("
                "\n    thread_id TEXT NOT NULL,"
                "\n    checkpoint_ns TEXT NOT NULL,"
                "\n    checkpoint_id TEXT NOT NULL,"
                "\n    task_id TEXT NOT NULL,"
                "\n    idx INT NOT NULL,"
                "\n    channel TEXT NOT NULL,"
                "\n    type TEXT,"
                "\n    blob JSONB NOT NULL"
                "\n);"
            )

        return cls(cls.__create_key, engine._pool, schema_name, serde)

    def _dump_checkpoint(self, checkpoint: Checkpoint) -> str:
        """
        Serializes a checkpoint into a JSON string.

        Args:
            checkpoint (Checkpoint): The checkpoint to serialize.

        Returns:
            str: The serialized checkpoint as a JSON string.
        """
        checkpoint["pending_sends"] = []
        return json.dumps(checkpoint)

    def _dump_metadata(self, metadata: CheckpointMetadata) -> str:
        """
        Serializes checkpoint metadata into a JSON string.

        Args:
            metadata (CheckpointMetadata): The metadata to serialize.

        Returns:
            str: The serialized metadata as a JSON string.
        """
        serialized_metadata = self.jsonplus_serde.dumps(metadata)
        return serialized_metadata.decode().replace("\\u0000", "")

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """
        Asynchronously stores a checkpoint with its configuration and metadata.

        Args:
            config (RunnableConfig): Configuration for the checkpoint.
            checkpoint (Checkpoint): The checkpoint to store.
            metadata (CheckpointMetadata): Additional metadata for the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        configurable = config["configurable"].copy()
        thread_id = configurable.pop("thread_id")
        checkpoint_ns = configurable.pop("checkpoint_ns")
        checkpoint_id = configurable.pop(
            "checkpoint_id", configurable.pop("thread_ts", None)
        )

        copy = checkpoint.copy()
        next_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

        query = f"""INSERT INTO "{self.schema_name}".{CHECKPOINTS_TABLE}(thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata)
                    VALUES (:thread_id, :checkpoint_ns, :checkpoint_id, :parent_checkpoint_id, :checkpoint, :metadata)
                    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id)
                    DO UPDATE SET
                        checkpoint = EXCLUDED.checkpoint,
                        metadata = EXCLUDED.metadata;"""

        async with self.pool.connect() as conn:
            await conn.execute(
                text(query),
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint["id"],
                    "parent_checkpoint_id": checkpoint_id,
                    "checkpoint": self._dump_checkpoint(copy),
                    "metadata": self._dump_metadata(metadata),
                },
            )
            await conn.commit()

        return next_config
