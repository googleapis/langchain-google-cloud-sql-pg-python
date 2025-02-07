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
from typing import Any, Optional, Sequence, Tuple, cast, AsyncIterator

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.types import TASKS
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from .engine import CHECKPOINT_WRITES_TABLE, CHECKPOINTS_TABLE, PostgresEngine

MetadataInput = Optional[dict[str, Any]]

# Select SQL used in `alist` method
SELECT = f"""
select
    thread_id,
    checkpoint,
    checkpoint_ns,
    checkpoint_id,
    parent_checkpoint_id,
    metadata,
    (
        select array_agg(array[bl.channel::bytea, bl.type::bytea, bl.blob])
        from jsonb_each_text(checkpoint -> 'channel_versions')
    ) as channel_values,
    (
        select
        array_agg(array[cw.task_id::text::bytea, cw.channel::bytea, cw.type::bytea, cw.blob] order by cw.task_id, cw.idx)
        from checkpoint_writes cw
        where cw.thread_id = checkpoints.thread_id
            and cw.checkpoint_ns = checkpoints.checkpoint_ns
            and cw.checkpoint_id = checkpoints.checkpoint_id
    ) as pending_writes,
    (
        select array_agg(array[cw.type::bytea, cw.blob] order by cw.task_path, cw.task_id, cw.idx)
        from checkpoint_writes cw
        where cw.thread_id = checkpoints.thread_id
            and cw.checkpoint_ns = checkpoints.checkpoint_ns
            and cw.checkpoint_id = checkpoints.parent_checkpoint_id
            and cw.channel = '{TASKS}'
    ) as pending_sends
from checkpoints
"""


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

        checkpoint_writes_table_schema = await engine._aload_table_schema(
            CHECKPOINT_WRITES_TABLE, schema_name
        )
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
        return {**checkpoint, "pending_sends": []}

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

    def _dump_writes(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        task_path: str,
        writes: Sequence[tuple[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
                "task_id": task_id,
                "task_path": task_path,
                "idx": WRITES_IDX_MAP.get(channel, idx),
                "channel": channel,
                "type": self.serde.dumps_typed(value)[0],
                "blob": self.serde.dumps_typed(value)[1],
            }
            for idx, (channel, value) in enumerate(writes)
        ]

    def _load_blobs(
        self, blob_values: list[tuple[bytes, bytes, bytes]]
    ) -> dict[str, Any]:
        if not blob_values:
            return {}
        return {
            k.decode(): self.serde.loads_typed((t.decode(), v))
            for k, t, v in blob_values
            if t.decode() != "empty"
        }

    def _load_checkpoint(
        self,
        checkpoint: dict[str, Any],
        channel_values: list[tuple[bytes, bytes, bytes]],
        pending_sends: list[tuple[bytes, bytes]],
    ) -> Checkpoint:
        return Checkpoint(
            v=checkpoint["v"],
            ts=checkpoint["ts"],
            id=checkpoint["id"],
            channel_values=self._load_blobs(channel_values),
            channel_versions=checkpoint["channel_versions"].copy(),
            versions_seen={k: v.copy() for k, v in checkpoint["versions_seen"].items()},
            pending_sends=[
                self.serde.loads_typed((c.decode(), b)) for c, b in pending_sends or []
            ],
        )

    def _load_metadata(self, metadata: str) -> CheckpointMetadata:
        return self.jsonplus_serde.loads(self.jsonplus_serde.dumps(metadata))

    def _load_writes(
        self, writes: list[tuple[bytes, bytes, bytes, bytes]]
    ) -> list[tuple[str, str, Any]]:
        return (
            [
                (
                    tid.decode(),
                    channel.decode(),
                    self.serde.loads_typed((t.decode(), v)),
                )
                for tid, channel, t, v in writes
            ]
            if writes
            else []
        )

    def _search_where(
        self,
        config: Optional[RunnableConfig],
        filter: MetadataInput,
        before: Optional[RunnableConfig] = None,
    ) -> tuple[str, list[Any]]:
        """Return WHERE clause predicates for alist() given config, filter, before.
        This method returns a tuple of a string and a tuple of values. The string
        is the parametered WHERE clause predicate (including the WHERE keyword):
        "WHERE column1 = $1 AND column2 IS $2". The list of values contains the
        values for each of the corresponding parameters.
        """
        wheres = []
        param_values = []

        # construct predicate for config filter
        if config:
            wheres.append("thread_id = %s ")
            param_values.append(config["configurable"]["thread_id"])
            checkpoint_ns = config["configurable"].get("checkpoint_ns")
            if checkpoint_ns is not None:
                wheres.append("checkpoint_ns = %s")
                param_values.append(checkpoint_ns)

            if checkpoint_id := get_checkpoint_id(config):
                wheres.append("checkpoint_id = %s ")
                param_values.append(checkpoint_id)

        # construct predicate for metadata filter
        if filter:
            wheres.append("metadata @> %s ")
            param_values.append(json.dumps(filter))

        # construct predicate for `before`
        if before is not None:
            wheres.append("checkpoint_id < %s ")
            param_values.append(get_checkpoint_id(before))

        return (
            "WHERE " + " AND ".join(wheres) if wheres else "",
            param_values,
        )

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
        upsert = f"""INSERT INTO "{self.schema_name}".{CHECKPOINT_WRITES_TABLE}(thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob)
                    VALUES (:thread_id, :checkpoint_ns, :checkpoint_id, :task_id, :idx, :channel, :type, :blob)
                    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO UPDATE SET
                    channel = EXCLUDED.channel,
                        type = EXCLUDED.type,
                        blob = EXCLUDED.blob;
                """
        insert = f"""INSERT INTO "{self.schema_name}".{CHECKPOINT_WRITES_TABLE}(thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob)
                    VALUES (:thread_id, :checkpoint_ns, :checkpoint_id, :task_id, :idx, :channel, :type, :blob)
                    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO NOTHING
                """
        query = upsert if all(w[0] in WRITES_IDX_MAP for w in writes) else insert

        params = self._dump_writes(
            config["configurable"]["thread_id"],
            config["configurable"]["checkpoint_ns"],
            config["configurable"]["checkpoint_id"],
            task_id,
            task_path,
            writes,
        )

        async with self.pool.connect() as conn:
            await conn.execute(
                text(query),
                params,
            )
            await conn.commit()

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Asynchronously list checkpoints that match the given criteria.
        Args:
            config (Optional[RunnableConfig]): Base configuration for filtering checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata.
            before (Optional[RunnableConfig]): List checkpoints created before this configuration.
            limit (Optional[int]): Maximum number of checkpoints to return.
        Returns:
            AsyncIterator[CheckpointTuple]: Async iterator of matching checkpoint tuples.
        """

        where, args = self._search_where(config, filter, before)
        query = SELECT + where + " ORDER BY checkpoint_id DESC"
        if limit:
            query += f" LIMIT {limit}"

        async with self.pool.connect() as conn:
            result = await conn.execute(text(query), args)
            rows = result.fetchall()  # Getting all the results

            for row in rows:
                value = dict(row._mapping)
                yield CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": value["thread_id"],
                            "checkpoint_ns": value["checkpoint_ns"],
                            "checkpoint_id": value["checkpoint_id"],
                        }
                    },
                    checkpoint=self._load_checkpoint(
                        value["checkpoint"],
                        value["channel_values"],
                        value["pending_sends"],
                    ),
                    metadata=self._load_metadata(value["metadata"]),
                    parent_config=(
                        {
                            "configurable": {
                                "thread_id": value["thread_id"],
                                "checkpoint_ns": value["checkpoint_ns"],
                                "checkpoint_id": value["parent_checkpoint_id"],
                            }
                        }
                        if value["parent_checkpoint_id"]
                        else None
                    ),
                    pending_writes=self._load_writes(value["pending_writes"]),
                )
