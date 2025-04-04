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
from typing import Any, AsyncIterator, Iterator, Optional, Sequence, Tuple

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
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.serde.types import TASKS
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from .engine import CHECKPOINTS_TABLE, PostgresEngine

MetadataInput = Optional[dict[str, Any]]

checkpoints_columns = [
    "thread_id",
    "checkpoint_ns",
    "checkpoint_id",
    "parent_checkpoint_id",
    "type",
    "checkpoint",
    "metadata",
]

writes_columns = [
    "thread_id",
    "checkpoint_ns",
    "checkpoint_id",
    "task_id",
    "idx",
    "channel",
    "type",
    "blob",
]


class AsyncPostgresSaver(BaseCheckpointSaver[str]):
    """Checkpoint stored in PgSQL"""

    __create_key = object()

    jsonplus_serde = JsonPlusSerializer()

    def __init__(
        self,
        key: object,
        pool: AsyncEngine,
        table_name: str = CHECKPOINTS_TABLE,
        schema_name: str = "public",
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        super().__init__(serde=serde)
        if key != AsyncPostgresSaver.__create_key:
            raise Exception(
                "only create class through 'create' or 'create_sync' methods"
            )
        self.pool = pool
        self.table_name = table_name
        self.table_name_writes = f"{table_name}_writes"
        self.schema_name = schema_name

    @classmethod
    async def create(
        cls,
        engine: PostgresEngine,
        table_name: str = CHECKPOINTS_TABLE,
        schema_name: str = "public",
        serde: Optional[SerializerProtocol] = None,
    ) -> "AsyncPostgresSaver":
        """Create a new AsyncPostgresSaver instance.

        Args:
            engine (PostgresEngine): PostgresEngine engine to use.
            schema_name (str): The schema name where the table is located (default: "public").
            serde (SerializerProtocol): Serializer for encoding/decoding checkpoints (default: None).
            table_name (str): Custom table name to use (default: CHECKPOINTS_TABLE).

        Raises:
            IndexError: If the table provided does not contain required schema.

        Returns:
            AsyncPostgresSaver: A newly created instance of AsyncPostgresSaver.
        """

        checkpoints_table_schema = await engine._aload_table_schema(
            table_name, schema_name
        )
        checkpoints_column_names = checkpoints_table_schema.columns.keys()

        if not (all(x in checkpoints_column_names for x in checkpoints_columns)):
            raise IndexError(
                f"Table checkpoints.'{schema_name}' has incorrect schema. Got "
                f"column names '{checkpoints_column_names}' but required column names "
                f"'{checkpoints_columns}'.\nPlease create table with following schema:"
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
            f"{table_name}_writes", schema_name
        )
        checkpoint_writes_column_names = checkpoint_writes_table_schema.columns.keys()

        if not (all(x in checkpoint_writes_column_names for x in writes_columns)):
            raise IndexError(
                f"Table checkpoint_writes.'{schema_name}' has incorrect schema. Got "
                f"column names '{checkpoint_writes_column_names}' but required column names "
                f"'{writes_columns}'.\nPlease create table with following schema:"
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
        return cls(cls.__create_key, engine._pool, table_name, schema_name, serde)

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
    ) -> tuple[str, dict[str, Any]]:
        """Return WHERE clause predicates for alist() given config, filter, before.

        This method returns a tuple of a string and a tuple of values. The string
        is the parameterized WHERE clause predicate (including the WHERE keyword):
        "WHERE column1 = $1 AND column2 IS $2". The list of values contains the
        values for each of the corresponding parameters.
        """
        wheres = []
        param_values = {}

        # construct predicate for config filter
        if config:
            wheres.append("thread_id = :thread_id")
            param_values.update({"thread_id": config["configurable"]["thread_id"]})
            checkpoint_ns = config["configurable"].get("checkpoint_ns")
            if checkpoint_ns is not None:
                wheres.append("checkpoint_ns = :checkpoint_ns")
                param_values.update({"checkpoint_ns": checkpoint_ns})

            if checkpoint_id := get_checkpoint_id(config):
                wheres.append("checkpoint_id = :checkpoint_id")
                param_values.update({"checkpoint_id": checkpoint_id})

        # construct predicate for metadata filter
        if filter:
            wheres.append("convert_from(metadata,'UTF8')::jsonb @> :metadata ")
            param_values.update({"metadata": f"{json.dumps(filter)}"})

        # construct predicate for `before`
        if before is not None:
            wheres.append("checkpoint_id < :checkpoint_id")
            param_values.update({"checkpoint_id": get_checkpoint_id(before)})

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
        """Asynchronously store a checkpoint with its configuration and metadata.

        Args:
            config (RunnableConfig): Configuration for the checkpoint.
            checkpoint (Checkpoint): The checkpoint to store.
            metadata (CheckpointMetadata): Additional metadata for the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        configurable = config["configurable"]
        thread_id = configurable.get("thread_id")
        checkpoint_ns = configurable.get("checkpoint_ns")
        checkpoint_id = configurable.get(
            "checkpoint_id", configurable.get("thread_ts", None)
        )

        next_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

        query = f"""INSERT INTO "{self.schema_name}"."{self.table_name}" (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata)
                    VALUES (:thread_id, :checkpoint_ns, :checkpoint_id, :parent_checkpoint_id, :type, :checkpoint, :metadata)
                    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id)
                    DO UPDATE SET
                        checkpoint = EXCLUDED.checkpoint,
                        metadata = EXCLUDED.metadata;
            """

        async with self.pool.connect() as conn:
            type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
            serialized_metadata = self.jsonplus_serde.dumps(metadata)
            await conn.execute(
                text(query),
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint["id"],
                    "parent_checkpoint_id": checkpoint_id,
                    "type": type_,
                    "checkpoint": serialized_checkpoint,
                    "metadata": serialized_metadata,
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
        upsert = f"""INSERT INTO "{self.schema_name}"."{self.table_name_writes}"(thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob)
                    VALUES (:thread_id, :checkpoint_ns, :checkpoint_id, :task_id, :idx, :channel, :type, :blob)
                    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO UPDATE SET
                    channel = EXCLUDED.channel,
                        type = EXCLUDED.type,
                        blob = EXCLUDED.blob;
                """
        insert = f"""INSERT INTO "{self.schema_name}"."{self.table_name_writes}"(thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob)
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
        SELECT = f"""
        SELECT
            thread_id,
            checkpoint,
            checkpoint_ns,
            checkpoint_id,
            parent_checkpoint_id,
            metadata,
            type,
            (
                SELECT array_agg(array[cw.task_id::text::bytea, cw.channel::bytea, cw.type::bytea, cw.blob] order by cw.task_id, cw.idx)
                FROM "{self.schema_name}"."{self.table_name_writes}" cw
                where cw.thread_id = c.thread_id
                    AND cw.checkpoint_ns = c.checkpoint_ns
                    AND cw.checkpoint_id = c.checkpoint_id
            ) AS pending_writes,
            (
                SELECT array_agg(array[cw.type::bytea, cw.blob] order by cw.task_path, cw.task_id, cw.idx)
                FROM "{self.schema_name}"."{self.table_name_writes}" cw
                WHERE cw.thread_id = c.thread_id
                    AND cw.checkpoint_ns = c.checkpoint_ns
                    AND cw.checkpoint_id = c.parent_checkpoint_id
                    AND cw.channel = '{TASKS}'
            ) AS pending_sends
        FROM "{self.schema_name}"."{self.table_name}" c
        """

        where, args = self._search_where(config, filter, before)
        query = SELECT + where + " ORDER BY checkpoint_id DESC"
        if limit:
            query += f" LIMIT {limit}"

        async with self.pool.connect() as conn:
            result = await conn.execute(text(query), args)
            while True:
                row = result.fetchone()
                if not row:
                    break
                value = row._mapping
                yield CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": value["thread_id"],
                            "checkpoint_ns": value["checkpoint_ns"],
                            "checkpoint_id": value["checkpoint_id"],
                        }
                    },
                    checkpoint=self.serde.loads_typed(
                        (value["type"], value["checkpoint"])
                    ),
                    metadata=(
                        self.jsonplus_serde.loads(value["metadata"])  # type: ignore
                        if value["metadata"] is not None
                        else {}
                    ),
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

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Asynchronously fetch a checkpoint tuple using the given configuration.

        Args:
            config (RunnableConfig): Configuration specifying which checkpoint to retrieve.

        Returns:
            Optional[CheckpointTuple]: The requested checkpoint tuple, or None if not found.
        """

        SELECT = f"""
        SELECT
            thread_id,
            checkpoint,
            checkpoint_ns,
            checkpoint_id,
            parent_checkpoint_id,
            metadata,
            type,
            (
                SELECT array_agg(array[cw.task_id::text::bytea, cw.channel::bytea, cw.type::bytea, cw.blob] order by cw.task_id, cw.idx)
                FROM "{self.schema_name}"."{self.table_name_writes}" cw
                where cw.thread_id = c.thread_id
                    AND cw.checkpoint_ns = c.checkpoint_ns
                    AND cw.checkpoint_id = c.checkpoint_id
            ) AS pending_writes,
            (
                SELECT array_agg(array[cw.type::bytea, cw.blob] order by cw.task_path, cw.task_id, cw.idx)
                FROM "{self.schema_name}"."{self.table_name_writes}" cw
                WHERE cw.thread_id = c.thread_id
                    AND cw.checkpoint_ns = c.checkpoint_ns
                    AND cw.checkpoint_id = c.parent_checkpoint_id
                    AND cw.channel = '{TASKS}'
            ) AS pending_sends
        FROM "{self.schema_name}"."{self.table_name}" c
        """

        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        if checkpoint_id:
            args = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
            where = "WHERE thread_id = :thread_id AND checkpoint_ns = :checkpoint_ns AND checkpoint_id = :checkpoint_id"
        else:
            args = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}
            where = "WHERE thread_id = :thread_id AND checkpoint_ns = :checkpoint_ns ORDER BY checkpoint_id DESC LIMIT 1"

        async with self.pool.connect() as conn:
            result = await conn.execute(text(SELECT + where), args)
            row = result.fetchone()
            if not row:
                return None
            value = row._mapping
            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": value["thread_id"],
                        "checkpoint_ns": value["checkpoint_ns"],
                        "checkpoint_id": value["checkpoint_id"],
                    }
                },
                checkpoint=self.serde.loads_typed((value["type"], value["checkpoint"])),
                metadata=(
                    self.jsonplus_serde.loads(value["metadata"])  # type: ignore
                    if value["metadata"] is not None
                    else {}
                ),
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

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Asynchronously store a checkpoint with its configuration and metadata.

        Args:
            config (RunnableConfig): Configuration for the checkpoint.
            checkpoint (Checkpoint): The checkpoint to store.
            metadata (CheckpointMetadata): Additional metadata for the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresSaver. Use PostgresSaver interface instead."
        )

    def put_writes(
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
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresSaver. Use PostgresSaver interface instead."
        )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """Asynchronously list checkpoints that match the given criteria.

        Args:
            config (Optional[RunnableConfig]): Base configuration for filtering checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata.
            before (Optional[RunnableConfig]): List checkpoints created before this configuration.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Returns:
            AsyncIterator[CheckpointTuple]: Async iterator of matching checkpoint tuples.
        """
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresSaver. Use PostgresSaver interface instead."
        )

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Asynchronously fetch a checkpoint tuple using the given configuration.

        Args:
            config (RunnableConfig): Configuration specifying which checkpoint to retrieve.

        Returns:
            Optional[CheckpointTuple]: The requested checkpoint tuple, or None if not found.
        """
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresSaver. Use PostgresSaver interface instead."
        )
