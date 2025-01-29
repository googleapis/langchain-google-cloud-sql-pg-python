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
from typing import Optional
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
from .engine import CHECKPOINTS_TABLE, CHECKPOINT_WRITES_TABLE, PostgresEngine


class AsyncPostgresCloudSQLSaver(BaseCheckpointSaver[str]):
    """Checkpoint stored in a PostgreSQL database running on Cloud SQL (GCP)."""

    __create_key = object()
    jsonplus_serde = JsonPlusSerializer()

    def __init__(
        self,
        key: object,
        engine: PostgresEngine,
        schema_name: str = "public",
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        super().__init__(serde=serde)
        if key != AsyncPostgresCloudSQLSaver.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync' methods"
            )
        self.engine = engine
        self.schema_name = schema_name

    @classmethod
    async def create(
        cls,
        engine: PostgresEngine,
        schema_name: str = "public",
        serde: Optional[SerializerProtocol] = None,
    ) -> "AsyncPostgresCloudSQLSaver":
        """Create a new AsyncPostgresCloudSQLSaver instance using a Cloud SQL connection."""

        async with engine._pool.connect() as conn:
            result = await conn.execute(
                text(
                    f"SELECT column_name FROM information_schema.columns WHERE table_schema = :schema AND table_name = :table"
                ),
                {"schema": schema_name, "table": CHECKPOINTS_TABLE},
            )
            checkpoints_column_names = [row[0] for row in result.fetchall()]

        required_columns = [
            "thread_id",
            "checkpoint_ns",
            "checkpoint_id",
            "parent_checkpoint_id",
            "type",
            "checkpoint",
            "metadata",
        ]

        if not all(x in checkpoints_column_names for x in required_columns):
            raise IndexError(
                f"Table {schema_name}.{CHECKPOINTS_TABLE} has incorrect schema. Expected: {required_columns}, Found: {checkpoints_column_names}"
            )

        async with engine._pool.connect() as conn:
            result = await conn.execute(
                text(
                    f"SELECT column_name FROM information_schema.columns WHERE table_schema = :schema AND table_name = :table"
                ),
                {"schema": schema_name, "table": CHECKPOINT_WRITES_TABLE},
            )
            checkpoint_writes_column_names = [row[0] for row in result.fetchall()]

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

        if not all(
            x in checkpoint_writes_column_names for x in checkpoint_writes_columns
        ):
            raise IndexError(
                f"Table {schema_name}.{CHECKPOINT_WRITES_TABLE} has incorrect schema. Expected: {checkpoint_writes_columns}, Found: {checkpoint_writes_column_names}"
            )

        return cls(cls.__create_key, engine, schema_name, serde)

    def _dump_checkpoint(self, checkpoint: Checkpoint) -> str:
        checkpoint["pending_sends"] = []
        return json.dumps(checkpoint)

    def _dump_metadata(self, metadata: CheckpointMetadata) -> str:
        serialized_metadata = self.jsonplus_serde.dumps(metadata)
        return serialized_metadata.decode().replace("\\u0000", "")

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Asynchronously store a checkpoint with its configuration and metadata in Cloud SQL."""

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

        query = f"""
        INSERT INTO "{self.schema_name}".{CHECKPOINTS_TABLE}
            (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata)
        VALUES
            (:thread_id, :checkpoint_ns, :checkpoint_id, :parent_checkpoint_id, :checkpoint, :metadata)
        ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id)
        DO UPDATE SET
            checkpoint = EXCLUDED.checkpoint,
            metadata = EXCLUDED.metadata;
        """

        async with self.engine._pool.connect() as conn:
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
