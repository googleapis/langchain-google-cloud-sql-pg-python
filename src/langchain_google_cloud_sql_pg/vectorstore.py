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

# TODO: Remove below import when minimum supported Python version is 3.10
from __future__ import annotations

from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVectorStore

from langchain_google_cloud_sql_pg import HybridSearchConfig

from .async_vectorstore import AsyncPostgresVectorStore
from .engine import PostgresEngine
from .indexes import (
    DEFAULT_DISTANCE_STRATEGY,
    BaseIndex,
    DistanceStrategy,
    QueryOptions,
)


class PostgresVectorStore(PGVectorStore):
    """Google Cloud SQL for PostgreSQL Vector Store class"""

    _engine: PostgresEngine
    __vs: AsyncPostgresVectorStore

    @classmethod
    async def create(
        cls,
        engine: PostgresEngine,  # type: ignore
        embedding_service: Embeddings,
        table_name: str,
        schema_name: str = "public",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: Optional[list[str]] = None,
        ignore_metadata_columns: Optional[list[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: Optional[str] = "langchain_metadata",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        index_query_options: Optional[QueryOptions] = None,
        hybrid_search_config: Optional[HybridSearchConfig] = None,
    ) -> PostgresVectorStore:
        """Create a new PostgresVectorStore instance.

        Args:
            engine (PostgresEngine): Connection pool engine for managing connections to Cloud SQL for PostgreSQL database.
            embedding_service (Embeddings): Text embedding model to use.
            table_name (str): Name of an existing table or table to be created.
            schema_name (str, optional): Database schema name of the table. Defaults to "public".
            content_column (str): Column that represent a Document's page_content. Defaults to "content".
            embedding_column (str): Column for embedding vectors. The embedding is generated from the document value. Defaults to "embedding".
            metadata_columns (list[str]): Column(s) that represent a document's metadata.
            ignore_metadata_columns (list[str]): Column(s) to ignore in pre-existing tables for a document's metadata. Can not be used with metadata_columns. Defaults to None.
            id_column (str): Column that represents the Document's id. Defaults to "langchain_id".
            metadata_json_column (str): Column to store metadata as JSON. Defaults to "langchain_metadata".
            distance_strategy (DistanceStrategy): Distance strategy to use for vector similarity search. Defaults to COSINE_DISTANCE.
            k (int): Number of Documents to return from search. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult (float): Number between 0 and 1 that determines the degree of diversity among the results with 0 corresponding to maximum diversity and 1 to minimum diversity. Defaults to 0.5.
            index_query_options (QueryOptions): Index query option.
            hybrid_search_config (HybridSearchConfig): Hybrid search configuration. Defaults to None.

        Returns:
            PostgresVectorStore
        """
        coro = AsyncPostgresVectorStore.create(
            engine,
            embedding_service,
            table_name,
            schema_name=schema_name,
            content_column=content_column,
            embedding_column=embedding_column,
            metadata_columns=metadata_columns,
            ignore_metadata_columns=ignore_metadata_columns,
            metadata_json_column=metadata_json_column,
            id_column=id_column,
            distance_strategy=distance_strategy,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            index_query_options=index_query_options,
            hybrid_search_config=hybrid_search_config,
        )
        vs = await engine._run_as_async(coro)
        return cls(cls._PGVectorStore__create_key, engine, vs)  # type: ignore

    @classmethod
    def create_sync(
        cls,
        engine: PostgresEngine,  # type: ignore
        embedding_service: Embeddings,
        table_name: str,
        schema_name: str = "public",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: Optional[list[str]] = None,
        ignore_metadata_columns: Optional[list[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: str = "langchain_metadata",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        index_query_options: Optional[QueryOptions] = None,
        hybrid_search_config: Optional[HybridSearchConfig] = None,
    ) -> PostgresVectorStore:
        """Create a new PostgresVectorStore instance.

        Args:
            engine (PostgresEngine): Connection pool engine for managing connections to Cloud SQL for PostgreSQL database.
            embedding_service (Embeddings): Text embedding model to use.
            table_name (str): Name of an existing table or table to be created.
            schema_name (str, optional): Database schema name of the table. Defaults to "public".
            content_column (str): Column that represent a Document's page_content. Defaults to "content".
            embedding_column (str): Column for embedding vectors. The embedding is generated from the document value. Defaults to "embedding".
            metadata_columns (list[str]): Column(s) that represent a document's metadata.
            ignore_metadata_columns (list[str]): Column(s) to ignore in pre-existing tables for a document's metadata. Can not be used with metadata_columns. Defaults to None.
            id_column (str): Column that represents the Document's id. Defaults to "langchain_id".
            metadata_json_column (str): Column to store metadata as JSON. Defaults to "langchain_metadata".
            distance_strategy (DistanceStrategy): Distance strategy to use for vector similarity search. Defaults to COSINE_DISTANCE.
            k (int): Number of Documents to return from search. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult (float): Number between 0 and 1 that determines the degree of diversity among the results with 0 corresponding to maximum diversity and 1 to minimum diversity. Defaults to 0.5.
            index_query_options (QueryOptions): Index query option.
            hybrid_search_config (HybridSearchConfig): Hybrid search configuration. Defaults to None.

        Returns:
            PostgresVectorStore
        """
        coro = AsyncPostgresVectorStore.create(
            engine,
            embedding_service,
            table_name,
            schema_name=schema_name,
            content_column=content_column,
            embedding_column=embedding_column,
            metadata_columns=metadata_columns,
            ignore_metadata_columns=ignore_metadata_columns,
            metadata_json_column=metadata_json_column,
            id_column=id_column,
            distance_strategy=distance_strategy,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            index_query_options=index_query_options,
            hybrid_search_config=hybrid_search_config,
        )
        vs = engine._run_as_sync(coro)
        return cls(cls._PGVectorStore__create_key, engine, vs)  # type: ignore
