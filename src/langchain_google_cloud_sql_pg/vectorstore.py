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

import json
import uuid
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from sqlalchemy.engine.row import RowMapping

from .engine import PostgresEngine
from .indexes import (
    DEFAULT_DISTANCE_STRATEGY,
    DEFAULT_INDEX_NAME_SUFFIX,
    BaseIndex,
    DistanceStrategy,
    ExactNearestNeighbor,
    QueryOptions,
)


class PostgresVectorStore(VectorStore):
    """Google Cloud SQL for PostgreSQL Vector Store class"""

    __create_key = object()

    def __init__(
        self,
        key: object,
        engine: PostgresEngine,
        embedding_service: Embeddings,
        table_name: str,
        schema_name: str = "public",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: List[str] = [],
        id_column: str = "langchain_id",
        metadata_json_column: Optional[str] = "langchain_metadata",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        index_query_options: Optional[QueryOptions] = None,
    ):
        """PostgresVectorStore constructor.
        Args:
            key (object): Prevent direct constructor usage.
            engine (PostgresEngine): Connection pool engine for managing connections to Postgres database.
            embedding_service (Embeddings): Text embedding model to use.
            table_name (str): Name of the existing table or the table to be created.
            schema_name (str, optional): Database schema name of the table. Defaults to "public".
            content_column (str): Column that represent a Document’s page_content. Defaults to "content".
            embedding_column (str): Column for embedding vectors. The embedding is generated from the document value. Defaults to "embedding".
            metadata_columns (List[str]): Column(s) that represent a document's metadata.
            id_column (str): Column that represents the Document's id. Defaults to "langchain_id".
            metadata_json_column (str): Column to store metadata as JSON. Defaults to "langchain_metadata".
            distance_strategy (DistanceStrategy): Distance strategy to use for vector similarity search. Defaults to COSINE_DISTANCE.
            k (int): Number of Documents to return from search. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult (float): Number between 0 and 1 that determines the degree of diversity among the results with 0 corresponding to maximum diversity and 1 to minimum diversity. Defaults to 0.5.
            index_query_options (QueryOptions): Index query option.


        Raises:
            Exception: If called directly by user.
        """
        if key != PostgresVectorStore.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync' methods!"
            )

        self.engine = engine
        self.embedding_service = embedding_service
        self.table_name = table_name
        self.schema_name = schema_name
        self.content_column = content_column
        self.embedding_column = embedding_column
        self.metadata_columns = metadata_columns
        self.id_column = id_column
        self.metadata_json_column = metadata_json_column
        self.distance_strategy = distance_strategy
        self.k = k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.index_query_options = index_query_options

    @classmethod
    async def create(
        cls,
        engine: PostgresEngine,
        embedding_service: Embeddings,
        table_name: str,
        schema_name: str = "public",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: List[str] = [],
        ignore_metadata_columns: Optional[List[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: Optional[str] = "langchain_metadata",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        index_query_options: Optional[QueryOptions] = None,
    ) -> PostgresVectorStore:
        """Create a new PostgresVectorStore instance.

        Args:
            engine (PostgresEngine): Connection pool engine for managing connections to Cloud SQL for PostgreSQL database.
            embedding_service (Embeddings): Text embedding model to use.
            table_name (str): Name of an existing table or table to be created.
            schema_name (str, optional): Database schema name of the table. Defaults to "public".
            content_column (str): Column that represent a Document's page_content. Defaults to "content".
            embedding_column (str): Column for embedding vectors. The embedding is generated from the document value. Defaults to "embedding".
            metadata_columns (List[str]): Column(s) that represent a document's metadata.
            ignore_metadata_columns (List[str]): Column(s) to ignore in pre-existing tables for a document's metadata. Can not be used with metadata_columns. Defaults to None.
            id_column (str): Column that represents the Document's id. Defaults to "langchain_id".
            metadata_json_column (str): Column to store metadata as JSON. Defaults to "langchain_metadata".
            distance_strategy (DistanceStrategy): Distance strategy to use for vector similarity search. Defaults to COSINE_DISTANCE.
            k (int): Number of Documents to return from search. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult (float): Number between 0 and 1 that determines the degree of diversity among the results with 0 corresponding to maximum diversity and 1 to minimum diversity. Defaults to 0.5.
            index_query_options (QueryOptions): Index query option.

        Returns:
            PostgresVectorStore
        """
        if metadata_columns and ignore_metadata_columns:
            raise ValueError(
                "Can not use both metadata_columns and ignore_metadata_columns."
            )
        # Get field type information
        stmt = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}' AND table_schema = '{schema_name}'"
        results = await engine._afetch(stmt)
        columns = {}
        for field in results:
            columns[field["column_name"]] = field["data_type"]

        # Check columns
        if id_column not in columns:
            raise ValueError(f"Id column, {id_column}, does not exist.")
        if content_column not in columns:
            raise ValueError(f"Content column, {content_column}, does not exist.")
        content_type = columns[content_column]
        if content_type != "text" and "char" not in content_type:
            raise ValueError(
                f"Content column, {content_column}, is type, {content_type}. It must be a type of character string."
            )
        if embedding_column not in columns:
            raise ValueError(f"Embedding column, {embedding_column}, does not exist.")
        if columns[embedding_column] != "USER-DEFINED":
            raise ValueError(
                f"Embedding column, {embedding_column}, is not type Vector."
            )

        metadata_json_column = (
            None if metadata_json_column not in columns else metadata_json_column
        )

        # If using metadata_columns check to make sure column exists
        for column in metadata_columns:
            if column not in columns:
                raise ValueError(f"Metadata column, {column}, does not exist.")

        # If using ignore_metadata_columns, filter out known columns and set known metadata columns
        all_columns = columns
        if ignore_metadata_columns:
            for column in ignore_metadata_columns:
                del all_columns[column]

            del all_columns[id_column]
            del all_columns[content_column]
            del all_columns[embedding_column]
            metadata_columns = [k for k in all_columns.keys()]

        return cls(
            cls.__create_key,
            engine,
            embedding_service,
            table_name,
            schema_name,
            content_column,
            embedding_column,
            metadata_columns,
            id_column,
            metadata_json_column,
            distance_strategy,
            k,
            fetch_k,
            lambda_mult,
            index_query_options,
        )

    @classmethod
    def create_sync(
        cls,
        engine: PostgresEngine,
        embedding_service: Embeddings,
        table_name: str,
        schema_name: str = "public",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: List[str] = [],
        ignore_metadata_columns: Optional[List[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: str = "langchain_metadata",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        index_query_options: Optional[QueryOptions] = None,
    ) -> PostgresVectorStore:
        """Create a new PostgresVectorStore instance.

        Args:
            engine (PostgresEngine): Connection pool engine for managing connections to Cloud SQL for PostgreSQL database.
            embedding_service (Embeddings): Text embedding model to use.
            table_name (str): Name of an existing table or table to be created.
            schema_name (str, optional): Database schema name of the table. Defaults to "public".
            content_column (str): Column that represent a Document's page_content. Defaults to "content".
            embedding_column (str): Column for embedding vectors. The embedding is generated from the document value. Defaults to "embedding".
            metadata_columns (List[str]): Column(s) that represent a document's metadata.
            ignore_metadata_columns (List[str]): Column(s) to ignore in pre-existing tables for a document's metadata. Can not be used with metadata_columns. Defaults to None.
            id_column (str): Column that represents the Document's id. Defaults to "langchain_id".
            metadata_json_column (str): Column to store metadata as JSON. Defaults to "langchain_metadata".
            distance_strategy (DistanceStrategy): Distance strategy to use for vector similarity search. Defaults to COSINE_DISTANCE.
            k (int): Number of Documents to return from search. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult (float): Number between 0 and 1 that determines the degree of diversity among the results with 0 corresponding to maximum diversity and 1 to minimum diversity. Defaults to 0.5.
            index_query_options (QueryOptions): Index query option.

        Returns:
            PostgresVectorStore
        """
        coro = cls.create(
            engine,
            embedding_service,
            table_name,
            schema_name,
            content_column,
            embedding_column,
            metadata_columns,
            ignore_metadata_columns,
            id_column,
            metadata_json_column,
            distance_strategy,
            k,
            fetch_k,
            lambda_mult,
            index_query_options,
        )
        return engine._run_as_sync(coro)

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_service

    async def _aadd_embeddings(
        self,
        texts: Iterable[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add embeddings to the table."""
        if not ids:
            ids = [str(uuid.uuid4()) for _ in texts]
        if not metadatas:
            metadatas = [{} for _ in texts]
        # Insert embeddings
        for id, content, embedding, metadata in zip(ids, texts, embeddings, metadatas):
            metadata_col_names = (
                ", " + ", ".join(self.metadata_columns)
                if len(self.metadata_columns) > 0
                else ""
            )
            insert_stmt = f'INSERT INTO "{self.schema_name}"."{self.table_name}"({self.id_column}, {self.content_column}, {self.embedding_column}{metadata_col_names}'
            values = {"id": id, "content": content, "embedding": str(embedding)}
            values_stmt = "VALUES (:id, :content, :embedding"

            # Add metadata
            extra = metadata
            for metadata_column in self.metadata_columns:
                if metadata_column in metadata:
                    values_stmt += f", :{metadata_column}"
                    values[metadata_column] = metadata[metadata_column]
                    del extra[metadata_column]
                else:
                    values_stmt += ",null"

            # Add JSON column and/or close statement
            insert_stmt += (
                f", {self.metadata_json_column})" if self.metadata_json_column else ")"
            )
            if self.metadata_json_column:
                values_stmt += ", :extra)"
                values["extra"] = json.dumps(extra)
            else:
                values_stmt += ")"

            query = insert_stmt + values_stmt
            await self.engine._aexecute(query, values)

        return ids

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Embed texts and add to the table."""
        embeddings = self.embedding_service.embed_documents(list(texts))
        ids = await self._aadd_embeddings(
            texts, embeddings, metadatas=metadatas, ids=ids, **kwargs
        )
        return ids

    async def aadd_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Embed documents and add to the table"""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = await self.aadd_texts(texts, metadatas=metadatas, ids=ids, **kwargs)
        return ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Embed texts and add to the table."""
        return self.engine._run_as_sync(
            self.aadd_texts(texts, metadatas, ids, **kwargs)
        )

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Embed documents and add to the table."""
        return self.engine._run_as_sync(self.aadd_documents(documents, ids, **kwargs))

    async def adelete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete records from the table."""
        if not ids:
            return False

        id_list = ", ".join([f"'{id}'" for id in ids])
        query = f'DELETE FROM "{self.schema_name}"."{self.table_name}" WHERE {self.id_column} in ({id_list})'
        await self.engine._aexecute(query)
        return True

    def delete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete records from the table."""
        return self.engine._run_as_sync(self.adelete(ids, **kwargs))

    @classmethod
    async def afrom_texts(  # type: ignore[override]
        cls: Type[PostgresVectorStore],
        texts: List[str],
        embedding: Embeddings,
        engine: PostgresEngine,
        table_name: str,
        schema_name: str = "public",
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: List[str] = [],
        ignore_metadata_columns: Optional[List[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: str = "langchain_metadata",
        **kwargs: Any,
    ) -> PostgresVectorStore:
        """Create an PostgresVectorStore instance from texts.
        Args:
            texts (List[str]): Texts to add to the vector store.
            embedding (Embeddings): Text embedding model to use.
            engine (PostgresEngine): Connection pool engine for managing connections to Postgres database.
            table_name (str): Name of the existing table or the table to be created.
            schema_name (str, optional): Database schema name of the table. Defaults to "public".
            metadatas (Optional[List[dict]]): List of metadatas to add to table records.
            ids: (Optional[List[str]]): List of IDs to add to table records.
            content_column (str): Column that represent a Document’s page_content. Defaults to "content".
            embedding_column (str): Column for embedding vectors. The embedding is generated from the document value. Defaults to "embedding".
            metadata_columns (List[str]): Column(s) that represent a document's metadata.
            ignore_metadata_columns (List[str]): Column(s) to ignore in pre-existing tables for a document's metadata. Can not be used with metadata_columns. Defaults to None.
            id_column (str): Column that represents the Document's id. Defaults to "langchain_id".
            metadata_json_column (str): Column to store metadata as JSON. Defaults to "langchain_metadata".

        Returns:
            PostgresVectorStore
        """
        vs = await cls.create(
            engine,
            embedding,
            table_name,
            schema_name,
            content_column,
            embedding_column,
            metadata_columns,
            ignore_metadata_columns,
            id_column,
            metadata_json_column,
        )
        await vs.aadd_texts(texts, metadatas=metadatas, ids=ids, **kwargs)
        return vs

    @classmethod
    async def afrom_documents(  # type: ignore[override]
        cls: Type[PostgresVectorStore],
        documents: List[Document],
        embedding: Embeddings,
        engine: PostgresEngine,
        table_name: str,
        schema_name: str = "public",
        ids: Optional[List[str]] = None,
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: List[str] = [],
        ignore_metadata_columns: Optional[List[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: str = "langchain_metadata",
        **kwargs: Any,
    ) -> PostgresVectorStore:
        """Create an PostgresVectorStore instance from documents.

        Args:
            documents (List[Document]): Documents to add to the vector store.
            embedding (Embeddings): Text embedding model to use.
            engine (PostgresEngine): Connection pool engine for managing connections to Postgres database.
            table_name (str): Name of the existing table or the table to be created.
            schema_name (str, optional): Database schema name of the table. Defaults to "public".
            metadatas (Optional[List[dict]]): List of metadatas to add to table records.
            ids: (Optional[List[str]]): List of IDs to add to table records.
            content_column (str): Column that represent a Document’s page_content. Defaults to "content".
            embedding_column (str): Column for embedding vectors. The embedding is generated from the document value. Defaults to "embedding".
            metadata_columns (List[str]): Column(s) that represent a document's metadata.
            ignore_metadata_columns (List[str]): Column(s) to ignore in pre-existing tables for a document's metadata. Can not be used with metadata_columns. Defaults to None.
            id_column (str): Column that represents the Document's id. Defaults to "langchain_id".
            metadata_json_column (str): Column to store metadata as JSON. Defaults to "langchain_metadata".

        Returns:
            PostgresVectorStore
        """
        vs = await cls.create(
            engine,
            embedding,
            table_name,
            schema_name,
            content_column,
            embedding_column,
            metadata_columns,
            ignore_metadata_columns,
            id_column,
            metadata_json_column,
        )
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        await vs.aadd_texts(texts, metadatas=metadatas, ids=ids, **kwargs)
        return vs

    @classmethod
    def from_texts(  # type: ignore[override]
        cls: Type[PostgresVectorStore],
        texts: List[str],
        embedding: Embeddings,
        engine: PostgresEngine,
        table_name: str,
        schema_name: str = "public",
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: List[str] = [],
        ignore_metadata_columns: Optional[List[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: str = "langchain_metadata",
        **kwargs: Any,
    ) -> PostgresVectorStore:
        """Create an PostgresVectorStore instance from texts.
        Args:
            texts (List[str]): Texts to add to the vector store.
            embedding (Embeddings): Text embedding model to use.
            engine (PostgresEngine): Connection pool engine for managing connections to Postgres database.
            table_name (str): Name of the existing table or the table to be created.
            schema_name (str, optional): Database schema name of the table. Defaults to "public".
            metadatas (Optional[List[dict]]): List of metadatas to add to table records.
            ids: (Optional[List[str]]): List of IDs to add to table records.
            content_column (str): Column that represent a Document’s page_content. Defaults to "content".
            embedding_column (str): Column for embedding vectors. The embedding is generated from the document value. Defaults to "embedding".
            metadata_columns (List[str]): Column(s) that represent a document's metadata.
            ignore_metadata_columns (List[str]): Column(s) to ignore in pre-existing tables for a document's metadata. Can not be used with metadata_columns. Defaults to None.
            id_column (str): Column that represents the Document's id. Defaults to "langchain_id".
            metadata_json_column (str): Column to store metadata as JSON. Defaults to "langchain_metadata".

        Returns:
            PostgresVectorStore
        """
        coro = cls.afrom_texts(
            texts,
            embedding,
            engine,
            table_name,
            schema_name,
            metadatas=metadatas,
            content_column=content_column,
            embedding_column=embedding_column,
            metadata_columns=metadata_columns,
            ignore_metadata_columns=ignore_metadata_columns,
            metadata_json_column=metadata_json_column,
            id_column=id_column,
            ids=ids,
            **kwargs,
        )
        return engine._run_as_sync(coro)

    @classmethod
    def from_documents(  # type: ignore[override]
        cls: Type[PostgresVectorStore],
        documents: List[Document],
        embedding: Embeddings,
        engine: PostgresEngine,
        table_name: str,
        schema_name: str = "public",
        ids: Optional[List[str]] = None,
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: List[str] = [],
        ignore_metadata_columns: Optional[List[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: str = "langchain_metadata",
        **kwargs: Any,
    ) -> PostgresVectorStore:
        """Create an PostgresVectorStore instance from documents.

        Args:
            documents (List[Document]): Documents to add to the vector store.
            embedding (Embeddings): Text embedding model to use.
            engine (PostgresEngine): Connection pool engine for managing connections to Postgres database.
            table_name (str): Name of the existing table or the table to be created.
            schema_name (str, optional): Database schema name of the table. Defaults to "public".
            metadatas (Optional[List[dict]]): List of metadatas to add to table records.
            ids: (Optional[List[str]]): List of IDs to add to table records.
            content_column (str): Column that represent a Document’s page_content. Defaults to "content".
            embedding_column (str): Column for embedding vectors. The embedding is generated from the document value. Defaults to "embedding".
            metadata_columns (List[str]): Column(s) that represent a document's metadata.
            ignore_metadata_columns (List[str]): Column(s) to ignore in pre-existing tables for a document's metadata. Can not be used with metadata_columns. Defaults to None.
            id_column (str): Column that represents the Document's id. Defaults to "langchain_id".
            metadata_json_column (str): Column to store metadata as JSON. Defaults to "langchain_metadata".

        Returns:
            PostgresVectorStore
        """
        coro = cls.afrom_documents(
            documents,
            embedding,
            engine,
            table_name,
            schema_name,
            content_column=content_column,
            embedding_column=embedding_column,
            metadata_columns=metadata_columns,
            ignore_metadata_columns=ignore_metadata_columns,
            metadata_json_column=metadata_json_column,
            id_column=id_column,
            ids=ids,
            **kwargs,
        )
        return engine._run_as_sync(coro)

    async def __query_collection(
        self,
        embedding: List[float],
        k: Optional[int] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> Sequence[RowMapping]:
        """Perform similarity search query on the vector store table."""
        k = k if k else self.k
        operator = self.distance_strategy.operator
        search_function = self.distance_strategy.search_function

        filter = f"WHERE {filter}" if filter else ""
        stmt = f"SELECT *, {search_function}({self.embedding_column}, '{embedding}') as distance FROM \"{self.schema_name}\".\"{self.table_name}\" {filter} ORDER BY {self.embedding_column} {operator} '{embedding}' LIMIT {k};"
        if self.index_query_options:
            query_options_stmt = f"SET LOCAL {self.index_query_options.to_string()};"
            results = await self.engine._afetch_with_query_options(
                stmt, query_options_stmt
            )
        else:
            results = await self.engine._afetch(stmt)
        return results

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected by similarity search on query."""
        return self.engine._run_as_sync(
            self.asimilarity_search(query, k=k, filter=filter, **kwargs)
        )

    async def asimilarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected by similarity search on query."""
        embedding = self.embedding_service.embed_query(text=query)

        return await self.asimilarity_search_by_vector(
            embedding=embedding, k=k, filter=filter, **kwargs
        )

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """Select a relevance function based on distance strategy."""
        # Calculate distance strategy provided in
        # vectorstore constructor
        if self.distance_strategy == DistanceStrategy.COSINE_DISTANCE:
            return self._cosine_relevance_score_fn
        if self.distance_strategy == DistanceStrategy.INNER_PRODUCT:
            return self._max_inner_product_relevance_score_fn
        elif self.distance_strategy == DistanceStrategy.EUCLIDEAN:
            return self._euclidean_relevance_score_fn

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and distance scores selected by similarity search on query."""
        embedding = self.embedding_service.embed_query(query)
        docs = await self.asimilarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter, **kwargs
        )
        return docs

    async def asimilarity_search_by_vector(
        self,
        embedding: List[float],
        k: Optional[int] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected by vector similarity search."""
        docs_and_scores = await self.asimilarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter, **kwargs
        )

        return [doc for doc, _ in docs_and_scores]

    async def asimilarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: Optional[int] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and distance scores selected by vector similarity search."""
        results = await self.__query_collection(
            embedding=embedding, k=k, filter=filter, **kwargs
        )

        documents_with_scores = []
        for row in results:
            metadata = (
                row[self.metadata_json_column]
                if self.metadata_json_column and row[self.metadata_json_column]
                else {}
            )
            for col in self.metadata_columns:
                metadata[col] = row[col]
            documents_with_scores.append(
                (
                    Document(
                        page_content=row[self.content_column],
                        metadata=metadata,
                    ),
                    row["distance"],
                )
            )

        return documents_with_scores

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance."""
        embedding = self.embedding_service.embed_query(text=query)

        return await self.amax_marginal_relevance_search_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance."""
        docs_and_scores = (
            await self.amax_marginal_relevance_search_with_score_by_vector(
                embedding,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=filter,
                **kwargs,
            )
        )

        return [result[0] for result in docs_and_scores]

    async def amax_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and distance scores selected using the maximal marginal relevance."""
        results = await self.__query_collection(
            embedding=embedding, k=fetch_k, filter=filter, **kwargs
        )

        k = k if k else self.k
        fetch_k = fetch_k if fetch_k else self.fetch_k
        lambda_mult = lambda_mult if lambda_mult else self.lambda_mult
        embedding_list = [json.loads(row[self.embedding_column]) for row in results]
        mmr_selected = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            embedding_list,
            k=k,
            lambda_mult=lambda_mult,
        )

        documents_with_scores = []
        for row in results:
            metadata = (
                row[self.metadata_json_column]
                if self.metadata_json_column and row[self.metadata_json_column]
                else {}
            )
            for col in self.metadata_columns:
                metadata[col] = row[col]
            documents_with_scores.append(
                (
                    Document(
                        page_content=row[self.content_column],
                        metadata=metadata,
                    ),
                    row["distance"],
                )
            )

        return [r for i, r in enumerate(documents_with_scores) if i in mmr_selected]

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and distance scores selected by similarity search on query."""
        coro = self.asimilarity_search_with_score(query, k, filter=filter, **kwargs)
        return self.engine._run_as_sync(coro)

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: Optional[int] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected by vector similarity search."""
        coro = self.asimilarity_search_by_vector(embedding, k, filter=filter, **kwargs)
        return self.engine._run_as_sync(coro)

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: Optional[int] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and distance scores selected by similarity search on vector."""
        coro = self.asimilarity_search_with_score_by_vector(
            embedding, k, filter=filter, **kwargs
        )
        return self.engine._run_as_sync(coro)

    def max_marginal_relevance_search(
        self,
        query: str,
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance."""
        coro = self.amax_marginal_relevance_search(
            query,
            k,
            filter=filter,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            **kwargs,
        )
        return self.engine._run_as_sync(coro)

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance."""
        coro = self.amax_marginal_relevance_search_by_vector(
            embedding,
            k,
            filter=filter,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            **kwargs,
        )
        return self.engine._run_as_sync(coro)

    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and distance scores selected using the maximal marginal relevance."""
        coro = self.amax_marginal_relevance_search_with_score_by_vector(
            embedding,
            k,
            filter=filter,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            **kwargs,
        )
        return self.engine._run_as_sync(coro)

    async def aapply_vector_index(
        self,
        index: BaseIndex,
        name: Optional[str] = None,
        concurrently: bool = False,
    ) -> None:
        """Create an index on the vector store table."""
        if isinstance(index, ExactNearestNeighbor):
            await self.adrop_vector_index()
            return

        filter = f"WHERE ({index.partial_indexes})" if index.partial_indexes else ""
        params = "WITH " + index.index_options()
        function = index.distance_strategy.index_function
        if name is None:
            if index.name == None:
                index.name = self.table_name + DEFAULT_INDEX_NAME_SUFFIX
            name = index.name
        stmt = f'CREATE INDEX {"CONCURRENTLY" if concurrently else ""} {name} ON "{self.schema_name}"."{self.table_name}" USING {index.index_type} ({self.embedding_column} {function}) {params} {filter};'
        if concurrently:
            await self.engine._aexecute_outside_tx(stmt)
        else:
            await self.engine._aexecute(stmt)

    async def areindex(self, index_name: Optional[str] = None) -> None:
        """Re-index the vector store table."""
        index_name = index_name or self.table_name + DEFAULT_INDEX_NAME_SUFFIX
        query = f"REINDEX INDEX {index_name};"
        await self.engine._aexecute(query)

    async def adrop_vector_index(
        self,
        index_name: Optional[str] = None,
    ) -> None:
        """Drop the vector index."""
        index_name = index_name or self.table_name + DEFAULT_INDEX_NAME_SUFFIX
        query = f"DROP INDEX IF EXISTS {index_name};"
        await self.engine._aexecute(query)

    async def is_valid_index(
        self,
        index_name: Optional[str] = None,
    ) -> bool:
        """Check if index exists in the table."""
        index_name = index_name or self.table_name + DEFAULT_INDEX_NAME_SUFFIX
        query = f"""
        SELECT tablename, indexname
        FROM pg_indexes
        WHERE tablename = '{self.table_name}' AND schemaname = '{self.schema_name}' AND indexname = '{index_name}';
        """
        results = await self.engine._afetch(query)
        return bool(len(results) == 1)


### The following is copied from langchain-community until it's moved into core

Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: list,
    lambda_mult: float = 0.5,
    k: int = 4,
) -> List[int]:
    """Calculate maximal marginal relevance."""
    if min(k, len(embedding_list)) <= 0:
        return []
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = cosine_similarity(embedding_list, selected)
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue
            redundant_score = max(similarity_to_selected[i])
            equation_score = (
                lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            )
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    return idxs


def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices."""
    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. X has shape {X.shape} "
            f"and Y has shape {Y.shape}."
        )
    try:
        import simsimd as simd  # type: ignore

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        Z = 1 - simd.cdist(X, Y, metric="cosine")
        if isinstance(Z, float):
            return np.array([Z])
        return Z
    except ImportError:
        X_norm = np.linalg.norm(X, axis=1)
        Y_norm = np.linalg.norm(Y, axis=1)
        # Ignore divide by zero errors run time warnings as those are handled below.
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
        similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
        return similarity


### End code from langchain-community
