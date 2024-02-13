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
from typing import Any, Callable, Iterable, List, Optional, Tuple

import numpy as np
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from .indexes import (
    DEFAULT_DISTANCE_STRATEGY,
    DistanceStrategy,
    HNSWIndex,
    HNSWQueryOptions,
    IVFFlatIndex,
    IVFFlatQueryOptions,
    QueryOptions,
)
from .postgresql_engine import PostgreSQLEngine


class CloudSQLVectorStore(VectorStore):
    """Google Cloud SQL for PostgreSQL Vector Store class"""

    __create_key = object()

    def __init__(
        self,
        key,
        engine: PostgreSQLEngine,
        embedding_service: Embeddings,
        table_name: str,
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
        if key != CloudSQLVectorStore.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync' methods!"
            )

        self.engine = engine
        self.embedding_service = embedding_service
        self.table_name = table_name
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
        engine: PostgreSQLEngine,
        embedding_service: Embeddings,
        table_name: str,
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
    ):
        """Constructor for CloudSQLVectorStore.
        Args:
            engine (PostgreSQLEngine): AsyncEngine with pool connection to the postgres database. Required.
            embedding_service (Embeddings): Text embedding model to use.
            table_name (str): Name of the existing table or the table to be created.
            id_column (str): Column that represents the Document's id. Defaults to "langchain_id".
            content_column (str): Column that represent a Document’s page_content. Defaults to "content".
            embedding_column (str): Column for embedding vectors.
                              The embedding is generated from the document value. Defaults to "embedding".
            metadata_columns (List[str]): Column(s) that represent a document's metadata.
            ignore_metadata_columns (List[str]): Column(s) to ignore in pre-existing tables for a document’s metadata.
                                     Can not be used with metadata_columns. Defaults to None.
            metadata_json_column (str): Column to store metadata as JSON. Defaults to "langchain_metadata".
        """
        if metadata_columns and ignore_metadata_columns:
            raise ValueError(
                "Can not use both metadata_columns and ignore_metadata_columns."
            )
        # Get field type information
        stmt = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}'"
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
            metadata_columns = [k for k, _ in all_columns.keys()]

        return cls(
            cls.__create_key,
            engine,
            embedding_service,
            table_name,
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
        engine: PostgreSQLEngine,
        embedding_service: Embeddings,
        table_name: str,
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
    ):
        coro = cls.create(
            engine,
            embedding_service,
            table_name,
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
        return engine.run_as_sync(coro)

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
        if not ids:
            ids = ["NULL" for _ in texts]
        if not metadatas:
            metadatas = [{} for _ in texts]
        # Insert embeddings
        for id, content, embedding, metadata in zip(ids, texts, embeddings, metadatas):
            metadata_col_names = (
                ", " + ", ".join(self.metadata_columns)
                if len(self.metadata_columns) > 0
                else ""
            )
            insert_stmt = f"INSERT INTO {self.table_name}({self.id_column}, {self.content_column}, {self.embedding_column}{metadata_col_names}"
            values_stmt = f" VALUES ('{id}','{content}','{embedding}'"
            extra = metadata
            for metadata_column in self.metadata_columns:
                if metadata_column in metadata:
                    values_stmt += f",'{metadata[metadata_column]}'"
                    del extra[metadata_column]
                else:
                    values_stmt += ",null"

            insert_stmt += (
                f", {self.metadata_json_column})" if self.metadata_json_column else ")"
            )
            values_stmt += (
                f",'{json.dumps(extra)}')" if self.metadata_json_column else ")"
            )
            query = insert_stmt + values_stmt
            await self.engine._aexecute(query)

        return ids

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
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
        return self.engine.run_as_sync(self.aadd_texts(texts, metadatas, ids, **kwargs))

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        return self.engine.run_as_sync(self.aadd_documents(documents, ids, **kwargs))

    async def adelete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        if not ids:
            return False

        id_list = ", ".join([f"'{id}'" for id in ids])
        query = f"DELETE FROM {self.table_name} WHERE {self.id_column} in ({id_list})"
        await self.engine._aexecute(query)
        return True

    def delete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        return self.engine.run_as_sync(self.adelete(ids, **kwargs))

    @classmethod
    async def afrom_texts(  # type: ignore[override]
        cls: Type[CloudSQLVectorStore],
        texts: List[str],
        embedding: Embeddings,
        engine: PostgreSQLEngine,
        table_name: str,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: List[str] = [],
        ignore_metadata_columns: Optional[List[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: str = "langchain_metadata",
        **kwargs: Any,
    ) -> CloudSQLVectorStore:
        vs = await cls.create(
            engine,
            embedding,
            table_name,
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
        cls: Type[CloudSQLVectorStore],
        documents: List[Document],
        embedding: Embeddings,
        engine: PostgreSQLEngine,
        table_name: str,
        ids: Optional[List[str]] = None,
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: List[str] = [],
        ignore_metadata_columns: Optional[List[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: str = "langchain_metadata",
        **kwargs: Any,
    ) -> CloudSQLVectorStore:
        vs = await cls.create(
            engine,
            embedding,
            table_name,
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
        cls: Type[CloudSQLVectorStore],
        texts: List[str],
        embedding: Embeddings,
        engine: PostgreSQLEngine,
        table_name: str,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: List[str] = [],
        ignore_metadata_columns: Optional[List[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: str = "langchain_metadata",
        **kwargs: Any,
    ):
        coro = cls.afrom_texts(
            texts,
            embedding,
            engine,
            table_name,
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
        return engine.run_as_sync(coro)

    @classmethod
    def from_documents(  # type: ignore[override]
        cls: Type[CloudSQLVectorStore],
        documents: List[Document],
        embedding: Embeddings,
        engine: PostgreSQLEngine,
        table_name: str,
        ids: Optional[List[str]] = None,
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: List[str] = [],
        ignore_metadata_columns: Optional[List[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: str = "langchain_metadata",
        **kwargs: Any,
    ) -> CloudSQLVectorStore:
        coro = cls.afrom_documents(
            documents,
            embedding,
            engine,
            table_name,
            content_column=content_column,
            embedding_column=embedding_column,
            metadata_columns=metadata_columns,
            ignore_metadata_columns=ignore_metadata_columns,
            metadata_json_column=metadata_json_column,
            id_column=id_column,
            ids=ids,
            **kwargs,
        )
        return engine.run_as_sync(coro)

    async def __query_collection(
        self,
        embedding: List[float],
        k: Optional[int] = None,
        filter: Optional[str] = None,
    ) -> List[Any]:
        k = k if k else self.k
        if self.distance_strategy == DistanceStrategy.EUCLIDEAN:
            operator = "<->"
            vector_function = "l2_distance"
        elif self.distance_strategy == DistanceStrategy.COSINE_DISTANCE:
            operator = "<=>"
            vector_function = "cosine_distance"
        elif self.distance_strategy == DistanceStrategy.INNER_PRODUCT:
            operator = "<#>"
            vector_function = "inner_product"
        else:
            raise ValueError(
                "distance strategy is not one of: 'EUCLIDEAN', 'COSINE_DISTANCE', 'INNER_PRODUCT'"
            )

        filter = f"WHERE {filter}" if filter else ""
        stmt = f"SELECT *, {vector_function}({self.embedding_column}, '{embedding}') as distance FROM {self.table_name} {filter} ORDER BY {self.embedding_column} {operator} '{embedding}' LIMIT {k};"
        if self.index_query_options:
            await self.engine._aexecute(
                f"SET LOCAL {self.index_query_options.to_string()};"
            )
        results = await self.engine._afetch(stmt)
        return results

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        return self.engine.run_as_sync(
            self.asimilarity_search(query, k=k, filter=filter, **kwargs)
        )

    async def asimilarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        embedding = self.embedding_service.embed_query(text=query)

        return await self.asimilarity_search_by_vector(
            embedding=embedding, k=k, filter=filter, **kwargs
        )

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
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

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        if self.distance_strategy == DistanceStrategy.COSINE_DISTANCE:
            return self._cosine_relevance_score_fn
        elif self.distance_strategy == DistanceStrategy.EUCLIDEAN:
            return self._euclidean_relevance_score_fn
        elif self.distance_strategy == DistanceStrategy.INNER_PRODUCT:
            return self._max_inner_product_relevance_score_fn
        else:
            raise ValueError(
                "No supported normalization function"
                f" for distance_strategy of {self.distance_strategy}."
                "Consider providing relevance_score_fn to PGVector constructor."
            )

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        coro = self.asimilarity_search_with_score(query, k, filter=filter, **kwargs)
        return self.engine.run_as_sync(coro)

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: Optional[int] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        coro = self.asimilarity_search_by_vector(embedding, k, filter=filter, **kwargs)
        return self.engine.run_as_sync(coro)

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: Optional[int] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        coro = self.asimilarity_search_with_score_by_vector(
            embedding, k, filter=filter, **kwargs
        )
        return self.engine.run_as_sync(coro)

    def max_marginal_relevance_search(
        self,
        query: str,
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        coro = self.amax_marginal_relevance_search(
            query,
            k,
            filter=filter,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            **kwargs,
        )
        return self.engine.run_as_sync(coro)

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        coro = self.amax_marginal_relevance_search_by_vector(
            embedding,
            k,
            filter=filter,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            **kwargs,
        )
        return self.engine.run_as_sync(coro)

    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        coro = self.amax_marginal_relevance_search_with_score_by_vector(
            embedding,
            k,
            filter=filter,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            **kwargs,
        )
        return self.engine.run_as_sync(coro)
