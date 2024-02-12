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

import asyncio
import json
from typing import Any, Awaitable, Callable, Iterable, List, Optional, Tuple

import nest_asyncio  # type: ignore
import numpy as np
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from .indexes import DEFAULT_DISTANCE_STRATEGY, DistanceStrategy
from .postgresql_engine import PostgreSQLEngine

nest_asyncio.apply()


class CloudSQLVectorStore(VectorStore):
    """Google Cloud SQL for PostgreSQL Vector Store class"""

    def __init__(
        self,
        engine: PostgreSQLEngine,
        embedding_service: Embeddings,
        table_name: str,
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: List[str] = [],
        ignore_metadata_columns: Optional[List[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: str = "langchain_metadata",
        overwrite_existing: bool = False,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
    ):
        """_summary_

        Args:
            engine (PostgreSQLEngine): _description_
            embedding_service (Embeddings): _description_
            table_name (str): _description_
            content_column (str): _description_
            embedding_column (str): _description_
            metadata_columns (List[str]): _description_
            ignore_metadata_columns (List[str]): _description_
            id_column (str): _description_
            metadata_json_column (str): _description_
            index_query_options (_type_): _description_
            distance_strategy (DistanceStrategy, optional): _description_. Defaults to DEFAULT_DISTANCE_STRATEGY.
        """
        self.engine = engine
        self.embedding_service = embedding_service
        self.table_name = table_name
        self.content_column = content_column
        self.embedding_column = embedding_column
        self.metadata_columns = metadata_columns
        self.ignore_metadata_columns = ignore_metadata_columns
        self.id_column = id_column
        self.metadata_json_column = metadata_json_column
        self.overwrite_existing = overwrite_existing
        self.store_metadata = False
        self.distance_strategy = distance_strategy
        self.k = k
        self.score_threshold = score_threshold
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult

        if metadata_columns and ignore_metadata_columns:
            raise ValueError(
                "Can not use both metadata_columns and ignore_metadata_columns."
            )
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.__post_init_async__())
        except RuntimeError:
            self.engine.run_as_sync(self.__post_init_async__())

    async def __post_init_async__(self) -> None:
        if self.overwrite_existing:
            await self.engine._aexecute(f"TRUNCATE TABLE {self.table_name}")

        # Get field type information
        stmt = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{self.table_name}'"
        results = await self.engine._afetch(stmt)
        columns = {}
        for field in results:
            columns[field["column_name"]] = field["data_type"]

        # Check columns
        if self.id_column not in columns:
            raise ValueError(f"Id column, {self.id_column}, does not exist.")
        if self.content_column not in columns:
            raise ValueError(
                f"Content column, {self.content_column}, does not exist."
            )
        content_type = columns[self.content_column]
        if content_type != "text" and "char" not in content_type:
            raise ValueError(
                f"Content column, {self.content_column}, is type, {content_type}. It must be a type of character string."
            )
        if self.embedding_column not in columns:
            raise ValueError(
                f"Embedding column, {self.embedding_column}, does not exist."
            )
        if columns[self.embedding_column] != "USER-DEFINED":
            raise ValueError(
                f"Embedding column, {self.embedding_column}, is not type Vector."
            )
        if self.metadata_json_column in columns:
            self.store_metadata = True

        # If using metadata_columns check to make sure column exists
        for column in self.metadata_columns:
            if column not in columns:
                raise ValueError(f"Metadata column, {column}, does not exist.")

        # If using ignore_metadata_columns, filter out known columns and set known metadata columns
        all_columns = columns
        if self.ignore_metadata_columns:
            for column in self.ignore_metadata_columns:
                del all_columns[column]

            del all_columns[self.id_column]
            del all_columns[self.content_column]
            del all_columns[self.embedding_column]
            self.metadata_columns = [k for k, _ in all_columns.keys()]

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
        for id, content, embedding, metadata in zip(
            ids, texts, embeddings, metadatas
        ):
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
                f", {self.metadata_json_column})"
                if self.store_metadata
                else ")"
            )
            values_stmt += (
                f",'{json.dumps(extra)}')" if self.store_metadata else ")"
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
        ids = await self.aadd_texts(
            texts, metadatas=metadatas, ids=ids, **kwargs
        )
        return ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        # try:
        #     loop = asyncio.get_running_loop()
        #     # loop = asyncio.get_event_loop()
        #     return loop.run_until_complete(
        #         self.aadd_texts(texts, metadatas, ids)
        #     )
        # except RuntimeError:
        return self.run_as_sync(self.aadd_texts(texts, metadatas, ids))

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        # try:
        #     loop = asyncio.get_running_loop()
        #     # loop = asyncio.get_event_loop()
        #     return loop.run_until_complete(self.aadd_documents(documents, ids))
        # except RuntimeError:
        #     return self.engine
        return self.run_as_sync(self.aadd_documents(documents, ids))

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
        return self.run_as_sync(self.adelete(ids))

    def run_as_sync(self, coro: Awaitable) -> Any:
        try:
            loop = asyncio.get_running_loop()
            # loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro)
        except RuntimeError:
            return self.engine.run_as_sync(coro)

    @classmethod
    def from_texts(cls):
        pass

    async def __query_collection(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[str] = None,
    ) -> List[Any]:
        k = self.k if self.k else k
        if self.distance_strategy == DistanceStrategy.EUCLIDEAN:
            operator = "<->"
            vector_function = "l2_distance"
        elif self.distance_strategy == DistanceStrategy.COSINE_DISTANCE:
            operator = "<=>"
            vector_function = "cosine_distance"
        else:  # Inner product
            operator = "<#>"
            vector_function = "inner_product"

        filter = f"WHERE {filter}" if filter else ""
        stmt = f"SELECT *, {vector_function}({self.embedding_column}, '{embedding}') as distance FROM {self.table_name} {filter} ORDER BY {self.embedding_column} {operator} '{embedding}' LIMIT {k};"
        results = await self.engine._afetch(stmt)
        return results

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        return self.run_as_sync(
            self.asimilarity_search(query, k=k, filter=filter, **kwargs)
        )

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
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
        k: int = 4,
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
        k: int = 4,
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
        k: int = 4,
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
                if self.store_metadata and row[self.metadata_json_column]
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
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
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
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
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
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        results = await self.__query_collection(
            embedding=embedding, k=fetch_k, filter=filter, **kwargs
        )

        k = self.k if self.k else k
        fetch_k = self.fetch_k if self.fetch_k else fetch_k
        lambda_mult = self.lambda_mult if self.lambda_mult else lambda_mult
        embedding_list = [
            json.loads(row[self.embedding_column]) for row in results
        ]
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
                if self.store_metadata and row[self.metadata_json_column]
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

        return [
            r for i, r in enumerate(documents_with_scores) if i in mmr_selected
        ]

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
