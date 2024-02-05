from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Iterable, List, Optional, Tuple, Type, Union, dict

import numpy as np
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from .postgresql_engine import PostgreSQLEngine

VST = TypeVar("VST", bound="CloudSQLVectorStore")


class CloudSQLVectorStore(VectorStore):
    """Google Cloud SQL for PostgreSQL vector store.

    To use, you need the following packages installed:
        pgvector-python
        sqlalchemy
    """

    def __init__(
        self,
        engine: PostgreSQLEngine,
        table_name: str,
        # vector_size: int,
        embedding_service: Embeddings,
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: [str, List[str]] = "metadata",
        ignore_metadata_columns: bool = False,
        index_query_options: [
            HNSWIndex.QueryOptions,
            IVFFlatIndex.QueryOptions,
        ] = None,
        # index: [HNSWIndex | IVFFlatIndex | BruteForce] = None,
        distance_strategy="L2",
        overwrite_existing: bool = False,
        # store_metadata: bool = True,
    ):
        """Constructor for CloudSQLVectorStore.

        Args:
            engine (PostgreSQLEngine): AsyncEngine with pool connection to the postgres database. Required.
            embedding_service (Embeddings): Text embedding model to use.
            table_name (str): Name of the existing table or the table to be created.
            content_column (str): Column that represent a Document’s page_content. Defaults to content
            embedding_column (str): Column for embedding vectors.
                              The embedding is generated from the document value. Defaults to embedding
            metadata_columns (List[str]): Column(s) that represent a document's metadata. Defaults to metadata
            ignore_metadata_columns (List[str]): Column(s) to ignore in pre-existing tables for a document’s metadata.
                                     Can not be used with metadata_columns. Defaults to None
            overwrite_existing (bool): Boolean for truncating table before inserting data. Defaults to False
            index_query_options : QueryOptions class with vector search parameters. Defaults to None
            distance_strategy (str):
                Determines the strategy employed for calculating
                the distance between vectors in the embedding space.
                Defaults to EUCLIDEAN_DISTANCE(L2).
                Available options are:
                - COSINE: Measures the similarity between two vectors of an inner
                    product space.
                - EUCLIDEAN_DISTANCE: Computes the Euclidean distance between
                    two vectors. This metric considers the geometric distance in
                    the vector space, and might be more suitable for embeddings
                    that rely on spatial relationships. This is the default behavior.
        """

        self.engine = engine
        self.table_name = table_name
        # self.vector_size = vector_size
        self.embedding_service = embedding_service
        self.embedding_column = embedding_column
        self.content_column = content_column
        self.metadata_columns = metadata_columns
        self.ignore_metadata_columns = ignore_metadata_columns
        self.overwrite_existing = overwrite_existing
        self.index_query_options = index_query_options
        self.store_metadata = store_metadata
        self.distance_strategy = distance_strategy
        # self.index = index
        asyncio.get_running_loop().run_until_complete(self.__post_init__())

    async def __post_init__(self) -> None:
        """Initialize table and validate existing tables"""

        # Check if table exists
        query = f"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = '{self.table_name}');"
        result = await self.engine._aexecute_fetch(query)
        # If table exists
        if result[0]["exists"]:
            # If overwrite_existing is True Truncate the Table
            if self.overwrite_existing:
                query = f"TRUNCATE TABLE {self.table_name} RESET IDENTITY"
                await self.engine._aexecute_update(query)

            # Checking if metadata and ignore_metadata are given together
            if (
                self.metadata_columns is not None
                and self.ignore_metadata_columns is not None
            ):
                raise ValueError(
                    "Both metadata_columns and ignore_metadata_columns have been provided."
                )

            get_name = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{self.table_name}'"
            result = await self.engine._aexecute_fetch(get_name)
            column_name = [col["column_name"] for col in result]
            dtypes = [dtype["data_type"] for dtype in result]

            # Check column names and datatype for embedding column
            if "uuid" not in column_name:
                raise ValueError("Column uuid does not exist")
            if self.content_column not in column_name:
                raise ValueError(f"Column {self.content_column} does not exist")
            if self.embedding_column in column_name:
                if "USER-DEFINED" not in dtypes:
                    raise ValueError(
                        f"Column {self.embedding_column} is not of type vector"
                    )
            else:
                raise ValueError(
                    f"Column {self.embedding_column} does not exist"
                )

            if "metadata" not in column_name:
                raise ValueError("Column metadata does not exist")

            # Check if there are non-nullable columns
            query = f"SELECT column_name FROM information_schema.columns WHERE table_name = '{self.table_name}' AND is_nullable = 'NO';"
            result = await self.engine._aexecute_fetch(query)
            non_nullable_list = [n["column_name"] for n in result]
            exceptions = set(["uuid", f"{self.content_column}"])
            other_values = [
                value for value in non_nullable_list if value not in exceptions
            ]

            if bool(other_values):
                raise ValueError(
                    f"Only uuid and {self.content_column} can be non-nullable"
                )

            # If both metadata and ignore_metadata are given, throw an error
            if (
                self.metadata_columns is not None
                and self.ignore_metadata_columns is not None
            ):
                raise ValueError(
                    "Both metadata_columns and ignore_metadata_columns have been provided."
                )

        else:
            await self.init_vectorstore_table(
                engine=self.engine,
                table_name=self.table_name,
                vector_size=self.vector_size,
                content_column=self.content_column,
                embedding_column=self.embedding_column,
                metadata_columns=self.metadata_columns,
                overwrite_existing=self.overwrite_existing,
                store_metadata=self.store_metadata,
            )

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_service

    async def create_vector_extension(self) -> None:
        """Creates the vector extsion to the specified database."""
        query = "CREATE EXTENSION IF NOT EXISTS vector"
        await self.engine._aexecute_update(query)

    async def init_vectorstore_table(
        self,
        engine: PostgreSQLEngine,
        table_name: str,
        vector_size: int,
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: Optional[str | List[str]] = "metadata",
        overwrite_existing: bool = False,
        store_metadata: bool = True,
    ) -> None:
        """Creating a non-default vectorstore table"""

        # Create vector extension if not exists
        await self.create_vector_extension()

        if overwrite_existing:
            query = f"TRUNCATE TABLE {self.table_name} RESET IDENTITY"
            await engine._aexecute_update(query)

        query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            uuid UUID PRIMARY KEY,
            {content_column} TEXT NOT NULL,
            {embedding_column} vector({vector_size}),
            {metadata_columns} JSON
        );
        """
        await engine._aexecute_update(query)

    # @classmethod
    # async def afrom_embeddings(
    #     cls: CloudSQLVectorStore,
    #     engine: PostgreSQLEngine,
    #     embedding_service: Embeddings,
    #     text_embeddings: List[Tuple[str, List[float]]],
    #     table_name: str,
    #     metadatas: List[dict] = None,
    #     ids: List[int] = None,
    # ) -> CloudSQLVectorStore:
    #     texts = [t[0] for t in text_embeddings]
    #     embeddings = [t[1] for t in text_embeddings]
    #     metadatas = [{} for _ in texts]

    #     table = cls(
    #         engine=engine,
    #         table_name=table_name,
    #         embedding_service=embedding_service,
    #     )

    #     await table.aadd_embeddings(
    #         texts=texts,
    #         engine=engine,
    #         embeddings=embeddings,
    #         metadatas=metadatas,
    #         ids=ids,
    #         table_name=table_name,
    #     )

    #     return table

    # @classmethod
    # async def afrom_documents(
    #     cls: CloudSQLVectorStore,
    #     documents: List[Document],
    #     engine: PostgreSQLEngine,
    #     table_name: str,
    #     embedding_service: Embeddings,
    #     ids: List[int] = None,
    # ) -> CloudSQLVectorStore:
    #     texts = [d.page_content for d in documents]
    #     metadatas = [json.dumps(d.metadata) for d in documents]

    #     embeddings = embedding_service.embed_documents(list(texts))

    #     table = cls(
    #         engine=engine,
    #         embedding_service=embedding_service,
    #         table_name=table_name,
    #     )

    #     await table.aadd_embeddings(
    #         texts=texts,
    #         engine=engine,
    #         embeddings=embeddings,
    #         metadatas=metadatas,
    #         ids=ids,
    #         table_name=table_name,
    #     )

    #     return table

    @classmethod
    async def afrom_texts(
        cls: Type[VST],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        engine: PostgreSQLEngine,
        embedding_service: Embeddings,
        table_name: str,
        **kwargs: Any,
    ) -> VST:
        """Return VectorStore initialized from texts and embeddings."""
        if not metadatas:
            metadatas = [{} for _ in texts]

        documents = []
        for text, meta in zip(texts, metadatas):
            docs = Document(page_content=text, metadata=meta)
            documents.append(docs)

        vs = cls(
            engine=engine,
            documents=documents,
            embedding_service=embedding_service,
            table_name=table_name,
        )
        return await vs.aadd_embeddings(texts, embeddings, metadatas, ids)

    async def aadd_embeddings(
        self,
        texts: Iterable[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        # if ids is None:
        #     ids = [str(uuid.uuid1()) for _ in texts]

        for id, content, embedding, meta in zip(
            ids, texts, embeddings, metadatas
        ):
            data_to_add = {
                "ids": id,
                "content": content,
                "embedding": embedding,
                "metadata": meta,
            }
            stmt = f"INSERT INTO {self.table_name}(uuid, content, embedding, metadata) VALUES (:ids,:content,:embedding,:metadata)"
            await self.engine._aexecute_update(stmt, data_to_add)

        return ids

    # async def aadd_documents(
    #     self, documents: List[Document], ids: List[int] = None, **kwargs: Any
    # ) -> List[str]:
    #     """Run more documents through the embeddings and add to the vectorstore.

    #     Args:
    #         documents (List[Document]): Iterable of Documents to add to the vectorstore.
    #         ids (List[str]): List of id strings. Defaults to None

    #     Returns:
    #         List of ids from adding the texts into the vectorstore.
    #     """

    #     texts = [d.page_content for d in documents]
    #     metadatas = [json.dumps(d.metadata) for d in documents]
    #     embeddings = self.embedding_service.embed_documents(list(texts))

    #     return await self.aadd_embeddings(
    #         texts=texts,
    #         embeddings=embeddings,
    #         metadatas=metadatas,
    #         ids=ids,
    #         engine=self.Engine,
    #         table_name=self.table_name,
    #     )

    # async def aadd_texts(
    #     self,
    #     texts: List[str],
    #     metadatas: List[dict] = None,
    #     ids: List[int] = None,
    # ) -> List[str]:
    #     """Run more texts through the embeddings and add to the vectorstore.

    #     Args:
    #         texts (str): Iterable of strings to add to the vectorstore.
    #         metadatas (List[dict]): Optional list of metadatas associated with the texts. Defaults to None.
    #         ids (List[str]): List of id strings. Defaults to None

    #     Returns:
    #         List of ids from adding the texts into the vectorstore.
    #     """

    #     if not metadatas:
    #         metadata = [{} for _ in texts]

    #     documents = []
    #     for text, meta in zip(texts, metadatas):
    #         docs = Document(page_content=text, metadata=meta)
    #         documents.append(docs)

    #     return await self.aadd_documents(documents=documents, ids=ids)

    async def __query_collection(
        self, embedding: List[float], k: int = 4, filter: str = None
    ) -> List[Any]:
        if filter is not None:
            condition = f"WHERE {filter}"

            query = f"""
            SELECT uuid, {self.content_column}, {self.embedding_column}, metadata,
            l2_distance({self.embedding_column}, '{embedding}') as distance
            FROM {self.table_name} {condition} ORDER BY {self.embedding_column} <-> '{embedding}' LIMIT {k}
            """
        else:
            query = f"""
            SELECT uuid, {self.content_column}, {self.embedding_column}, metadata,
            l2_distance({self.embedding_column}, '{embedding}') as distance
            FROM {self.table_name} ORDER BY {self.embedding_column} <-> '{embedding}' LIMIT {k}
        """
        results = await self.engine._aexecute_fetch(query)

        return results

    async def asimilarity_search(
        self, query: str, k: int = 4, filter: str = None
    ) -> List[Document]:
        embedding = self.embedding_service.embed_query(text=query)

        return await self.asimilarity_search_by_vector(
            embedding=embedding, k=k, filter=filter
        )

    async def asimilarity_search_by_vector(
        self, embedding: List[float], k: int = 4, filter: str = None
    ) -> List[Document]:
        docs_and_scores = await self.asimilarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )

        return [doc for doc, _ in docs_and_scores]

    async def asimilarity_search_with_score(
        self, query: str, k: int = 4, filter: str = None
    ) -> List[Tuple[Document, float]]:
        embedding = self.embedding_service.embed_query(query)
        docs = await self.asimilarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return docs

    async def asimilarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4, filter: str = None
    ) -> List[Tuple[Document, float]]:
        results = await self.__query_collection(
            embedding=embedding, k=k, filter=filter
        )
        documents_with_scores = [
            (
                Document(
                    page_content=i[f"{self.content_column}"],
                    metadata=i["metadata"],
                ),
                i["distance"],
            )
            for i in results
        ]
        return documents_with_scores

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: str = None,
    ) -> List[Document]:
        embedding = await self.embedding_service.embed_query(text=query)

        return self.amax_marginal_relevance_search_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
        )

    async def amax_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: str = None,
    ) -> List[Tuple[Document, float]]:
        results = await self.__query_collection(
            embedding=embedding, k=fetch_k, filter=filter
        )
        embedding_list = [i[f"{self.embedding_column}"] for i in results]

        mmr_selected = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            embedding_list,
            k=k,
            lambda_mult=lambda_mult,
        )

        candidates = [
            (
                Document(
                    page_content=i[f"{self.content_column}"],
                    metadata=i["metadata"],
                ),
                i["distance"],
            )
            for i in results
        ]

        return [r for i, r in enumerate(candidates) if i in mmr_selected]

    async def _acreate_index(
        self, index: Union[HNSWIndex, IVFFlatIndex, BruteForce]
    ):
        if isinstance(index, BruteForce):
            return None

        distance = (
            "l2"
            if self.distance_strategy == "L2"
            else "ip" if distance_strategy == "INNER" else "cosine"
        )
        index_type = "hnsw" if isinstance(index, HNSWIndex()) else "ivfflat"
        if partial_indexes == None:
            condition = ""
        else:
            condition = f"WHERE (partial_indexes)"

        if index_type == "hnsw":
            query = f"CREATE INDEX ON {self.table_name} USING hnsw ({self.embedding_column} vector_{distance}_ops) WITH (m={index.m}, ef_construction={index.ef_construction}) {condition}"
        else:
            query = f"CREATE INDEX ON {self.table_name} USING ivfflat ({self.embedding_column} vector_{distance}_ops) WITH (lists={index.lists}) {condition}"

        await self.engine._aexecute_update(query)

    async def _aindex_query_options(
        self,
        index_query_options: [
            HNSWIndex.QueryOptions | IVFFlatIndex.QueryOptions
        ],
    ):
        if isinstance(index_query_options, HNSWIndex.QueryOptions):
            query_options = index_query_options.ef_search
            query = f"SET hnsw.ef_search = {query_options}"
        else:
            query_options = index_query_options.probes
            query = f"SET ivfflat.probes = {query_options}"

        await self.engine._aexecute_update(query)

    async def areindex(
        self,
        index: Union[HNSWIndex, IVFFlatIndex, BruteForce],
        index_name: Optional[str],
    ):
        if index_name:
            query = f"REINDEX INDEX {index_name}"
            await self.engine._aexecute_update(query)
        else:
            await self._acreate_index(index)

    async def adrop_index(self):
        query = f"SELECT indexname, indexdef FROM pg_indexes WHERE tablename='{self.table_name}'"
        current_index = await self.engine._aexecute_fetch(query)
        index_def = current_index[0]["indexdef"]
        if "hnsw" in index_def or "ivfflat" in index_def:
            current_index = current_index["indexname"]
            query = f"DROP INDEX {current_index}"
            await self.engine._aexecute_update(query)
        else:
            raise ValueError("Cannot drop Index")

    async def aset_index_query_options(
        self, distance_strategy, index_query_options
    ):
        self.distance_strategy = distance_strategy
        self.index_query_options = index_query_options
        await self._aindex_query_options()


class BruteForce:
    def __init__(self, distance_strategy: str = "L2"):
        self.distance_strategy = distance_strategy


class HNSWIndex:
    def __init__(
        self,
        name: str = "LangChainHNSWIndex",
        m: int = 16,
        ef_construction: int = 64,
        partial_indexes: List = [],
        distance_strategy: str = "L2",
    ):
        self.name = name
        self.m = m
        self.ef_construction = ef_construction
        self.partial_indexes = partial_indexes
        self.distance_strategy = distance_strategy

    class QueryOptions:
        def __init__(self, ef_search):
            self.ef_search = ef_search


class IVFFlatIndex:
    def __init__(
        self,
        name: str = "LangChainIVFFlatIndex",
        lists: int = 1,
        partial_indexes: List = [],
        distance_strategy: str = "L2",
    ):
        self.name = name
        self.lists = lists
        self.partial_indexes = partial_indexes
        self.distance_strategy = distance_strategy

    class QueryOptions:
        def __init__(self, probes):
            self.probes = probes
