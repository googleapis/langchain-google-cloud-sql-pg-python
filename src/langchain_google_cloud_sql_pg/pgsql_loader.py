from __future__ import annotations

from typing import List, Optional, Iterator

from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter

# TODO Move to PgSQLEngine Implementation from Vectorstore PR once it is merged
from langchain_google_cloud_sql_pg.pgsql_engine import PgSQLEngine

DEFAULT_METADATA_COL = "langchain_metadata"


class PgSQLLoader(BaseLoader):
    """Load documents from `CloudSQL Postgres`.

    Each document represents one row of the result. The `content_columns` are
    written into the `content_columns`of the document. The `metadata_columns` are written
    into the `metadata_columns` of the document. By default, first columns is written into
    the `page_content` and everything else into the `metadata`.
    """

    def __init__(
        self,
        engine: PgSQLEngine,
        query: str,
        table_name: str,
        *,
        content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
        format: Optional[str] = None,
        read_only: Optional[bool] = None,
        time_out: Optional[int]  = None,

    ) -> None:
        """Initialize CloudSQL Postgres document loader."""

        self.engine = engine
        self.table_name = table_name
        self.query = query
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns
        self.format = format
        self.read_only = read_only
        self.time_out = time_out

    # Partially Implemented
    def load(self) -> List[Document]:
        """Load CloudSQL Postgres data into Document objects."""
        return self.alazy_load()

    # Partially Implemented
    def load_and_split(
        self, text_splitter: Optional[TextSplitter] = None
    ) -> List[Document]:
        """Load Documents and split into chunks. Chunks are returned as Documents.

        Args:
            text_splitter: TextSplitter instance to use for splitting documents.
              Defaults to RecursiveCharacterTextSplitter.

        Returns:
            List of Documents.
        """

        if text_splitter is None:
            _text_splitter: TextSplitter = RecursiveCharacterTextSplitter()
        else:
            _text_splitter = text_splitter
        #docs = list(self.alazy_load())

        #return _text_splitter.split_documents(docs)
        raise NotImplementedError("load_and_split: Method not implemented fully")

    async def alazy_load(self) -> Iterator[Document]:
        """Load CloudSQL Postgres data into Document objects lazily."""
        content_columns = self.content_columns
        metadata_columns = self.metadata_columns

        if self.table_name is None and self.query is None:
            raise ValueError("Need at least one of the parameters table_name or query to be provided")
        if self.table_name is not None and self.query is None:
            self.query = "select * from " + self.table_name

        # [P0] Load documents via query / table - Implementation
        query_results = await self.engine._aexecute_fetch(self.query)
        result_columns = list(query_results[0].keys())

        if content_columns is None and metadata_columns is None:
            content_columns = [result_columns[0]]
            metadata_columns = result_columns[1:]
        elif content_columns is None and metadata_columns:
            content_columns = [col for col in result_columns if col not in metadata_columns]
        elif content_columns and metadata_columns is None:
            metadata_columns = [col for col in result_columns if col not in content_columns]

        for row in query_results:  # for each doc in the response
            try:
                page_content = " ".join(
                    f"{k}: {v}" for k, v in row.items() if k in content_columns
                )
                # TODO Improve this comment and see compatibility with mysql loader implementation
                # If metadata_columns has langchain_metadata json column
                #   Unnest langchain_metadata json column
                #   add that unnested fields to metadata
                #   proceed with remaining columns

                if DEFAULT_METADATA_COL in metadata_columns and isinstance(row[DEFAULT_METADATA_COL], dict):
                    metadata = {k: v for k, v in row[DEFAULT_METADATA_COL].items()}
                    metadata.update(
                        {k: v for k, v in row.items() if k in metadata_columns if k != DEFAULT_METADATA_COL})
                else:
                    metadata = {k: v for k, v in row.items() if k in metadata_columns}

                yield Document(page_content=page_content, metadata=metadata)
            except (
                KeyError
            ) as e:  # either content_columns or metadata_columns is invalid
                raise ValueError(
                    e.args[0], self.query
                )
