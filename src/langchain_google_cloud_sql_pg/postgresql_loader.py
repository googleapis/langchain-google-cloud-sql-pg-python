from __future__ import annotations

import asyncio
import json
from threading import Thread
from typing import List, Optional, Iterator, Iterable, Dict, Any

from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document

# TODO Move to PostgreSQLEngine Implementation from Vectorstore PR once it is merged
# from langchain_google_cloud_sql_pg.pgsql_engine import PostgreSQLEngine
from postgresql_engine import PostgreSQLEngine

DEFAULT_CONTENT_COL = "page_content"
DEFAULT_METADATA_COL = "langchain_metadata"


class PostgreSQLLoader(BaseLoader):
    """Load documents from `CloudSQL Postgres`.

    Each document represents one row of the result. The `content_columns` are
    written into the `content_columns`of the document. The `metadata_columns` are written
    into the `metadata_columns` of the document. By default, first columns is written into
    the `page_content` and everything else into the `metadata`.
    """

    def __init__(
        self,
        engine: PostgreSQLEngine,
        query: str = None,
        table_name: str = None,
        *,
        content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
        format: Optional[str] = None,
        read_only: Optional[bool] = None,
        time_out: Optional[int] = None,
        formatter: Optional = None,
        metadata_json_column: Optional[str] = None,

    ) -> None:
        """Initialize CloudSQL Postgres document loader."""

        self.engine = engine
        self.table_name = table_name
        self.query = query
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns
        self.page_content_format = format
        self.read_only = read_only
        self.time_out = time_out
        self.formatter = formatter
        self.metadata_json_column = (
            metadata_json_column if metadata_json_column else DEFAULT_METADATA_COL
        )

    async def _collect_async_items(self):
        docs = []
        docs_generator = self.alazy_load()
        async for doc in docs_generator:
            docs.append(doc)
        return docs

    def load(self) -> List[Document]:
        """Load CloudSQL Postgres data into Document objects."""
        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        documents = asyncio.run_coroutine_threadsafe(self._collect_async_items(), self._loop).result()

        return documents

    async def alazy_load(self) -> Iterator[Document]:
        """Load CloudSQL Postgres data into Document objects lazily."""
        content_columns = self.content_columns
        metadata_columns = self.metadata_columns

        if self.table_name is None and self.query is None:
            raise ValueError("Need at least one of the parameters table_name or query to be provided")
        if self.table_name is not None and self.query is None:
            self.query = "select * from " + self.table_name

        # [P0] Load documents via query / table - Implementation
        query_results = await self.engine._afetch(self.query)
        if not query_results:
            return

        result_columns = list(query_results[0].keys())
        if content_columns is None and metadata_columns is None:
            content_columns = [result_columns[0]]
            metadata_columns = result_columns[1:]
        elif content_columns is None and metadata_columns:
            # content_columns = [col for col in result_columns if col not in metadata_columns]
            # Following mysql implementation
            content_columns = [result_columns[0]]
        elif content_columns and metadata_columns is None:
            metadata_columns = [col for col in result_columns if col not in content_columns]

        for row in query_results:  # for each doc in the response
            try:
                formatter = self.formatter
                if formatter:
                    page_content_raw = {k:str(v) for k, v in row.items() if k in content_columns}
                    page_content = formatter(**page_content_raw)
                else:
                    if self.page_content_format and self.page_content_format.lower() in ["yaml", "json"]:
                        page_content = "\n".join(
                            f"{k}:{v}" for k, v in row.items() if k in content_columns
                        )
                    else:
                        page_content = " ".join(
                            f"{v}" for k, v in row.items() if k in content_columns
                        )

                # TODO Improve this comment and see compatibility with PgSQL loader implementation
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
                print (e)
                raise ValueError(
                    e.args[0], self.query
                )


def _parse_doc_from_row(
    content_columns: Iterable[str], metadata_columns: Iterable[str], row: Dict
) -> Document:
    page_content = " ".join(
        str(row[column]) for column in content_columns if column in row
    )
    metadata: Dict[str, Any] = {}
    # unnest metadata from langchain_metadata column
    if DEFAULT_METADATA_COL in metadata_columns and row.get(DEFAULT_METADATA_COL):
        for k, v in row[DEFAULT_METADATA_COL].items():
            metadata[k] = v
    # load metadata from other columns
    for column in metadata_columns:
        if column in row and column != DEFAULT_METADATA_COL:
            metadata[column] = row[column]
    return Document(page_content=page_content, metadata=metadata)


def _parse_row_from_doc(column_names: Iterable[str], doc: Document) -> Dict:
    doc_metadata = doc.metadata.copy()
    row: Dict[str, Any] = {"page_content": doc.page_content}
    for entry in doc.metadata:
        if entry in column_names:
            row[entry] = doc_metadata[entry]
            del doc_metadata[entry]
    # store extra metadata in langchain_metadata column in json format
    if DEFAULT_METADATA_COL in column_names and len(doc_metadata) > 0:
        row[DEFAULT_METADATA_COL] = doc_metadata
    return row


class PostgreSQLDocumentSaver:
    """A class for saving langchain documents into a Cloud SQL PgSQL database table."""

    def __init__(
        self,
        engine: PostgreSQLEngine,
        table_name: str,
    ):
        """
        PostgreSQLDocumentSaver allows for saving of langchain documents in dataabase. If the table
        doesn't exists, a table with default schema will be created. The default schema:
            - page_content (type: text)
            - langchain_metadata (type: JSON)

        Args:
          engine: PostgreSQLEngine object to connect to the PgSQL database.
          table_name: The name of table for saving documents.
        """
        self.engine = engine
        self.table_name = table_name

    async def aadd_documents(self, docs: List[Document]) -> None:
        """
        Save documents in the DocumentSaver table. Document’s metadata is added to columns if found or
        stored in langchain_metadata JSON column.

        Args:
            docs (List[langchain_core.documents.Document]): a list of documents to be saved.
        """
        self._table = await self.engine._load_document_table(self.table_name)
        for doc in docs:
            row = _parse_row_from_doc(self._table.columns.keys(), doc)
            # @ TODO check why sqlalchemy insert is not returning the right columns

            values = tuple(
                json.dumps(value) if isinstance(value, dict) else value
                for value in row.values()
            )
            values = values if len(values) > 1 else f"('{values[0]}')"  # Unpack single-element tuple
            stmt = f"INSERT INTO {self.table_name} ({', '.join(row.keys())}) VALUES {values};"
            await self.engine._aexecute(stmt)

    async def adelete(self, docs: List[Document]) -> None:
        """
        Delete all instances of a document from the DocumentSaver table by matching the entire Document
        object.

        Args:
            docs (List[langchain_core.documents.Document]): a list of documents to be deleted.
        """
        for doc in docs:
            row = _parse_row_from_doc(self._table.columns.keys(), doc)
            # delete by matching all fields of document
            where_conditions = []
            for key, value in row.items():
                if isinstance(value, dict):
                    # Handle nested dictionaries using JSON extraction
                    for inner_key, inner_value in value.items():
                        where_conditions.append(f"{key} ->> '{inner_key}' = '{inner_value}'")
                else:
                    # Handle simple key-value pairs
                    where_conditions.append(f"{key} = '{value}'")

            where_conditions = " AND ".join(where_conditions)
            stmt = f"DELETE FROM {self.table_name} WHERE {where_conditions};"

            await self.engine._aexecute(stmt)
