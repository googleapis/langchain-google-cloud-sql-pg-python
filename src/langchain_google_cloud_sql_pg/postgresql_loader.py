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

from __future__ import annotations

import json
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
)

import sqlalchemy
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document

from .engine import PostgresEngine

DEFAULT_CONTENT_COL = "page_content"
DEFAULT_METADATA_COL = "langchain_metadata"


def text_formatter(row, content_columns) -> str:
    return " ".join(str(row[column]) for column in content_columns if column in row)


def csv_formatter(row, content_columns) -> str:
    return ", ".join(str(row[column]) for column in content_columns if column in row)


def yaml_formatter(row, content_columns) -> str:
    return "\n".join(
        f"{column}: {str(row[column])}" for column in content_columns if column in row
    )


def json_formatter(row, content_columns) -> str:
    dictionary = {}
    for column in content_columns:
        if column in row:
            dictionary[column] = row[column]
    return json.dumps(dictionary)


def _parse_doc_from_row(
    content_columns: Iterable[str],
    metadata_columns: Iterable[str],
    row: dict,
    metadata_json_column: Optional[str] = DEFAULT_METADATA_COL,
    formatter: Callable = text_formatter,
) -> Document:
    page_content = formatter(row, content_columns)
    metadata: Dict[str, Any] = {}
    # unnest metadata from langchain_metadata column
    if metadata_json_column and row.get(metadata_json_column):
        for k, v in row[metadata_json_column].items():
            metadata[k] = v
    # load metadata from other columns
    for column in metadata_columns:
        if column in row and column != metadata_json_column:
            metadata[column] = row[column]

    return Document(page_content=page_content, metadata=metadata)


def _parse_row_from_doc(
    doc: Document,
    column_names: Iterable[str],
    content_column: str = DEFAULT_CONTENT_COL,
    metadata_json_column: Optional[str] = DEFAULT_METADATA_COL,
) -> Dict:
    doc_metadata = doc.metadata.copy()
    row: Dict[str, Any] = {content_column: doc.page_content}
    for entry in doc.metadata:
        if entry in column_names:
            row[entry] = doc_metadata[entry]
            del doc_metadata[entry]
    # store extra metadata in langchain_metadata column in json format
    if metadata_json_column:
        row[metadata_json_column] = doc_metadata
    return row


class PostgresLoader(BaseLoader):
    """Load documents from PostgreSQL`.

    Each document represents one row of the result. The `content_columns` are
    written into the `content_columns`of the document. The `metadata_columns` are written
    into the `metadata_columns` of the document. By default, first columns is written into
    the `page_content` and everything else into the `metadata`.
    """

    __create_key = object()

    def __init__(
        self,
        key,
        engine: PostgresEngine,
        query: str,
        content_columns: List[str],
        metadata_columns: List[str],
        formatter: Callable,
        metadata_json_column: Optional[str] = None,
    ) -> None:
        if key != PostgresLoader.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync' methods!"
            )

        self.engine = engine
        self.query = query
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns
        self.formatter = formatter
        self.metadata_json_column = metadata_json_column

    @classmethod
    async def create(
        cls,
        engine: PostgresEngine,
        query: Optional[str] = None,
        table_name: Optional[str] = None,
        content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
        metadata_json_column: Optional[str] = None,
        format: Optional[str] = None,
        formatter: Optional[Callable] = None,
    ):
        """Constructor for PostgresLoader

        Args:
            engine (PostgresEngine):AsyncEngine with pool connection to the postgres database
            query (Optional[str], optional): SQL query. Defaults to None.
            table_name (Optional[str], optional): Name of table to query. Defaults to None.
            content_columns (Optional[List[str]], optional): Column that represent a Document's page_content. Defaults to the first column.
            metadata_columns (Optional[List[str]], optional): Column(s) that represent a Document's metadata.. Defaults to None.
            metadata_json_column (Optional[str], optional): Column to store metadata as JSON. Defaults to "langchain_metadata".
            format (Optional[str], optional): Format of page content (OneOf: text, csv, YAML, JSON). Defaults to 'text'.
            formatter (Optional[Callable], optional): A function to format page content (OneOf: format, formatter). Defaults to None.

        Returns:
            PostgresLoader
        """
        if table_name and query:
            raise ValueError("Only one of 'table_name' or 'query' should be specified.")
        if not table_name and not query:
            raise ValueError(
                "At least one of the parameters 'table_name' or 'query' needs to be provided"
            )
        if format and formatter:
            raise ValueError("Only one of 'format' or 'formatter' should be specified.")

        if format and format not in ["csv", "text", "JSON", "YAML"]:
            raise ValueError("format must be type: 'csv', 'text', 'JSON', 'YAML'")
        if formatter:
            formatter = formatter
        elif format == "csv":
            formatter = csv_formatter
        elif format == "YAML":
            formatter = yaml_formatter
        elif format == "JSON":
            formatter = json_formatter
        else:
            formatter = text_formatter

        if not query:
            query = f'SELECT * FROM "{table_name}"'
        stmt = sqlalchemy.text(query)

        async with engine._engine.connect() as connection:
            result_proxy = await connection.execute(stmt)

            column_names = list(result_proxy.keys())
            # Select content or default to first column
            content_columns = content_columns or [column_names[0]]
            # Select metadata columns
            metadata_columns = metadata_columns or [
                col for col in column_names if col not in content_columns
            ]
            # Check validity of metadata json column
            if metadata_json_column and metadata_json_column not in column_names:
                raise ValueError(
                    f"Column {metadata_json_column} not found in query result {column_names}."
                )
            # use default metadata json column if not specified
            if metadata_json_column and metadata_json_column in column_names:
                metadata_json_column = metadata_json_column
            elif DEFAULT_METADATA_COL in column_names:
                metadata_json_column = DEFAULT_METADATA_COL
            else:
                metadata_json_column = None

            # check validity of other column
            all_names = content_columns + metadata_columns
            for name in all_names:
                if name not in column_names:
                    raise ValueError(
                        f"Column {name} not found in query result {column_names}."
                    )
        return cls(
            cls.__create_key,
            engine,
            query,
            content_columns,
            metadata_columns,
            formatter,
            metadata_json_column,
        )

    @classmethod
    def create_sync(
        cls,
        engine: PostgresEngine,
        query: Optional[str] = None,
        table_name: Optional[str] = None,
        content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
        metadata_json_column: Optional[str] = None,
        format: Optional[str] = None,
        formatter: Optional[Callable] = None,
    ):
        coro = cls.create(
            engine,
            query,
            table_name,
            content_columns,
            metadata_columns,
            metadata_json_column,
            format,
            formatter,
        )
        return engine._run_as_sync(coro)

    async def _collect_async_items(self, docs_generator):
        return [doc async for doc in docs_generator]

    def load(self) -> List[Document]:
        """Load PostgreSQL data into Document objects."""
        documents = self.engine._run_as_sync(
            self._collect_async_items(self.alazy_load())
        )
        return documents

    async def aload(self) -> List[Document]:
        """Load PostgreSQL data into Document objects."""
        return [doc async for doc in self.alazy_load()]

    def lazy_load(self) -> Iterator[Document]:
        """Load PostgreSQL data into Document objects lazily."""
        yield from self.engine._run_as_sync(
            self._collect_async_items(self.alazy_load())
        )

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Load PostgreSQL data into Document objects lazily."""
        stmt = sqlalchemy.text(self.query)
        async with self.engine._engine.connect() as connection:
            result_proxy = await connection.execute(stmt)
            # load document one by one
            while True:
                row = result_proxy.fetchone()
                if not row:
                    break

                row_data = {}
                column_names = self.content_columns + self.metadata_columns
                column_names += (
                    [self.metadata_json_column] if self.metadata_json_column else []
                )
                for column in column_names:
                    value = getattr(row, column)
                    row_data[column] = value

                yield _parse_doc_from_row(
                    self.content_columns,
                    self.metadata_columns,
                    row_data,
                    self.metadata_json_column,
                    self.formatter,
                )


class PostgresDocumentSaver:
    """A class for saving langchain documents into a PostgreSQL database table."""

    __create_key = object()

    def __init__(
        self,
        key,
        engine: PostgresEngine,
        table_name: str,
        content_column: str,
        metadata_columns: List[str] = [],
        metadata_json_column: Optional[str] = None,
    ):
        if key != PostgresDocumentSaver.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync' methods!"
            )
        self.engine = engine
        self.table_name = table_name
        self.content_column = content_column
        self.metadata_columns = metadata_columns
        self.metadata_json_column = metadata_json_column

    @classmethod
    async def create(
        cls,
        engine: PostgresEngine,
        table_name: str,
        content_column: str = DEFAULT_CONTENT_COL,
        metadata_columns: List[str] = [],
        metadata_json_column: Optional[str] = DEFAULT_METADATA_COL,
    ):
        table_schema = await engine._aload_table_schema(table_name)
        column_names = table_schema.columns.keys()
        if content_column not in column_names:
            raise ValueError(f"Content column, {content_column}, does not exist.")

        # Set metadata columns to all columns if not set
        if len(metadata_columns) == 0:
            metadata_columns = [
                column
                for column in column_names
                if column != content_column and column != metadata_json_column
            ]

        # Check and set metadata json column
        for column in metadata_columns:
            if column not in column_names:
                raise ValueError(f"Metadata column, {column}, does not exist.")

        if (
            metadata_json_column
            and metadata_json_column != DEFAULT_METADATA_COL
            and metadata_json_column not in column_names
        ):
            raise ValueError(f"Metadata JSON column, {column}, does not exist.")
        elif metadata_json_column not in column_names:
            metadata_json_column = None

        return cls(
            cls.__create_key,
            engine,
            table_name,
            content_column,
            metadata_columns,
            metadata_json_column,
        )

    @classmethod
    def create_sync(
        cls,
        engine: PostgresEngine,
        table_name: str,
        content_column: str = DEFAULT_CONTENT_COL,
        metadata_columns: List[str] = [],
        metadata_json_column: str = DEFAULT_METADATA_COL,
    ):
        coro = cls.create(
            engine,
            table_name,
            content_column,
            metadata_columns,
            metadata_json_column,
        )
        return engine._run_as_sync(coro)

    async def aadd_documents(self, docs: List[Document]) -> None:
        """
        Save documents in the DocumentSaver table. Document’s metadata is added to columns if found or
        stored in langchain_metadata JSON column.

        Args:
            docs (List[langchain_core.documents.Document]): a list of documents to be saved.
        """

        for doc in docs:
            row = _parse_row_from_doc(
                doc,
                self.metadata_columns,
                self.content_column,
                self.metadata_json_column,
            )
            for key, value in row.items():
                if isinstance(value, dict):
                    row[key] = json.dumps(value)

            # Create list of column names
            insert_stmt = f'INSERT INTO "{self.table_name}"({self.content_column}'
            values_stmt = f"VALUES (:{self.content_column}"

            # Add metadata
            for metadata_column in self.metadata_columns:
                if metadata_column in doc.metadata:
                    insert_stmt += f", {metadata_column}"
                    values_stmt += f", :{metadata_column}"

            # Add JSON column and/or close statement
            insert_stmt += (
                f", {self.metadata_json_column})" if self.metadata_json_column else ")"
            )
            if self.metadata_json_column:
                values_stmt += f", :{self.metadata_json_column})"
            else:
                values_stmt += ")"

            query = insert_stmt + values_stmt
            await self.engine._aexecute(query, row)

    def add_documents(self, docs: List[Document]) -> None:
        self.engine._run_as_sync(self.aadd_documents(docs))

    async def adelete(self, docs: List[Document]) -> None:
        """
        Delete all instances of a document from the DocumentSaver table by matching the entire Document
        object.

        Args:
            docs (List[langchain_core.documents.Document]): a list of documents to be deleted.
        """
        for doc in docs:
            row = _parse_row_from_doc(
                doc,
                self.metadata_columns,
                self.content_column,
                self.metadata_json_column,
            )
            # delete by matching all fields of document
            where_conditions_list = []
            for key, value in row.items():
                if isinstance(value, dict):
                    where_conditions_list.append(
                        f"{key}::jsonb @> '{json.dumps(value)}'::jsonb"
                    )
                else:
                    # Handle simple key-value pairs
                    where_conditions_list.append(f"{key} = :{key}")

            where_conditions = " AND ".join(where_conditions_list)
            stmt = f'DELETE FROM "{self.table_name}" WHERE {where_conditions};'
            values = {}
            for key, value in row.items():
                if type(value) is int:
                    values[key] = str(value)
                else:
                    values[key] = value

            await self.engine._aexecute(stmt, values)

    def delete(self, docs: List[Document]) -> None:
        self.engine._run_as_sync(self.adelete(docs))

    async def _aload_table_schema(self) -> sqlalchemy.Table:
        """
        Load table schema from existing table in PgSQL database.

        Returns:
            (sqlalchemy.Table): The loaded table.
        """
        metadata = sqlalchemy.MetaData()
        async with self.engine._engine.connect() as conn:
            await conn.run_sync(metadata.reflect, only=[self.table_name])

        table = sqlalchemy.Table(self.table_name, metadata)
        # Extract the schema information
        schema = []
        for column in table.columns:
            schema.append(
                {
                    "name": column.name,
                    "type": column.type.python_type,
                    "max_length": getattr(column.type, "length", None),
                    "nullable": not column.nullable,
                }
            )

        return metadata.tables[self.table_name]
