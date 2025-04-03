Cloud SQL for PostgreSQL for LangChain
===================================================================

|preview| |pypi| |versions|

- `Client Library Documentation`_
- `Product Documentation`_

The **Cloud SQL for PostgreSQL for LangChain** package provides a first class experience for connecting to
Cloud SQL instances from the LangChain ecosystem while providing the following benefits:

- **Simplified & Secure Connections**: easily and securely create shared connection pools to connect to Google Cloud databases utilizing IAM for authorization and database authentication without needing to manage SSL certificates, configure firewall rules, or enable authorized networks.
- **Improved performance & Simplified management**: use a single-table schema can lead to faster query execution, especially for large collections.
- **Improved metadata handling**: store metadata in columns instead of JSON, resulting in significant performance improvements.
- **Clear separation**: clearly separate table and extension creation, allowing for distinct permissions and streamlined workflows.

.. |preview| image:: https://img.shields.io/badge/support-preview-orange.svg
   :target: https://github.com/googleapis/google-cloud-python/blob/main/README.rst#stability-levels
.. |pypi| image:: https://img.shields.io/pypi/v/langchain-google-cloud-sql-pg.svg
   :target: https://pypi.org/project/langchain-google-cloud-sql-pg/
.. |versions| image:: https://img.shields.io/pypi/pyversions/langchain-google-cloud-sql-pg.svg
   :target: https://pypi.org/project/langchain-google-cloud-sql-pg/
.. _Client Library Documentation: https://cloud.google.com/python/docs/reference/langchain-google-cloud-sql-pg/latest
.. _Product Documentation: https://cloud.google.com/sql/docs

Quick Start
-----------

In order to use this library, you first need to go through the following steps:

1. `Select or create a Cloud Platform project.`_
2. `Enable billing for your project.`_
3. `Enable the Cloud SQL Admin API.`_
4. `Setup Authentication.`_

.. _Select or create a Cloud Platform project.: https://console.cloud.google.com/project
.. _Enable billing for your project.: https://cloud.google.com/billing/docs/how-to/modify-project#enable_billing_for_a_project
.. _Enable the Cloud SQL Admin API.:
.. _Setup Authentication.: https://googleapis.dev/python/google-api-core/latest/auth.html

Installation
~~~~~~~~~~~~

Install this library in a virtual environment using `venv`_. `venv`_ is a tool that
creates isolated Python environments. These isolated environments can have separate
versions of Python packages, which allows you to isolate one project's dependencies
from the dependencies of other projects.

With `venv`_, it's possible to install this library without needing system
install permissions, and without clashing with the installed system
dependencies.

.. _`venv`: https://docs.python.org/3/library/venv.html


Supported Python Versions
^^^^^^^^^^^^^^^^^^^^^^^^^

Python >= 3.9

Mac/Linux
^^^^^^^^^

.. code-block:: console

    pip install virtualenv
    virtualenv <your-env>
    source <your-env>/bin/activate
    <your-env>/bin/pip install langchain-google-cloud-sql-pg


Windows
^^^^^^^

.. code-block:: console

    pip install virtualenv
    virtualenv <your-env>
    <your-env>\Scripts\activate
    <your-env>\Scripts\pip.exe install langchain-google-cloud-sql-pg


Example Usage
-------------

Code samples and snippets live in the `samples/`_ folder.

.. _samples/: https://github.com/googleapis/langchain-google-cloud-sql-pg-python/tree/main/samples


Vector Store Usage
~~~~~~~~~~~~~~~~~~~

Use a Vector Store to store embedded data and perform vector search.

.. code-block:: python

        from langchain_google_cloud_sql_pg import PostgresVectorstore, PostgresEngine
        from langchain.embeddings import VertexAIEmbeddings


        engine = PostgresEngine.from_instance("project-id", "region", "my-instance", "my-database")
        engine.init_vectorstore_table(
            table_name="my-table",
            vector_size=768,  # Vector size for `VertexAIEmbeddings()`
        )
        embeddings_service = VertexAIEmbeddings(model_name="textembedding-gecko@003")
        vectorstore = PostgresVectorStore.create_sync(
            engine,
            table_name="my-table",
            embeddings=embedding_service
        )

See the full `Vector Store`_ tutorial.

.. _`Vector Store`: https://github.com/googleapis/langchain-google-cloud-sql-pg-python/tree/main/docs/vector_store.ipynb

Document Loader Usage
~~~~~~~~~~~~~~~~~~~~~

Use a document loader to load data as Documents.

.. code-block:: python

        from langchain_google_cloud_sql_pg import PostgresEngine, PostgresLoader


        engine = PostgresEngine.from_instance("project-id", "region", "my-instance", "my-database")
        loader = PostgresSQLLoader.create_sync(
            engine,
            table_name="my-table-name"
        )
        docs = loader.lazy_load()

See the full `Document Loader`_ tutorial.

.. _`Document Loader`: https://github.com/googleapis/langchain-google-cloud-sql-pg-python/tree/main/docs/document_loader.ipynb

Chat Message History Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use Chat Message History to store messages and provide conversation history to LLMs.

.. code-block:: python

        from langchain_google_cloud_sql_pg import PostgresChatMessageHistory, PostgresEngine


        engine = PostgresEngine.from_instance("project-id", "region", "my-instance", "my-database")
        engine.init_chat_history_table(table_name="my-message-store")
        history = PostgresChatMessageHistory.create_sync(
            engine,
            table_name="my-message-store",
            session_id="my-session_id"
        )

See the full `Chat Message History`_ tutorial.

.. _`Chat Message History`: https://github.com/googleapis/langchain-google-cloud-sql-pg-python/tree/main/docs/chat_message_history.ipynb

Langgraph Checkpoint Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``PostgresSaver`` to save snapshots of the graph state at a given point in time.

.. code:: python

   from langchain_google_cloud_sql_pg import PostgresSaver, PostgresEngine
   
   engine = PostgresEngine.from_instance("project-id", "region", "my-instance", "my-database")
   checkpoint = PostgresSaver.create_sync(engine)

See the full `Checkpoint`_ tutorial.

.. _`Checkpoint`: https://github.com/googleapis/langchain-google-cloud-sql-pg-python/blob/main/docs/langgraph_checkpoint.ipynb

Example Usage
-------------

Code examples can be found in the `samples/`_ folder.

.. _samples/: https://github.com/googleapis/langchain-google-cloud-sql-pg-python/tree/main/samples

Converting between Sync & Async Usage
-------------------------------------

Async functionality improves the speed and efficiency of database connections through concurrency,
which is key for providing enterprise quality performance and scaling in GenAI applications. This
package uses a native async Postgres driver, `asyncpg`_, to optimize Python's async functionality.

LangChain supports `async programming`_, since LLM based application utilize many I/O-bound operations,
such as making API calls to language models, databases, or other services. All components should provide
both async and sync versions of all methods.

`asyncio`_ is a Python library used for concurrent programming and is used as the foundation for multiple
Python asynchronous frameworks. asyncio uses `async` / `await` syntax to achieve concurrency for
non-blocking I/O-bound tasks using one thread with cooperative multitasking instead of multi-threading.

.. _`async programming`: https://python.langchain.com/docs/concepts/async/
.. _`asyncio`: https://docs.python.org/3/library/asyncio.html
.. _`asyncpg`: https://github.com/MagicStack/asyncpg

Converting Sync to Async
~~~~~~~~~~~~~~~~~~~~~~~~

Update sync methods to `await` async methods

.. code:: python

   engine = await PostgresEngine.afrom_instance("project-id", "region", "my-instance", "my-database")
   await engine.ainit_vectorstore_table(table_name="my-table", vector_size=768)
   vectorstore = await PostgresVectorStore.create(
      engine,
      table_name="my-table",
      embedding_service=VertexAIEmbeddings(model_name="textembedding-gecko@003")
   )

Run the code: notebooks
^^^^^^^^^^^^^^^^^^^^^^^

ipython and jupyter notebooks support the use of the `await` keyword without any additional setup

Run the code: FastAPI
^^^^^^^^^^^^^^^^^^^^^

Update routes to use `async def`.

.. code:: python

   @app.get("/invoke/")
   async def invoke(query: str):
      return await retriever.ainvoke(query)


Run the code: Local python file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is recommend to create a top-level async method definition: `async def` to wrap multiple async methods.
Then use `asyncio.run()` to run the the top-level entrypoint, e.g. "main()"

.. code:: python

   async def main():
      response = await retriever.ainvoke(query)
      print(response)

   asyncio.run(main())


Contributions
-------------

Contributions to this library are always welcome and highly encouraged.

See `CONTRIBUTING`_ for more information how to get started.

Please note that this project is released with a Contributor Code of Conduct. By participating in
this project you agree to abide by its terms. See `Code of Conduct`_ for more
information.

.. _`CONTRIBUTING`: https://github.com/googleapis/langchain-google-cloud-sql-pg-python/tree/main/CONTRIBUTING.md
.. _`Code of Conduct`: https://github.com/googleapis/langchain-google-cloud-sql-pg-python/tree/main/CODE_OF_CONDUCT.md

License
-------

Apache 2.0 - See
`LICENSE <https://github.com/googleapis/langchain-google-cloud-sql-pg-python/tree/main/LICENSE>`_
for more information.

Disclaimer
----------

This is not an officially supported Google product.
