Cloud SQL for PostgreSQL for LangChain
===================================================================

|preview| |pypi| |versions|

- `Client Library Documentation`_
- `Product Documentation`_

.. |preview| image:: https://img.shields.io/badge/support-preview-orange.svg
   :target: https://github.com/googleapis/google-cloud-python/blob/main/README.rst#stability-levels
.. |pypi| image:: https://img.shields.io/pypi/v/langchain-google-cloud-sql-pg.svg
   :target: https://pypi.org/project/langchain-google-cloud-sql-pg/
.. |versions| image:: https://img.shields.io/pypi/pyversions/langchain-google-cloud-sql-pg.svg
   :target: https://pypi.org/project/langchain-google-cloud-sql-pg/
.. _Client Library Documentation: https://github.com/googleapis/langchain-google-cloud-sql-pg-python
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

Python >= 3.8

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

.. code-block::

        from langchain_google_cloud_sql_pg import PostgresVectorstore, PostgresEngine
        from langchain.embeddings import VertexAIEmbeddings


        engine = PostgresEngine.from_instance("project-id", "region", "my-instance", "my-database")
        engine.init_vectorstore_table(
            table_name="my-table",
            vector_size=768,  # Vector size for `VertexAIEmbeddings()`
        )
        embeddings_service = VertexAIEmbeddings()
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

.. code-block::

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

.. code-block::

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

Contributions
~~~~~~~~~~~~~

Contributions to this library are always welcome and highly encouraged.

See `CONTRIBUTING`_ for more information how to get started.

Please note that this project is released with a Contributor Code of Conduct. By participating in
this project you agree to abide by its terms. See `Code of Conduct`_ for more
information.

.. _`CONTRIBUTING`: https://github.com/googleapis/langchain-google-cloud-sql-pg-python/tree/main/CONTRIBUTING.md
.. _`Code of Conduct`: https://github.com/googleapis/langchain-google-cloud-sql-pg-python/tree/main/CODE_OF_CONDUCT.md

Disclaimer
~~~~~~~~~~~

This is not an officially supported Google product.