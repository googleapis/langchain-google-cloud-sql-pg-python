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
3. `Enable the Cloud SQL for PostgreSQL for LangChain - Python API.`_
4. `Setup Authentication.`_

.. _Select or create a Cloud Platform project.: https://console.cloud.google.com/project
.. _Enable billing for your project.: https://cloud.google.com/billing/docs/how-to/modify-project#enable_billing_for_a_project
.. _Enable the Cloud SQL for PostgreSQL for LangChain - Python API.:
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

Python >= 3.7

Mac/Linux
^^^^^^^^^

.. code-block:: console

    python3 -m venv <your-env>
    source <your-env>/bin/activate
    pip install langchain-google-cloud-sql-pg


Windows
^^^^^^^

.. code-block:: console

    py -m venv <your-env>
    .\<your-env>\Scripts\activate
    pip install langchain-google-cloud-sql-pg



Example Usage
-------------

Code samples and snippets live in the `samples/`_ folder.

.. _samples/: https://github.com/googleapis/langchain-google-cloud-sql-pg-python/tree/main/samples


Vector Store Usage
~~~~~~~~~~~~~~~~~~~

Use a Vector Store to store embedded data and perform vector search.

.. code:: python

    from langchain_google_cloud_sql_pg import PostgreSQLEngine, PostgreSQLLoader

    engine = PostgreSQLEngine.from_instance("my-cluster", "region", "my-instance", "my-database")
    vs = CloudSQLVectorStore(engine, "my-table", embedding_service)

See the full `Vector Store`_ tutorial.

.. _`Vector Store`: https://github.com/googleapis/langchain-google-cloud-sql-pg-python/tree/main/samples/notebook/vector_store.ipynb

Document Loader Usage
~~~~~~~~~~~~~~~~~~~~~

Use a document loader to load data as Documents.

.. code:: python

    from langchain_google_cloud_sql_pg import PostgreSQLEngine, PostgreSQLLoader

    engine = PostgreSQLEngine.from_instance("region", "my-instance", "my-database")
    loader = PostgreSQLLoader(
        engine,
        table_name="my-table-name"
    )
    docs = loader.lazy_load()

See the full `Document Loader`_ tutorial.

.. _`Document Loader`: https://github.com/googleapis/langchain-google-cloud-sql-pg-python/tree/main/samples/notebook/document_loader.ipynb

Chat Message History Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use Chat Message History to store messages and provide conversation history to LLMs.

.. code:: python

    from langchain_google_cloud_sql_pg import PostgreSQLEngine, PostgreSQLLoader

    engine = PostgreSQLEngine.from_instance("region", "my-instance", "my-database")
    history = PostgreSQLChatMessageHistory(
        engine,
        session_id="foo",
        table_name="message_store",
    )

See the full `Chat Message History`_ tutorial.

.. _`Chat Message History`: https://github.com/googleapis/langchain-google-cloud-sql-pg-python/tree/main/samples/notebook/chat_message_history.ipynb