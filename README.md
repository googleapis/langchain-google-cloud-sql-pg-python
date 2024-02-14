# Cloud SQL for PostgreSQL for LangChain

This package contains the [LangChain][langchain] integrations for Cloud SQL for PostgreSQL.

> **🧪 Preview:** This feature is covered by the Pre-GA Offerings Terms of the Google Cloud Terms of Service. Please note that pre-GA products and features might have limited support, and changes to pre-GA products and features might not be compatible with other pre-GA versions. For more information, see the [launch stage descriptions](https://cloud.google.com/products#product-launch-stages)

* [Documentation](docs/)
* [API Reference]()

## Getting Started

In order to use this library, you first need to go through the following steps:

1. [Select or create a Cloud Platform project.][project]
2. [Enable billing for your project.][billing]
3. [Enable the Google Cloud SQL API.][api]
4. [Setup Authentication.][auth]

### Installation

Install this library in a [`virtualenv`][venv] using pip. [`virtualenv`][venv] is a tool to
create isolated Python environments. The basic problem it addresses is one of
dependencies and versions, and indirectly permissions.

With [`virtualenv`][venv], it's possible to install this library without needing system
install permissions, and without clashing with the installed system
dependencies.

```bash
pip install virtualenv
virtualenv <your-env>
source <your-env>/bin/activate
<your-env>/bin/pip install langchain-google-cloud-sql-pg
```

## Vector Store Usage

Use a vector store to store embedded data and perform vector search.

```python
from langchain_google_cloud_sql_pg import CloudSQLVectorstore, PostgreSQLEngine
from langchain.embeddings import VertexAIEmbeddings


engine = PostgreSQLEngine.from_instance("region", "my-instance", "my-database")
embeddings_service = VertexAIEmbeddings()
vectorstore = CloudQLVectorStore(
    engine,
    table_name="my-table",
    embeddings=embedding_service
)
```

See the full [Vector Store][vectorstore] tutorial.

## Document Loader Usage

Use a document loader to load data as LangChain `Document`s.

```python
from langchain_google_cloud_sql_pg import PostgreSQLEngine, PostgreSQLLoader


engine = PostgreSQLEngine.from_instance("region", "my-instance", "my-database")
loader = PostgresSQLLoader(
    engine,
    table_name="my-table-name"
)
docs = loader.lazy_load()
```

See the full [Document Loader][loader] tutorial.

## Chat Message History Usage

Use `ChatMessageHistory` to store messages and provide conversation history to LLMs.

```python
from langchain_google_cloud_sql_pg import PostgreSQLChatMessageHistory, PostgreSQLEngine


engine = PostgreSQLEngine.from_instance("region", "my-instance", "my-database")
history = PostgreSQLChatMessageHistory(
    engine,
    table_name="my-message-store",
    session_id="my-session_id"
)
```

See the full [Chat Message History][history] tutorial.

## Contributing

Contributions to this library are always welcome and highly encouraged.

See [CONTRIBUTING](CONTRIBUTING.md) for more information how to get started.

Please note that this project is released with a Contributor Code of Conduct. By participating in
this project you agree to abide by its terms. See [Code of Conduct](CODE_OF_CONDUCT.md) for more
information.

## License

Apache 2.0 - See [LICENSE](LICENSE) for more information.

## Disclaimer

This is not an officially supported Google product.

[langchain]: https://github.com/langchain-ai/langchain
[project]: https://console.cloud.google.com/project
[billing]: https://cloud.google.com/billing/docs/how-to/modify-project#enable_billing_for_a_project
[api]: https://console.cloud.google.com/flows/enableapi?apiid=sqladmin.googleapis.com
[auth]: https://googleapis.dev/python/google-api-core/latest/auth.html
[venv]: https://virtualenv.pypa.io/en/latest/
[vectorstore]: ./docs/vector_store.ipynb
[loader]: ./docs/document_loader.ipynb
[history]: ./docs/chat_message_history.ipynb
