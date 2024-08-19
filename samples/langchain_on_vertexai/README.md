# LangChain on Vertex AI RAG Templates

[LangChain on Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/reasoning-engine/overview)
is a managed service that helps you to build and deploy LangChain apps to a managed Reasoning Engine runtime.

## Templates

Use the following templates to deploy Retrieval Augmented Generation (RAG) applications with an Cloud SQL for Postgres database.

Description | Sample
----------- | ------
Deploy a pre-built `LangchainAgent` with custom RAG tool | [prebuilt_langchain_agent_template.py](prebuilt_langchain_agent_template.py)
Build and deploy a question-answering RAG application | [retriever_chain_template.py](retriever_chain_template.p)
Build and deploy an Agent with RAG tool and Memory | [retriever_agent_with_history_template.py](retriever_agent_with_history_template.py)

## Before you begin

1. In the Google Cloud console, on the project selector page, select or [create a Google Cloud project](https://cloud.google.com/resource-manager/docs/creating-managing-projects).
1. [Make sure that billing is enabled for your Google Cloud project](https://cloud.google.com/billing/docs/how-to/verify-billing-enabled#console).
1. [Create a Cloud Storage bucket](https://cloud.google.com/storage/docs/creating-buckets).
1. Enable [AI Platform, Cloud SQL Admin, and Service Networking APIs](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com,sqladmin.googleapis.com,servicenetworking.googleapis.com&_ga=2.92928541.1293093187.1719511698-1945987529.1719351858)
1. [Create a Cloud SQL for Postgres instance.](https://cloud.google.com/sql/docs/postgres/create-instance)
1. [Configure Public IP.](https://cloud.google.com/sql/docs/postgres/configure-ip)
1. [Create a database.](https://cloud.google.com/sql/docs/postgres/create-manage-databases)
1. Create a [vector store table](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/blob/main/docs/vector_store.ipynb) and [chat message history table](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/blob/main/docs/chat_message_history.ipynb).
1. Grant IAM permissions, `roles/cloudsql.client`, `roles/aiplatform.user`, and `serviceusage.serviceUsageConsumer` to the AI Platform Reasoning Engine Service Agent service account: `service-PROJECT_NUMBER@gcp-sa-aiplatform-re.iam.gserviceaccount.com` to connect to the Cloud SQL instance.
1. (Optional) [Add an IAM user or service account to a database instance](https://cloud.google.com/sql/docs/postgres/add-manage-iam-users) and
[grant database privileges to the IAM user](https://cloud.google.com/sql/docs/postgres/add-manage-iam-users#grant-db-privileges).
1. Add your project's values to [`config.py`](config.py).
1. Use [`create_embeddings.py`](create_embeddings.py) to add data to your vector store.

Learn more at [Deploying a RAG Application with Cloud SQL for Postgres to LangChain on Vertex AI](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/reasoning-engine/tutorial_cloud_sql_pg_rag_agent.ipynb).


