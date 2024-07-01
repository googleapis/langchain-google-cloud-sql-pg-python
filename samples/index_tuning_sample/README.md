
# Guide for Index Tuning

## Introduction

This guide walks through the process of fine-tuning your LangChain PostgreSQL index for better vector similarity search results. Every dataset has different data types, distribution, and structure, thus needs its own index configuration for the best indexing performance. Start by assessing how well your index works with your dataset, focusing on recall (accuracy) and latency (speed). Then, experiment with different parameters to see which ones work best for your dataset. We will lead you step by step using a sample code repository.

## Before You Begin

1. Make sure you have a [Google Cloud project](https://cloud.google.com/resource-manager/docs/creating-managing-projects) and [billing is enabled](https://cloud.google.com/billing/docs/how-to/verify-billing-enabled#console).

1. [Install the gcloud CLI](https://cloud.google.com/sdk/docs/install).

1. Set gcloud project:

    ```bash
    gcloud config set project $PROJECT_ID
    ```

1. Enable APIs:

    ```bash
    gcloud services enable sqladmin.googleapis.com
    gcloud services enable compute.googleapis.com
    gcloud services enable cloudresourcemanager.googleapis.com
    ```

1. If you haven't already, create an Cloud SQL instance following this [guide](https://cloud.google.com/sql/docs/postgres/create-instance).

## Step 1: Clone sample code

Run `git clone` command in your local directory to pull the sample code:

```bash
git clone https://github.com/googleapis/langchain-google-cloud-sql-pg-python.git
```

## Step 2: Move into your local sample directory

Move into the sample directory:

```bash
cd samples/index_tuning_sample
```

## Step 3:  Install Dependencies

In your local sample code directory, run this command to install required dependencies:

```bash
pip3 install -r requirements.txt
```

## Step 4: Fill In Database Info

Enter your database information and credentials in `create_vector_embeddings.py`:

```python
PROJECT_ID = ""
REGION = ""
INSTANCE_NAME = ""
DATABASE_NAME = ""
USER = "" # Use your super user `postgres`
PASSWORD = ""
```

## Step 5: Generate Embeddings

If you already have vector embeddings ready in your database, you can skip this step.
Otherwise, create embeddings from the sample dataset `wine_reviews_dataset.csv` by running this command in your sample code directory:

```bash
python3 create_vector_embeddings.py
```

Now your database is populated with 100k vector embeddings.

## Step 6: Learn About Index Benchmarking Metrics

It is time to measure the performance of this index on your dataset. Recall and latency are key metrics for assessing the performance of vector similarity search indexing.

### What is latency and how to measure it?

Latency refers to the time it takes for the system to complete a search query and return results. Lower latency means faster search results and is crucial for user experience.

To measure the query latency of an index, we measure how long it takes to perform a similarity search query against the index. After creating and applying the index, we put a timer around the similarity search statement, with the time difference being the query latency.

### What is recall and how to measure it?

In the context of ANN indexing, index recall is determined by the proportion of accurate results to the total results returned. Accurate results are identified by the overlap between the current index's results and those obtained through brute force iteration (KNN). By dividing the count of accurate results by the total results returned, we obtain the recall of the index, indicating the index's search accuracy.

## Step 7: Index Benchmarking

Run this command to get an index performance report printed out on your terminal:

```bash
python3 index_search.py
```

The sample code tested the recall and latency of both HNSW and IVFFlat on the current dataset. Let us understand how each measurement is implemented, so you could customize your own benchmarking script.

We will calculate the average latency of multiple queries to have a better understanding of the system's performance under different conditions and to reduce the impact of outliers and variability in individual measurements.

1. First we create some sample query vectors. Using more sample queries will increase the benchmark accuracy when calculating average latency:

    ```python
    query_1 = "Brooding aromas of barrel spice."
    query_2 = "Aromas include tropical fruit, broom, brimstone and dried herb."
    query_3 = "Wine from spain."
    query_4 = "dry-farmed vineyard"
    query_5 = "balanced elegance of some kind"
    queries = [query_1, query_2, query_3, query_4, query_5]
    ```

1. Then we do a KNN brute force query to serve as a base mark:

    ```python
    async def knn_search(vector_store):
        latencies = []
        knn_docs = []
        for query in queries:
            docs, latency = await query_vector_with_timing(vector_store, query)
            latencies.append(latency)
            knn_docs.append(docs)
        average_latency = sum(latencies) / len(latencies)
        return knn_docs, average_latency
    ```

1. In order to calculate query latency, we need to set up timers before and after the query is performed. We create a helper function to achieve this purpose:

    ```python
    async def query_vector_with_timing(vector_store, query):
        start_time = time.monotonic()  # timer starts
        docs = await vector_store.asimilarity_search(k=k, query=query)
        end_time = time.monotonic()  # timer ends
        latency = end_time - start_time
        return docs, latency
    ```

1. Next, we create the index we want to use with the Cloud SQL for PostgreSQL for LangChain library, take HNSW as an example. Depending on the dataset size, the index creation can take from minutes to hours. Our sample dataset takes 3 minutes to create an HNSW index:

    ```python
    from langchain_google_cloud_sql_pg.indexes import HNSWIndex

    hnsw_index = HNSWIndex(name="hnsw")
    await vector_store.aapply_vector_index(hnsw_index)
    assert await vector_store.is_valid_index(hnsw_index.name)
    print("HNSW index created.")
    ```

1. Finally we do similarity search on the query vectors, measure the times cost for each search, and calculate the average recall and latency:

    ```python
    latencies = []
    recalls = []
    for i in range(len(queries)):
        hnsw_docs, latency = await query_vector_with_timing(vector_store, queries[i])
        latencies.append(latency)
        recalls.append(calculate_recall(knn_docs[i], hnsw_docs))

    await vector_store.adrop_vector_index(hnsw_index.name)
    # calculate average recall & latency
    average_latency = sum(latencies) / len(latencies)
    average_recall = sum(recalls) / len(recalls)
    return average_latency, average_recall
    ```

## Step 8: Index Tuning

### HNSW

```python
class HNSWIndex(
    name: str = DEFAULT_INDEX_NAME,
    index_type: str = "hnsw",
    # Distance strategy does not affect recall and has minimal little on latency; refer to this guide to learn more https://cloud.google.com/spanner/docs/choose-vector-distance-function
    distance_strategy: DistanceStrategy = lambda : DistanceStrategy.COSINE_DISTANCE,
    partial_indexes: List[str] | None = None,
    m: int = 16,
    ef_construction: int = 64
)

class HNSWQueryOptions(QueryOptions):
    ef_search: int = 40
```

For HNSW index, there are several parameters that impact search recalls and latency:

- `m`: controls the number of bi-directional links created for each element in the space. A higher m value increases the index size but can increase search recall and reduce latency.
- `ef_construction`: indicates how many entry points will be explored when building the HNSW index. Increasing `ef_construction` increases the index construction time and improves the quality of the constructed graph. At a certain threshold, increasing ef_construction does not improve the quality of the index. If your recall is lower than 0.9, try increasing `ef_construction` to improve search accuracy.
- `ef_search`: determines the size of the dynamic candidate list during the search process. Higher `ef_search` increases both recall and latency.

### HNSW Index Tuning

1. Let us try changing several parameters of the HNSW index aiming for a better performance.
Our default values for `m` is 16 and `ef_construction` is 64. Modify your code to increase `m` to 64 and `ef_construciton` to 128. This will increase the node number of the graph and improve the index quality, while also increasing the index creation time:

    ```python
    hnsw_index = HNSWIndex(
            name="hnsw", m=64, ef_construction=128)
    ```

1. Now let us modify the vector store initialization to add a query parameter `ef_search`, which determines the size of the search candidate list, leading to an increase in recall but also a higher search latency:

    ```python
    vector_store = await PostgresVectorStore.create(
        engine=engine,
        table_name=vector_table_name,
        embedding_service=embedding,
        index_query_options=HNSWQueryOptions(ef_search=256),
    )
    ```

1. Re-run this command to see the difference in recall and latency:

    ```bash
    python3 index_search.py
    ```

### IVFFlat

```python
class IVFFlatIndex(
    name: str = DEFAULT_INDEX_NAME,
    index_type: str = "ivfflat",
    distance_strategy: DistanceStrategy = lambda : DistanceStrategy.COSINE_DISTANCE,
    partial_indexes: List[str] | None = None,
    lists: int = 1
)

class IVFFlatQueryOptions(QueryOptions):
    probes: int = 1
```

IVFFlat index-specific parameter:

- `lists`: the number of clusters into which the dataset is divided. Increasing `lists` generally improves recall but may increase latency as well. Tune `lists` to find the balance between recall and latency that is most suitable for your application.
- `probes`: the number of inverted lists (clusters) to examine during similarity search. A higher number of `probes` increases both recall and latency.

### IVF Index Tuning

1. Let us try changing several parameters of the IVF index aiming for a better performance.
Our default values for `lists` is 100. Modify your code to increase `lists` to 200.

    ```python
    ivfflat_index = IVFFlatIndex(name="ivfflat", lists = 200)
    ```

1. Now let us modify the vector store initialization to add a query parameter `probes`, which determines the number of lists searched during query:

    ```python
    vector_store = await PostgresVectorStore.create(
        engine=engine,
        table_name=vector_table_name,
        embedding_service=embedding,
        index_query_options=IVFFLATQueryOptions(probes=50),
    )
    ```

1. Re-run this command to see the difference in recall and latency:

    ```bash
    python3 index_search.py
    ```

### Distance Strategy (Optional)

The default distance strategy for vector store and indexes is `DistanceStrategy.COSINE_DISTANCE`. We recommend you to keep using the default because it is the best strategy for our sample use case. If you wish to [change this parameter](https://cloud.google.com/spanner/docs/choose-vector-distance-function), remember to change both the vector store and the index initilization because the `distance_strategy` parameter needs to match in two places for the index to be used.

### Partial Indexes (Optional)

A partial index involves creating indexes on subsets of the entire dataset based on certain criteria or conditions, rather than indexing the entire dataset at once. This approach can be particularly beneficial in scenarios where searches are often targeted towards specific segments of data.

To create a partial index, first determine the basis on which your dataset will be segmented for partial indexing. This could be based on features like timestamps (e.g., indexing data from different time periods separately), categories (e.g., products in an e-commerce setting), or any division in your dataset.

For each segment identified, create a separate vector index. This process involves selecting a subset of your dataset based on the segmentation criteria and then applying your chosen indexing algorithm (e.g., HNSW, IVFFlat) to this subset. This step is repeated for each segment, resulting in multiple partial indexes.
