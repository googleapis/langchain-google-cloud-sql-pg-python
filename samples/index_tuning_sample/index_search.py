# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import time

from create_vector_embeddings import (
    DATABASE_NAME,
    INSTANCE_NAME,
    PASSWORD,
    PROJECT_ID,
    REGION,
    USER,
    vector_table_name,
)
from langchain_google_vertexai import VertexAIEmbeddings

from langchain_google_cloud_sql_pg import PostgresEngine, PostgresVectorStore
from langchain_google_cloud_sql_pg.indexes import (
    DistanceStrategy,
    HNSWIndex,
    HNSWQueryOptions,
    IVFFlatIndex,
)

DISTANCE_STRATEGY = DistanceStrategy.COSINE_DISTANCE
k = 10
query_1 = "Brooding aromas of barrel spice."
query_2 = "Aromas include tropical fruit, broom, brimstone and dried herb."
query_3 = "Wine from spain."
query_4 = "Condensed and dark on the bouquet"
query_5 = (
    "Light, fresh and silky—just what might be expected from cool-climate Pinot Noir"
)
queries = [query_1, query_2, query_3, query_4, query_5]


embedding = VertexAIEmbeddings(
    model_name="textembedding-gecko@latest", project=PROJECT_ID
)


async def get_vector_store():
    engine = await PostgresEngine.afrom_instance(
        project_id=PROJECT_ID,
        region=REGION,
        instance=INSTANCE_NAME,
        database=DATABASE_NAME,
        user=USER,
        password=PASSWORD,
    )

    vector_store = await PostgresVectorStore.create(
        engine=engine,
        distance_strategy=DISTANCE_STRATEGY,
        table_name=vector_table_name,
        embedding_service=embedding,
        index_query_options=HNSWQueryOptions(ef_search=256),
    )
    return vector_store


async def query_vector_with_timing(vector_store, query):
    start_time = time.monotonic()  # timer starts
    docs = await vector_store.asimilarity_search(k=k, query=query)
    end_time = time.monotonic()  # timer ends
    latency = end_time - start_time
    return docs, latency


async def hnsw_search(vector_store, knn_docs):
    hnsw_index = HNSWIndex(
        name="hnsw",
        distance_strategy=DISTANCE_STRATEGY,
        m=36,
        ef_construction=96,
    )
    await vector_store.aapply_vector_index(hnsw_index)
    assert await vector_store.is_valid_index(hnsw_index.name)
    print("HNSW index created.")
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


async def ivfflat_search(vector_store, knn_docs):
    ivfflat_index = IVFFlatIndex(name="ivfflat", distance_strategy=DISTANCE_STRATEGY)
    await vector_store.aapply_vector_index(ivfflat_index)
    assert await vector_store.is_valid_index(ivfflat_index.name)
    print("IVFFLAT index created.")
    latencies = []
    recalls = []

    for i in range(len(queries)):
        ivfflat_docs, latency = await query_vector_with_timing(vector_store, queries[i])
        latencies.append(latency)
        recalls.append(calculate_recall(knn_docs[i], ivfflat_docs))

    await vector_store.adrop_vector_index(ivfflat_index.name)
    # calculate average recall & latency
    average_latency = sum(latencies) / len(latencies)
    average_recall = sum(recalls) / len(recalls)
    return average_latency, average_recall


async def knn_search(vector_store):
    latencies = []
    knn_docs = []
    for query in queries:
        docs, latency = await query_vector_with_timing(vector_store, query)
        latencies.append(latency)
        knn_docs.append(docs)
    average_latency = sum(latencies) / len(latencies)
    return knn_docs, average_latency


def calculate_recall(base, target):
    # size of intersection / total number of results
    base = {doc.page_content for doc in base}
    target = {doc.page_content for doc in target}
    return len(base & target) / len(base)


async def main():
    vector_store = await get_vector_store()
    knn_docs, knn_latency = await knn_search(vector_store)
    hnsw_average_latency, hnsw_average_recall = await hnsw_search(
        vector_store, knn_docs
    )
    ivfflat_average_latency, ivfflat_average_recall = await ivfflat_search(
        vector_store, knn_docs
    )

    print(f"KNN recall: 1.0            KNN latency: {knn_latency}")
    print(
        f"HNSW average recall: {hnsw_average_recall}          HNSW average latency: {hnsw_average_latency}"
    )
    print(
        f"IVFFLAT average recall: {ivfflat_average_recall}    IVFFLAT latency: {ivfflat_average_latency}"
    )
    await vector_store._engine.close()
    await vector_store._engine._connector.close()


if __name__ == "__main__":
    asyncio.run(main())
