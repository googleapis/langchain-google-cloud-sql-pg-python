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

import enum
from typing import List, Optional


class DistanceStrategy(str, enum.Enum):
    """Enumerator of the Distance strategies."""

    EUCLIDEAN = "l2"
    COSINE = "cosine"
    INNER_PRODUCT = "inner"


DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE


class BruteForce:
    index_type = "knn"

    def __init__(
        self,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
    ):
        self.distance_strategy = distance_strategy


class HNSWIndex:
    index_type = "hnsw"

    def __init__(
        self,
        name: str = "langchainhnsw",
        m: int = 16,
        ef_construction: int = 64,
        partial_indexes: List = [],
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
    ):
        self.name = name
        self.m = m
        self.ef_construction = ef_construction
        self.partial_indexes = partial_indexes
        self.distance_strategy = distance_strategy
        self.query_options = self.QueryOptions()

    def index_options(self) -> str:
        return f"(m = {self.m}, ef_construction = {self.ef_construction})"

    class QueryOptions:
        def __init__(self, ef_search: Optional[int] = None):
            self.ef_search = ef_search


class IVFFlatIndex:
    index_type = "ivfflat"

    def __init__(
        self,
        name: str = "langchainivfflat",
        lists: int = 1,
        partial_indexes: List = [],
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
    ):
        self.name = name
        self.lists = lists
        self.partial_indexes = partial_indexes
        self.distance_strategy = distance_strategy
        self.query_options = self.QueryOptions()

    def index_options(self) -> str:
        return f"(lists = {self.lists})"

    class QueryOptions:
        def __init__(self, probes: Optional[int] = None):
            self.probes = probes
