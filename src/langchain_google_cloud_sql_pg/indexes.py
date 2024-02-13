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
from dataclasses import dataclass
from typing import List, Optional


class DistanceStrategy(enum.Enum):
    """Enumerator of the Distance strategies."""

    EUCLIDEAN = "l2"
    COSINE_DISTANCE = "cosine"
    INNER_PRODUCT = "inner"


DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE_DISTANCE
DEFAULT_INDEX_NAME = "langchainvectorindex"


@dataclass
class BaseIndex:
    name: str = DEFAULT_INDEX_NAME
    index_type: str = "base"
    distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY
    partial_indexes: Optional[List] = None


class BruteForce(BaseIndex):
    index_type = "knn"


@dataclass
class HNSWIndex(BaseIndex):
    index_type = "hnsw"
    m: int = 16
    ef_construction: int = 64

    def index_options(self) -> str:
        return f"(m = {self.m}, ef_construction = {self.ef_construction})"


@dataclass
class QueryOptions:
    def to_string(self):
        pass


@dataclass
class HNSWQueryOptions(QueryOptions):
    ef_search: int

    def to_string(self):
        return f"hnsw.ef_search = {self.ef_search}"


@dataclass
class IVFFlatIndex(BaseIndex):
    index_type: str = "ivfflat"
    lists: int = 1

    def index_options(self) -> str:
        return f"(lists = {self.lists})"


@dataclass
class IVFFlatQueryOptions(QueryOptions):
    probes: int

    def to_string(self):
        return f"ivflfat.probes = {self.probes}"
