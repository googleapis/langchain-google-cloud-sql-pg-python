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

import threading
from typing import Any, Optional

from langgraph.checkpoint.postgres.base import BasePostgresSaver
from langgraph.checkpoint.serde.base import SerializerProtocol

from .async_checkpoint import AsyncAlloyDBPostgresSaver
from langchain_google_alloydb_pg import AlloyDBEngine

class AlloyDBPostgresSaver(BasePostgresSaver):
    lock: threading.Lock
    
    __create_key = object()
    
    def __init__(
        self,
        key: object,
        pool: AlloyDBEngine,
        serde: Optional[SerializerProtocol] = None
    ) -> None:
        if key != AsyncAlloyDBPostgresSaver.__create_key:
            raise Exception(
                "only create class through 'create' or 'create_sync' methods"
            )
        self.pool = pool
        self.lock = threading.Lock()
    
    def setup(self):
        pass
    
    def list(self):
        pass
    
    def get_tuple(self):
        pass
    
    def put(self):
        pass
    
    def put_writes(self):
        pass