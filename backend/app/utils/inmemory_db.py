from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class InsertOneResult:
    inserted_id: Any


@dataclass
class FakeCollection:
    docs: List[Dict[str, Any]] = field(default_factory=list)
    _id_counter: int = 1

    async def create_index(self, *args, **kwargs):  # pragma: no cover
        return None

    async def insert_one(self, doc: Dict[str, Any]) -> InsertOneResult:
        if "_id" not in doc:
            doc = dict(doc)
            doc["_id"] = self._id_counter
            self._id_counter += 1
        self.docs.append(doc)
        return InsertOneResult(inserted_id=doc["_id"])

    async def find_one(self, query: Dict[str, Any], sort: Optional[List[Tuple[str, int]]] = None) -> Optional[Dict[str, Any]]:
        matches = [d for d in self.docs if all(d.get(k) == v for k, v in query.items())]
        if not matches:
            return None
        if sort:
            key, direction = sort[0]
            reverse = direction < 0
            matches.sort(key=lambda d: d.get(key) or datetime.min, reverse=reverse)
        return dict(matches[0])

    async def update_one(self, query: Dict[str, Any], update: Dict[str, Any]) -> None:
        for i, d in enumerate(self.docs):
            if all(d.get(k) == v for k, v in query.items()):
                if "$set" in update:
                    self.docs[i] = {**d, **update["$set"]}
                else:
                    self.docs[i] = {**d, **update}
                return


@dataclass
class FakeDB:
    sessions: FakeCollection = field(default_factory=FakeCollection)
    execution_log: FakeCollection = field(default_factory=FakeCollection)

