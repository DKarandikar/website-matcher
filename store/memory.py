from torch import Tensor

from store.base import Store, StoredValue, empty


class InMemoryStore(Store):
    def __init__(self):
        self.store: dict[str, StoredValue] = {}

    def ensure_name(self, name: str) -> StoredValue:
        return self.store.setdefault(name, empty(name))

    def get(self, name: str) -> StoredValue | None:
        return self.store.get(name)

    def save_url(self, name: str, url: str) -> StoredValue:
        self.store.setdefault(name, empty(name)).url = url
        return self.store[name]

    def save_text(self, name: str, text: str) -> StoredValue:
        self.store.setdefault(name, empty(name)).text = text
        return self.store[name]

    def save_embedding(self, name: str, embedding: Tensor) -> StoredValue:
        self.store.setdefault(name, empty(name)).embedding = embedding
        return self.store[name]
