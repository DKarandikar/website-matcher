import dataclasses

from torch import Tensor


@dataclasses.dataclass
class StoredValue:
    name: str
    url: str | None
    text: str | None
    embedding: Tensor | None


def empty(name: str) -> StoredValue:
    return StoredValue(name, None, None, None)


class Store:
    def get(self, name: str) -> StoredValue | None:
        raise NotImplementedError

    def ensure_name(self, name: str) -> StoredValue:
        raise NotImplementedError

    def save_url(self, name: str, url: str) -> StoredValue:
        raise NotImplementedError

    def save_text(self, name: str, text: str) -> StoredValue:
        raise NotImplementedError

    def save_embedding(self, name: str, embedding: Tensor) -> StoredValue:
        raise NotImplementedError
