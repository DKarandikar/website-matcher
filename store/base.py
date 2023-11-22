import dataclasses

from torch import Tensor


@dataclasses.dataclass
class StoredValue:
    url: str
    text: str | None
    embedding: None | Tensor


class Store:
    def get(self, url: str) -> StoredValue | None:
        raise NotImplementedError

    def ensure_url(self, url: str) -> StoredValue:
        raise NotImplementedError

    def save_text(self, url: str, text: str) -> StoredValue:
        raise NotImplementedError

    def save_embedding(self, url: str, embedding: Tensor) -> StoredValue:
        raise NotImplementedError
