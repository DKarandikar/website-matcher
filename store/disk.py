import os

import pickledb
from torch import Tensor, save, load

from store.base import Store, StoredValue, empty


def to_safe_filename(location: str) -> str:
    return "".join([c for c in location if c.isalpha() or c.isdigit() or c==' ']).rstrip()


def to_db_repr(existing: StoredValue) -> dict[str, str | None]:
    return {"name": existing.name, "text": existing.text, "url": existing.url, "embedding": existing.embedding}


class OnDiskStore(Store):
    def __init__(self, model_name: str, data_dir: str):
        self.data_dir = data_dir
        self.model_name = model_name

        os.makedirs(os.path.dirname(f'{data_dir}/index.db'), exist_ok=True)
        self.db = pickledb.load(f'{data_dir}/index.db', True)

    def _load_embedding(self, location: str):
        with open(location, "rb") as f:
            return load(f)

    def _save_embedding(self, location: str, embedding: Tensor):
        os.makedirs(os.path.dirname(location), exist_ok=True)
        with open(location, "wb") as f:
            return save(embedding, f)

    def _gen_filename(self, name: str):
        return f"{self.data_dir}/{self.model_name}/{to_safe_filename(name)}.pt"

    def get(self, name: str) -> StoredValue | None:
        existing = self.db.get(name)
        if existing:
            embedding = None
            if existing["embedding"] is not None:
                embedding = self._load_embedding(existing["embedding"])
            return StoredValue(name, existing["url"], existing["text"], embedding)
        return None

    def ensure_name(self, name: str) -> StoredValue:
        if existing := self.get(name):
            return existing

        self.db.set(name, {"name": name, "url": None, "text": None, "embedding": None})
        return empty(name)

    def save_url(self, name: str, url: str) -> StoredValue:
        existing = self.ensure_name(name)

        existing.url = url
        self.db.set(name, {**to_db_repr(existing), "url": url})
        return existing

    def save_text(self, name: str, text: str) -> StoredValue:
        existing = self.ensure_name(name)

        existing.text = text
        self.db.set(name, {**to_db_repr(existing), "text": text})
        return existing

    def save_embedding(self, name: str, embedding: Tensor) -> StoredValue:
        existing = self.ensure_name(name)

        existing.embedding = embedding
        fname = self._gen_filename(name)
        self._save_embedding(fname, embedding)

        self.db.set(name, {**to_db_repr(existing), "embedding": fname})
        return existing
