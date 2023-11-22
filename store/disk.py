import os

import pickledb
from torch import Tensor, save, load

from store.base import Store, StoredValue


class OnDiskStore(Store):
    def __init__(self, model_name: str):
        self.db = pickledb.load('cos_data/index.db', True)
        self.model_name = model_name

    def _load_embedding(self, location: str):
        with open(location, "rb") as f:
            return load(f)

    def _save_embedding(self, location: str, embedding: Tensor):
        os.makedirs(os.path.dirname(location), exist_ok=True)
        with open(location, "wb") as f:
            return save(embedding, f)

    def _gen_filename(self, url: str):
        return f"cos_data/{self.model_name}/{hash(url)}.pt"

    def get(self, url: str) -> StoredValue | None:
        existing = self.db.get(url)
        if existing:
            embedding = None
            if existing["embedding"] is not None:
                embedding = self._load_embedding(existing["embedding"])
            return StoredValue(url, existing["text"], embedding)
        return None

    def ensure_url(self, url: str) -> StoredValue:
        if existing := self.get(url):
            return existing

        self.db.set(url, {"url": url, "text": None, "embedding": None})
        return StoredValue(url, None, None)

    def save_text(self, url: str, text: str) -> StoredValue:
        existing = self.ensure_url(url)

        existing.text = text
        self.db.set(url, {"url": existing.url, "text": text, "embedding": existing.embedding})
        return existing

    def save_embedding(self, url: str, embedding: Tensor) -> StoredValue:
        existing = self.ensure_url(url)

        existing.embedding = embedding
        fname = self._gen_filename(url)
        self._save_embedding(fname, embedding)

        self.db.set(url, {"url": existing.url, "text": existing.text, "embedding": fname})
        return existing
