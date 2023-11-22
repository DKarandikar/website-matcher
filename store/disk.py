import os

import pickledb
from torch import Tensor, save, load

from store.base import Store, StoredValue


def to_safe_filename(location: str) -> str:
    return "".join([c for c in location if c.isalpha() or c.isdigit() or c==' ']).rstrip()


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

    def _gen_filename(self, url: str):
        return f"{self.data_dir}/{self.model_name}/{to_safe_filename(url)}.pt"

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
