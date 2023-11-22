import os
import dataclasses
import itertools
from urllib.request import urlopen

import pickledb
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from torch import Tensor, save, load

from store.base import Store, StoredValue


class InMemoryStore(Store):
    def __init__(self):
        self.store: dict[str, StoredValue] = {}

    def ensure_url(self, url: str) -> StoredValue:
        return self.store.setdefault(url, StoredValue(url, None, None))

    def get(self, url: str) -> StoredValue | None:
        return self.store.get(url)

    def save_text(self, url: str, text: str) -> StoredValue:
        self.store.setdefault(url, StoredValue(url, None, None)).text = text
        return self.store[url]

    def save_embedding(self, url: str, embedding: Tensor) -> StoredValue:
        self.store.setdefault(url, StoredValue(url, None, None)).embedding = embedding
        return self.store[url]
