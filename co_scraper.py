import os
import dataclasses
import itertools
from urllib.request import urlopen

import pickledb
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from torch import Tensor, save, load

model_name = 'allenai-specter'

model = SentenceTransformer(model_name)


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


class OnDiskStore(Store):
    def __init__(self):
        self.db = pickledb.load('cos_data/index.db', True)

    def _load_embedding(self, location: str):
        with open(location, "rb") as f:
            return load(f)

    def _save_embedding(self, location: str, embedding: Tensor):
        os.makedirs(os.path.dirname(location), exist_ok=True)
        with open(location, "wb") as f:
            return save(embedding, f)

    def _gen_filename(self, url: str):
        return f"cos_data/{model_name}/{hash(url)}.pt"

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


def get_site_text(url: str):
    html = urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text


def get_match_matrix(urls: list[str], store: Store):
    storedValues = []

    for url in urls:
        stored = store.ensure_url(url)

        if stored.text is None:
            stored = store.save_text(url, get_site_text(url))

        if stored.embedding is None:
            stored = store.save_embedding(url, model.encode(stored.text, convert_to_tensor=True))

        storedValues.append(stored)

    combinations = itertools.combinations(storedValues, 2)

    rv = []
    for combo in combinations:
        rv.append({
            "url1": combo[0].url,
            "url2": combo[1].url,
            "value": util.cos_sim(combo[0].embedding, combo[1].embedding),
        })

    return rv


def main():
    store = OnDiskStore()

    urls = ["https://www.revolut.com", "https://www.monzo.com", "https://www.bbc.co.uk"]

    print(get_match_matrix(urls, store))


if __name__ == "__main__":
    main()
