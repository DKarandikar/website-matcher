import itertools

from sentence_transformers import SentenceTransformer, util

from scraper import get_site_text
from store.base import Store
from store.disk import OnDiskStore

model_name = 'allenai-specter'

model = SentenceTransformer(model_name)


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
    store = OnDiskStore(model_name)

    urls = ["https://www.revolut.com", "https://www.monzo.com", "https://www.bbc.co.uk"]

    print(get_match_matrix(urls, store))


if __name__ == "__main__":
    main()
