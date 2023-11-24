import itertools
from statistics import mean

from sentence_transformers import SentenceTransformer, util

from scraper import get_site_text, get_company_url
from store.base import Store, StoredValue
from store.disk import OnDiskStore

model_name = 'allenai-specter'
data_dir = 'cos_data'

model_cache = {}


def get_model(name: str):
    return model_cache.setdefault(name, SentenceTransformer(name))


def populate_company_data(name: str, store: Store) -> StoredValue:
    name = name.lower()  # To be sure
    stored = store.ensure_name(name)

    url = stored.url
    if stored.url is None:
        url = get_company_url(name)
        if not url:
            raise RuntimeError(f"Failed to get url for: ${name}")
        stored = store.save_url(name, url)

    if stored.text is None:
        stored = store.save_text(name, get_site_text(url))

    if stored.embedding is None:
        stored = store.save_embedding(name, get_model(model_name).encode(stored.text, convert_to_tensor=True))

    return stored


def get_match_score(names: list[str], store: Store):
    stored_values = []
    lower_names = [x.lower() for x in names]

    for name in lower_names:
        stored_values.append(populate_company_data(name, store))

    combinations = itertools.combinations(stored_values, 2)

    rv = []
    for combo in combinations:
        rv.append({
            "url1": combo[0].url,
            "url2": combo[1].url,
            "value": util.cos_sim(combo[0].embedding, combo[1].embedding).item(),
        })

    return mean([x["value"] for x in rv])


def main():
    store = OnDiskStore(model_name, data_dir)

    print(get_match_score(["Revolut", "Monzo", "BBC", "HSBC", "OpenAI"], store))
    print(get_match_score(["Revolut", "Monzo", "HSBC"], store))
    print(get_match_score(["BBC", "OpenAI"], store))
    print(get_match_score(["Revolut", "Monzo"], store))


if __name__ == "__main__":
    main()
