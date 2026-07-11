from .models import MinimalSource
from collections import defaultdict
import os
from .classes.configs import configs
from .classes.caches import caches
import json

def rrf(min_source_lists: list[list[MinimalSource]], k: int) -> list[MinimalSource]:
    scores: dict[tuple[str, int, int], float] = defaultdict(float)
    items: dict[tuple[str, int, int], MinimalSource] = {}
    rrf_k = 60 #smoothing constant, can adjust, 60 default
    for source_list in min_source_lists:
        for rank, source in enumerate(source_list):
            key = (source.file_path, source.first_character_index, source.last_character_index)
            scores[key] += 1 / (rank + rrf_k)  #RRF score
            items[key] = source

    #sort by score and return top k res
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    top_k_sources = [items[key] for key in sorted_keys[:k]]
    return top_k_sources

def rewrite_queries(query: str) -> list[str]:
    cache_path = "data/processed/alt_queries"

    # load existing cache or start empty
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            data: dict[str, list[str]] = json.load(f)
    else:
        data: dict[str, list[str]] = {}

    # return cached result if exists
    if query in data:
        return data[query]

    prompt_messages = [
        {
            "role": "system",
            "content": "Rewrite the given question in two different ways. Output only the two questions, one per line. Nothing else. /no_think",
        },
        {
        "role": "user",
            "content": f"Question: {query}",
        },
    ]

    model_client = caches.get_model_client()
    response = model_client.chat.completions.create(
        model=configs.MODEL,
        messages=prompt_messages,  # type: ignore
    )
    answer = response.choices[0].message.content
    queries = [q.strip() for q in answer.strip().splitlines() if q.strip()]

    queries.insert(0, query) #insert the original query at the start of the list

    #save to cache
    data[query] = queries
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=4)

    return queries
