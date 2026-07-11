from .models import MinimalSource
from collections import defaultdict

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