from .models import MinimalSource, MinimalSearchResults, StudentSearchResults
import bm25s
import json

retriever =  bm25s.BM25.load("data/processed/bm25_index")
with open("data/processed/chunks", "r") as f:
    chunks_data = json.load(f)

# main search logic
def search_core(query: str, k: int, question_id: str) -> MinimalSearchResults:
    if not retriever:
        raise Exception("cant find bm25 index")

    query_tokens = bm25s.tokenize(query)
    results, score = retriever.retrieve(query_tokens, k=k) # results and scores are 2D arr [[0,1,2] ...]
    
    retrieved_sources : list[MinimalSource]= []
    for idx in results[0]:
        retrieved_sources.append(MinimalSource(
            file_path = chunks_data[idx]["file_path"],
            first_character_index = chunks_data[idx]["first_character_index"],
            last_character_index = chunks_data[idx]["last_character_index"]
        ))

    min_search_res = MinimalSearchResults(
        question_id = question_id,
        question_str = query,
        retrieved_sources = retrieved_sources
    )
    return min_search_res


def search_main(query: str, k: int) -> StudentSearchResults | None:
    try:
        if not retriever:
            raise Exception("cant find bm25 index")

        search_res = search_core(query, k, "single_query")

        student_search_res = StudentSearchResults(
            search_results = [search_res],
            k = k
        )

        print("------------ Search Output ------------")
        print(json.dumps(student_search_res.model_dump(), indent=2))

        return student_search_res

    except Exception as e:
        print(f"Error: {e}")
        return None