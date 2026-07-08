from .models import MinimalSource, MinimalSearchResults, StudentSearchResults
import json
from .index import collection
import bm25s
from typing import Any
from .classes.configs import configs

retriever: bm25s.BM25 | None = None
chunks_data: list[dict[str, Any]] | None = None

# main search logic
def search_bm25(query: str, k: int, question_id: str) -> list[MinimalSource]:    
    global retriever, chunks_data

    #load retriever and chunks_data if not none exists yet
    if not retriever or not chunks_data:
        retriever =  bm25s.BM25.load("data/processed/bm25_index")
        with open("data/processed/chunks", "r") as f:
            chunks_data = json.load(f)
    if not retriever or not chunks_data:
        raise Exception("Failed to load index or chunks data")

    query_tokens = bm25s.tokenize(query)
    results, score = retriever.retrieve(query_tokens, k=k) # results and scores are 2D arr [[0,1,2] ...]
    
    retrieved_sources : list[MinimalSource]= []
    for idx in results[0]:
        retrieved_sources.append(MinimalSource(
            file_path= chunks_data[idx]["file_path"], # type: ignore[reportUnknownArgumentType]
            first_character_index = chunks_data[idx]["first_character_index"], # type: ignore[reportUnknownArgumentType]
            last_character_index = chunks_data[idx]["last_character_index"] # type: ignore[reportUnknownArgumentType]
        ))
    return retrieved_sources

def search_chromadb(query: str, k: int, question_id: str) -> list[MinimalSource]:
    if not collection:
        raise Exception("Chromadb collection is not initialized")
    
    results = collection.query(
        query_texts = [query],
        n_results = k
    )

    if not results or not results["metadatas"]:
        raise Exception("No results found in the collection")

    retrieved_sources : list[MinimalSource]= []
    for metadata in results["metadatas"][0]:
        retrieved_sources.append(MinimalSource(
            file_path= metadata["file_path"], # type: ignore[reportUnknownArgumentType]
            first_character_index = metadata["first_character_index"], # type: ignore[reportUnknownArgumentType]
            last_character_index = metadata["last_character_index"] # type: ignore[reportUnknownArgumentType]
        ))
    return retrieved_sources

def search_core(query: str, k: int, question_id: str) -> MinimalSearchResults:
    if configs.RETRIEVAL_METHOD == "bm25":
        retrieved_sources = search_bm25(query, k, question_id)
    elif configs.RETRIEVAL_METHOD == "chromadb":
        retrieved_sources = search_chromadb(query, k, question_id)
    else:
        raise Exception(f"Invalid search strategy: {configs.RETRIEVAL_METHOD}")

    min_search_res = MinimalSearchResults(
        question_id = question_id,
        question_str = query,
        retrieved_sources = retrieved_sources
    )
    return min_search_res


def search_main(query: str, k: int) -> StudentSearchResults | None:
    try:
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