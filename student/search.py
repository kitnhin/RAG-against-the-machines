import chromadb

from .models import MinimalSource, MinimalSearchResults, StudentSearchResults
import json
import bm25s
from typing import Any
from .classes.configs import configs

retriever: bm25s.BM25 | None = None
chunks_data: list[dict[str, Any]] | None = None

# main search logic
def search_bm25(query: str, k: int, type: str) -> list[MinimalSource]:    
    global retriever, chunks_data

    #load retriever and chunks_data if not none exists yet
    if not retriever or not chunks_data:
        if type == "docs":
            retriever =  bm25s.BM25.load("data/processed/bm25_docs_index")
            chunks_data = json.load(open("data/processed/docs_chunks", "r"))
        elif type == "code":
            retriever =  bm25s.BM25.load("data/processed/bm25_code_index")
            chunks_data = json.load(open("data/processed/code_chunks", "r"))
        else:
            retriever =  bm25s.BM25.load("data/processed/bm25_all_index")
            chunks_data = json.load(open("data/processed/all_chunks", "r"))

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

def search_chromadb(query: str, k: int, type: str) -> list[MinimalSource]:
    client = chromadb.PersistentClient(path="data/processed/chromadb_index")

    if type == "docs":
        collection = client.get_or_create_collection(name="docs_chunks") #type: ignore
    elif type == "code":
        collection = client.get_or_create_collection(name="code_chunks") #type: ignore
    else:
        collection = client.get_or_create_collection(name="all_chunks") #type: ignore
    
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

def search_core(query: str, k: int, question_id: str, type: str = "all") -> MinimalSearchResults:
    if configs.RETRIEVAL_METHOD == "bm25":
        retrieved_sources = search_bm25(query, k, type)
    elif configs.RETRIEVAL_METHOD == "chromadb":
        retrieved_sources = search_chromadb(query, k, type)
    else:
        raise Exception(f"Invalid search strategy: {configs.RETRIEVAL_METHOD}")

    min_search_res = MinimalSearchResults(
        question_id = question_id,
        question_str = query,
        retrieved_sources = retrieved_sources
    )
    return min_search_res


def search_main(query: str, k: int, type: str) -> StudentSearchResults | None:
    try:
        search_res = search_core(query, k, "single_query", type)

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