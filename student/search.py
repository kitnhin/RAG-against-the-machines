from .models import MinimalSource, MinimalSearchResults, StudentSearchResults
import json
import bm25s
# from typing import Any
from .classes.configs import configs
from .classes.caches import caches
from .search_utils import rrf, rewrite_queries, generate_hyde

# main search logic
def search_bm25(query: str, k: int, type: str) -> list[MinimalSource]:    

    #load retriever and chunks_data if not none exists yet
    retriever = caches.get_bm25_retriever(type)
    chunks_data = caches.get_chunks_data(type)

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
    client = caches.get_chromadb_client()

    if configs.SPLIT_CHUNKS == True:
        if type == "docs":
            collection = client.get_or_create_collection(name="docs_chunks") #type: ignore
        elif type == "code":
            collection = client.get_or_create_collection(name="code_chunks") #type: ignore
        else:
            collection = client.get_or_create_collection(name="all_chunks") #type: ignore
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
    def _retrieve_sources(q: str):
        if configs.RETRIEVAL_METHOD == "bm25":
            retrieved_sources = search_bm25(q, k, type)
        elif configs.RETRIEVAL_METHOD == "chromadb":
            retrieved_sources = search_chromadb(q, k, type)
        elif configs.RETRIEVAL_METHOD == "hybrid":
            min_source_lists: list[list[MinimalSource]] = []
            min_source_lists.append(search_bm25(q, k, type))
            min_source_lists.append(search_chromadb(q, k, type))
            retrieved_sources = rrf(min_source_lists, k)
        else:
            raise Exception(f"Invalid search strategy: {configs.RETRIEVAL_METHOD}")
        return retrieved_sources
    
    queries = [query]
    min_source_lists: list[list[MinimalSource]] = []
    
    if configs.MULTI_QUERY:
        queries.extend(rewrite_queries(query))
    
    if configs.HYDE:
        queries = generate_hyde(queries)

    for q in queries:
        min_source_lists.append(_retrieve_sources(q))
    
    retrieved_sources = rrf(min_source_lists, k)

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