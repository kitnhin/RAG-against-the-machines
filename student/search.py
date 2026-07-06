import os
from .models import MinimalSource, MinimalSearchResults, StudentSearchResults
import bm25s
import json
from tqdm import tqdm

def search_main(query: str, k: int, verbose=True) -> MinimalSearchResults:
    try:
        retriever =  bm25s.BM25.load("data/processed/bm25_index")

        with open("data/processed/chunks", "r") as f:
            chunks_data = json.load(f)

        query_tokens = bm25s.tokenize(query)
        results, score = retriever.retrieve(query_tokens, k=k) # results and scores are 2D arr [[0,1,2] ...]
        
        retrieved_sources : list[MinimalSource]= []
        for idx in results[0]:
            retrieved_sources.append(MinimalSource(
                file_path = chunks_data[idx]["file_path"],
                first_character_index = chunks_data[idx]["first_character_index"],
                last_character_index = chunks_data[idx]["last_character_index"]
            ))

        search_res = MinimalSearchResults(
            question_id = "single_query",
            question_str = query,
            retrieved_sources = retrieved_sources
        )

        student_search_res = StudentSearchResults(
            search_results = [search_res],
            k = k
        )

        if verbose:
            print("------------ Search Output ------------")
            print(json.dumps(student_search_res.model_dump(), indent=2))

        return student_search_res

    except Exception as e:
        print(f"Error: {e}")