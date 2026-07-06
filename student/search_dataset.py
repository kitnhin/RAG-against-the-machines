import os
from .models import MinimalSource, MinimalSearchResults, StudentSearchResults
import bm25s
import json
from tqdm import tqdm

def search_dataset_main(dataset_path: str, k: int, save_directory: str) -> MinimalSearchResults:
    try:
        retriever =  bm25s.BM25.load("data/processed/bm25_index")
        with open("data/processed/chunks", "r") as f:
            chunks_data = json.load(f)

        with open(dataset_path, "r") as f:
            dataset = json.load(f)
        
        all_search_res : list[MinimalSearchResults]= []

        for question in tqdm(dataset["rag_questions"], desc="Searching dataset"):
            query = question["question"]
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
                question_id = question["question_id"],
                question_str = query,
                retrieved_sources = retrieved_sources
            )

            all_search_res.append(search_res)

        student_search_res = StudentSearchResults(
            search_results = all_search_res,
            k = k
        )

        # output results
        dataset_filename = os.path.basename(dataset_path)
        output_path = os.path.join(save_directory, dataset_filename)
        os.makedirs(save_directory, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(student_search_res.model_dump(), f, indent=2)

        # print("------------ Search Output ------------")
        # print(json.dumps(student_search_res.model_dump(), indent=2))

        return student_search_res

    except Exception as e:
        print(f"Error: {e}")