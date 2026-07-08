import os

from .search import search_core
from .models import MinimalSearchResults, StudentSearchResults
import json
from tqdm import tqdm

def search_dataset_main(dataset_path: str, k: int, save_directory: str, type: str) -> StudentSearchResults | None:
    try:
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
        
        all_search_res : list[MinimalSearchResults]= []
        print("Type", type)

        for question in tqdm(dataset["rag_questions"], desc="Searching dataset"):
            query = question["question"]
            min_search_res = search_core(query, k, question["question_id"], type)
            all_search_res.append(min_search_res)

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
        return None