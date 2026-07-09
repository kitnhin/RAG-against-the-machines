import os
from .answer import answer_core
from .models import MinimalAnswer, StudentSearchResultsAndAnswer
import json
from tqdm import tqdm

def answer_dataset_main(dataset_path: str, k: int, save_directory: str) -> StudentSearchResultsAndAnswer | None:
    try:
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
        
        all_search_and_ans : list[MinimalAnswer]= []

        for question in tqdm(dataset["rag_questions"], desc="Searching dataset"):
            query = question["question"]
            min_search_and_ans = answer_core(query, k)
            all_search_and_ans.append(min_search_and_ans)

        student_search_and_ans = StudentSearchResultsAndAnswer(
            search_results = all_search_and_ans,
            k = k
        )

        # output results
        dataset_filename = os.path.basename(dataset_path)
        output_path = os.path.join(save_directory, dataset_filename)
        os.makedirs(save_directory, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(student_search_and_ans.model_dump(), f, indent=2)

        # print("------------ Search Output ------------")
        # print(json.dumps(student_search_and_ans.model_dump(), indent=2))

        return student_search_and_ans

    except Exception as e:
        print(f"Error: {e}")
        return None