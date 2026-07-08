import json
from .search import search_core
from transformers import pipeline, logging
from .models import MinimalAnswer, StudentSearchResultsAndAnswer

generator = None
file_cache: dict[str, str]= {}
logging.set_verbosity_error() # prevents warning by huggingface from being printed

def get_chunk_data(file_path: str, first_character_index: int, last_character_index: int) -> str:

    if file_path in file_cache:
        file_contents = file_cache[file_path]
    else:
        with open(file_path, 'r') as f:
            file_contents = f.read()
        file_cache[file_path] = file_contents

    chunk_content = file_contents[first_character_index:last_character_index]
    return chunk_content

def extract_generated_str(generated_text: str) -> str:
    idx = generated_text.find("</think>")

    if idx == -1:
        return generated_text
    
    return generated_text[idx + len("</think>"):].strip()

def answer_core(query: str, k: int) -> MinimalAnswer:
    global generator

    if not generator:
        generator = pipeline("text-generation", model="Qwen/Qwen3-0.6B")

    min_search_results = search_core(query, k, "single_query")

    context_chunks : list[str]= []
    for source in min_search_results.retrieved_sources:
        chunk_content = get_chunk_data(source.file_path, source.first_character_index, source.last_character_index)
        context_chunks.append(chunk_content)
    
    prompt_messages = [
        {
            "role": "system",
            "content": "Answer the question based only on the provided context. Be concise. /no_think",
        },
        {
            "role": "user",
            "content": f"Context:\n{''.join(context_chunks)}\n\nQuestion: {query}",
        },
    ]

    result = generator(prompt_messages, max_new_tokens=42)
    generated_text = extract_generated_str(result[0]['generated_text'][-1]["content"])
    # generated_text = "temp texts"

    return MinimalAnswer(
        **min_search_results.model_dump(),
        answer = generated_text
    )

def answer_main(query: str, k: int = 5) -> StudentSearchResultsAndAnswer | None:
    try:
        print("------------ Answer Output ------------")
        min_answer = answer_core(query, k)
        student_search_and_answer = StudentSearchResultsAndAnswer(
            search_results = [min_answer],
            k = k
        )

        print(json.dumps(student_search_and_answer.model_dump(), indent=2))

        return student_search_and_answer

    except Exception as e:
        print(f"Error: {e}")
        return None
