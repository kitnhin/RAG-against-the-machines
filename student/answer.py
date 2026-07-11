import json
from .search import search_core
from transformers import logging
from .models import MinimalAnswer, StudentSearchResultsAndAnswer
from .classes.configs import configs
from .classes.caches import caches


logging.set_verbosity_error() # prevents warning by huggingface from being printed

def get_chunk_data(file_path: str, first_character_index: int, last_character_index: int) -> str:

    if file_path in caches.file_cache:
        file_contents = caches.file_cache[file_path]
    else:
        with open(file_path, 'r') as f:
            file_contents = f.read()
        caches.file_cache[file_path] = file_contents

    chunk_content = file_contents[first_character_index:last_character_index]
    return chunk_content

def extract_generated_str(generated_text: str) -> str:
    idx = generated_text.find("</think>")

    if idx == -1:
        return generated_text
    
    return generated_text[idx + len("</think>"):].strip()

def answer_core(query: str, k: int) -> MinimalAnswer:

    if query in caches.answer_cache:
        return caches.answer_cache[query]

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

    model_client = caches.get_model_client()
    response = model_client.chat.completions.create(
        model=configs.MODEL,
        messages=prompt_messages, #type: ignore
    )
    answer = response.choices[0].message.content

    min_ans = MinimalAnswer(
        **(min_search_results.model_dump()), #model dump converts pydantic to dict, **unpacks the dict into fields
        answer = answer #type: ignore
    )

    caches.answer_cache[query] = min_ans
    return min_ans

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
