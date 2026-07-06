import json
from .search import search_main
from transformers import pipeline

def get_chunk_data(file_path, first_character_index, last_character_index):
    with open(file_path, 'r') as f:
        file_contents = f.read()
    chunk_content = file_contents[first_character_index:last_character_index]
    return chunk_content

def answer_main(query: str, k: int = 5):
    try:
        print("------------ Answer Output ------------")
        generator = pipeline("text-generation", model="Qwen/Qwen3-0.6B")

        search_results = search_main(query, k, False)

        context_chunks = []
        for result in search_results.search_results:
            for source in result.retrieved_sources:
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
        print(f"Generating response ...")
        result = generator(prompt_messages, max_new_tokens=42)
        generated_text = result[0]['generated_text'][-1]["content"]
        print(f"Generated Answer: {generated_text}")
        return generated_text


    except Exception as e:
        print(f"Error: {e}")
