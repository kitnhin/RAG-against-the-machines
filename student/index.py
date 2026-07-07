import os
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from .models import Chunk
import bm25s
import json
from tqdm import tqdm

repo_dir = "data/raw/vllm-0.10.1"
extensions = {"py", "md", "txt"}

def get_content_indexes(chunks: list[str], file_contents: str, chunks_list: list[Chunk], file_path: str) -> None:
    search_start = 0
    for chunk in chunks:
        first_char_index = file_contents.find(chunk, search_start)
        last_char_index = first_char_index + len(chunk) - 1
        search_start = first_char_index + 1
        chunks_list.append(Chunk(
            content = chunk,
            file_path = file_path,
            first_character_index = first_char_index,
            last_character_index = last_char_index
        ))

def process_file(file_path: str, max_chunk_size: int, chunks_list: list[Chunk], extension: str, overlap: int) -> None:
    with open (file_path, 'r') as f:
        file_contents = f.read()
    if extension == "py":
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=max_chunk_size,
            chunk_overlap=overlap
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=overlap,
        )

    chunks = splitter.split_text(file_contents)
    get_content_indexes(chunks, file_contents, chunks_list, file_path)

    # # printing
    # print(f"\n======== Chunks for file: {file_path} ========")
    
    # chunk_num = 1
    # for chunk in chunks_list:
    #     if chunk.file_path == file_path:
    #         print(f"CHUNK #{chunk_num}")
    #         print(f"   Indices: [{chunk.first_character_index} -> {chunk.last_character_index}]")
    #         print(f"   --- Content ---")
            
    #         # Indent content with a vertical pipe border to keep it scannable
    #         indented_content = "\n".join(f"   | {line}" for line in chunk.content.splitlines())
    #         print(indented_content)
            
    #         print(f"   {'-' * 40}\n")
    #         chunk_num += 1
            
    # print(f"================================================\n")

def index_chunks(chunks_list: list[Chunk]) -> None:
    tokens = bm25s.tokenize([chunk.content for chunk in chunks_list])
    retriever = bm25s.BM25()
    retriever.index(tokens)
    retriever.save("data/processed/bm25_index")
    print("Succesfully indexed chunks tp data/processed/bm25_index")

def save_chunks(chunks_list: list[Chunk]) -> None:
    with open("data/processed/chunks", "w") as f:
        chunk_dicts = [chunk.model_dump() for chunk in tqdm(chunks_list, desc="Saving chunks")]
        json.dump(chunk_dicts, f, indent=4)
    print("Succesfully saved chunks to data/processed/chunks.json")

    
def index_main(max_chunk_size: int, overlap: int) -> None:
    try:
        if not os.path.isdir(repo_dir):
            raise Exception(f"Index dir {repo_dir} doesn't exist")

        chunks_list: list[Chunk] = []
        files_list: list[tuple[str, str]] = []
        for root, dirs, files in os.walk(repo_dir):
            for file in files:
                extension = file.split(".")[-1]
                file_path = os.path.join(root, file)
                if extension in extensions:
                    files_list.append((file_path, extension))
            
        for file_path, extension in tqdm(files_list, desc="Chunking files"):
            process_file(file_path, max_chunk_size, chunks_list, extension, overlap)

        index_chunks(chunks_list)
        save_chunks(chunks_list)

    except Exception as e:
        print(f"Error: {e}")