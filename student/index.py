import os
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from .models import Chunk
import bm25s
import chromadb
import json
from tqdm import tqdm
from .classes.configs import configs

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
    # print_chunk_list(chunks_list, file_path)

def index_chunks_bm25(chunks_list: list[Chunk]) -> None:
    def build_and_save_chunks(chunks_list: list[Chunk], file_path: str) -> None:
        tokens = bm25s.tokenize([chunk.content for chunk in chunks_list])
        retriever = bm25s.BM25()
        retriever.index(tokens)
        retriever.save(file_path)

    build_and_save_chunks([chunk for chunk in chunks_list if chunk.file_path.split(".")[-1] != "py"], "data/processed/bm25_docs_index")
    build_and_save_chunks([chunk for chunk in chunks_list if chunk.file_path.split(".")[-1] == "py"], "data/processed/bm25_code_index")
    build_and_save_chunks(chunks_list, "data/processed/bm25_all_index")



def index_chunks_chromadb(chunks_list: list[Chunk]) -> None:
    client = chromadb.PersistentClient(path="data/processed/chromadb_index")
    docs_collection = client.get_or_create_collection(name="docs_chunks") #type: ignore
    code_collection = client.get_or_create_collection(name="code_chunks") #type: ignore

    docs_chunks = [chunk for chunk in chunks_list if chunk.file_path.split(".")[-1] != "py"]
    code_chunks = [chunk for chunk in chunks_list if chunk.file_path.split(".")[-1] == "py"]

    batch_size = 500 # process in batch cuz chromadb got limit how much can add per .add() call (5461 items)
    for i in tqdm(range(0, len(docs_chunks), batch_size), desc="Indexing doc chunks"):
        batch = docs_chunks[i:i + batch_size]
        docs_collection.add(
            documents = [chunk.content for chunk in batch],
            ids = [str(i) for i in range(len(batch))],
            metadatas = [{
                "file_path": chunk.file_path,
                "first_character_index": chunk.first_character_index,
                "last_character_index": chunk.last_character_index
            } for chunk in batch]
        )
    
    for i in tqdm(range(0, len(code_chunks), batch_size), desc="Indexing code chunks"):
        batch = code_chunks[i:i + batch_size]
        code_collection.add(
            documents = [chunk.content for chunk in batch],
            ids = [str(i) for i in range(len(batch))],
            metadatas = [{
                "file_path": chunk.file_path,
                "first_character_index": chunk.first_character_index,
                "last_character_index": chunk.last_character_index
            } for chunk in batch]
        )

    print("Successfully indexed all chunks to chromadb")

def save_chunks(chunks_list: list[Chunk]) -> None:
    def save_helper(chunks_list: list[Chunk], file_path: str) -> None:
        with open(file_path, "w") as f:
            chunk_dicts = [chunk.model_dump() for chunk in chunks_list]
            json.dump(chunk_dicts, f, indent=4)
    save_helper([chunk for chunk in chunks_list if chunk.file_path.split(".")[-1] != "py"], "data/processed/docs_chunks")
    save_helper([chunk for chunk in chunks_list if chunk.file_path.split(".")[-1] == "py"], "data/processed/code_chunks")
    save_helper(chunks_list, "data/processed/all_chunks")
    print("Succesfully saved chunks to chunks")

    
def index_main(max_chunk_size: int, overlap: int) -> None:
    try:
        global client, collection
        if not os.path.isdir(configs.REPO_DIR):
            raise Exception(f"Index dir {configs.REPO_DIR} doesn't exist")

        chunks_list: list[Chunk] = []
        files_list: list[tuple[str, str]] = []
        for root, dirs, files in os.walk(configs.REPO_DIR):
            for file in files:
                extension = file.split(".")[-1]
                file_path = os.path.join(root, file)
                if extension in configs.PROCESS_EXTENSIONS:
                    files_list.append((file_path, extension))
            
        for file_path, extension in tqdm(files_list, desc="Chunking files"):
            process_file(file_path, max_chunk_size, chunks_list, extension, overlap)

        #index chunk
        if configs.RETRIEVAL_METHOD == "bm25":
            index_chunks_bm25(chunks_list)
        elif configs.RETRIEVAL_METHOD == "chromadb":
            index_chunks_chromadb(chunks_list)
        
        save_chunks(chunks_list)

    except Exception as e:
        print(f"Error: {e}")


def print_chunk_list(chunks_list: list[Chunk], file_path: str) -> None:
    # printing
    print(f"\n======== Chunks for file: {file_path} ========")
    
    chunk_num = 1
    for chunk in chunks_list:
        if chunk.file_path == file_path:
            print(f"CHUNK #{chunk_num}")
            print(f"   Indices: [{chunk.first_character_index} -> {chunk.last_character_index}]")
            print(f"   --- Content ---")
            
            # Indent content with a vertical pipe border to keep it scannable
            indented_content = "\n".join(f"   | {line}" for line in chunk.content.splitlines())
            print(indented_content)
            
            print(f"   {'-' * 40}\n")
            chunk_num += 1
            
    print(f"================================================\n")