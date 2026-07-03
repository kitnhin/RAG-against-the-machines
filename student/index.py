import os
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from .models import Chunk

repo_dir = "test/"
extensions = {"py", "md", "txt", "cu", "cuh", "h", "hpp", "cpp", "jinja"}

def get_content_indexes(chunks: list[str], file_contents: str, chunks_list: list[Chunk], file_path: str) -> None:
    search_start = 0
    for chunk in chunks:
        first_char_index = file_contents.find(chunk, search_start)
        last_char_index = first_char_index + len(chunk) - 1
        search_start = first_char_index + 1
        chunks_list.append(Chunk(
            chunk_id = -1,
            content = chunk,
            file_path = file_path,
            first_character_index = first_char_index,
            last_character_index = last_char_index
        ))

def process_file(file_path: str, max_chunk_size: int, chunks_list: list[Chunk], extension: str) -> None:
    overlap = 40
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
    
def index_main(max_chunk_size: int) -> None:
    try:
        chunks_list = []
        for root, dirs, files in os.walk(repo_dir):
            for file in files:
                extension = file.split(".")[-1]
                file_path = os.path.join(root, file)
                if extension in extensions:
                    process_file(file_path, max_chunk_size, chunks_list, extension)
    except Exception as e:
        print(f"Error: {e}")