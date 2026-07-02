import os

repo_dir = "test/"
extensions = {"py", "md", "txt", "cu", "cuh", "h", "hpp", "cpp", "jinja"}

def process_normal_file(file_path, max_chunk_size):
    with open(file_path, 'r') as f:
        file_contents = f.read()

def process_python_file(file_path, max_chunk_size):
    with open (file_path, 'r') as f:
        file_contents = f.read()
    sections = file_contents.split("\n\n")
    cur_start = 0
    chunks = []
    for section in sections:
        if len(section) > max_chunk_size:
            
            




def index_main(max_chunk_size = 2000):
    try:
        for root, dirs, files in os.walk(repo_dir):
            for file in files:
                extension = file.split(".")[-1]
                file_path = root + file
                if extension in extensions:
                    if extension == "py":
                        process_python_file(file_path, max_chunk_size)
                    else:
                        process_normal_file(file_path, max_chunk_size)
    except Exception as e:
        print(f"Error processing files: {e}")