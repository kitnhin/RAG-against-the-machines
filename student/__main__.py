import fire
from .index import index_main
from .search import search_main
from .search_dataset import search_dataset_main
from .answer import answer_main

class CLI:
    def test(self):
        print("Hello, World!")

    def index(self, max_chunk_size = 2000, overlap = 200):
        index_main(max_chunk_size, overlap)

    def search(self, query: str, k: int = 5):
        search_main(query, k)

    def search_dataset(self, dataset_path: str, k: int = 3, save_directory: str = "data/output/search_results"):
        search_dataset_main(dataset_path, k, save_directory)

    def answer(self, query: str, k: int = 5):
        answer_main(query, k)

if __name__ == "__main__":
    fire.Fire(CLI)