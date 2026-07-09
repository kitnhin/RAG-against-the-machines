import fire
from .index import index_main
from .search import search_main
from .search_dataset import search_dataset_main
from .answer import answer_main
from .answer_dataset import answer_dataset_main

class CLI:
    def test(self):
        print("Hello, World!")

    def index(self, max_chunk_size: int = 2000, overlap: int = 200):
        index_main(max_chunk_size, overlap)

    def search(self, query: str, k: int = 5, type: str = "docs"):
        search_main(query, k, type)

    def search_dataset(self, dataset_path: str, k: int = 3, save_directory: str = "data/output/search_results", type: str = "docs"):
        search_dataset_main(dataset_path, k, save_directory, type)

    def answer(self, query: str, k: int = 5):
        answer_main(query, k)

    def answer_dataset(self, dataset_path: str, k: int = 3, save_directory: str = "data/output/answer_results"):
        answer_dataset_main(dataset_path, k, save_directory)

if __name__ == "__main__":
    fire.Fire(CLI)