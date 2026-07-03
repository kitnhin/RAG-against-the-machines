import fire
from .index import index_main

class CLI:
    def test(self):
        print("Hello, World!")
    def index(self, max_chunk_size = 100):
        index_main(max_chunk_size)

if __name__ == "__main__":
    fire.Fire(CLI)