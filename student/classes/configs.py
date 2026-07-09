class configs: #singleton class
    _instance = None

    #main configs
    RETRIEVAL_METHOD: str = "bm25" # "bm25" or "chromadb"

    #index
    REPO_DIR: str = "data/raw/vllm-0.10.1"
    PROCESS_EXTENSIONS: set[str] = {"py", "md", "txt"}

    def __new__(cls):
        if cls._instance is None:
            return super().__new__(cls)
        return cls._instance