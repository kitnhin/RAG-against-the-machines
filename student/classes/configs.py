class configs: #singleton class
    _instance = None

    #main configs
    RETRIEVAL_METHOD: str = "hybrid" # "bm25" or "chromadb" or "hybrid"
    SPLIT_CHUNKS: bool = True #specifies whether to separate "docs" and "code" chunks when searching, or just use all
    MULTI_QUERY: bool = False #rewrites multiple vers of the query then use vers to get chunks
    HYDE: bool = True #Hypothetical Document Embeddings, gives draft ans, then uses that to retrieve chunks

    #index
    REPO_DIR: str = "data/raw/vllm-0.10.1"
    PROCESS_EXTENSIONS: set[str] = {"py", "md", "txt"}

    #Ai configs
    BASE_URL = "http://localhost:11434/v1" 
    API_KEY = "not-needed" #API key securely hidden (in the codebase :D)
    MODEL = "qwen3:0.6b"
    #currently unused:
    EMBEDDING_MODEL = "all-minilm:22m" 
    EMBEDDING_URL = "http://localhost:11434/api/embeddings"

    def __new__(cls):
        if cls._instance is None:
            return super().__new__(cls)
        return cls._instance