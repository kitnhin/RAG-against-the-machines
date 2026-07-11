import bm25s
from typing import Any
from .configs import configs
from openai import OpenAI
import json
import chromadb

from student.models import MinimalAnswer

class caches:
    _instance = None

    #lazy loaded:
    _model_client: Any = None

    #search
    _bm25_retriever: bm25s.BM25 | None = None
    _chunks_data: list[dict[str, Any]] | None = None
    _chromadb_client: Any = None

    #access directly:
    #answer
    file_cache: dict[str, str]= {}
    answer_cache: dict[str, MinimalAnswer]= {}

    def __new__(cls):
        if cls._instance is None:
            return super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_model_client(cls) -> Any:
        if not cls._model_client:
            cls._model_client = OpenAI(
                base_url=configs.BASE_URL,
                api_key=configs.API_KEY
            )
        return cls._model_client
    
    @classmethod
    def get_bm25_retriever(cls, type: str) -> bm25s.BM25 | None:
        if not cls._bm25_retriever:
            if configs.SPLIT_CHUNKS == True:
                if type == "docs":
                    cls._bm25_retriever =  bm25s.BM25.load("data/processed/bm25_docs_index")
                elif type == "code":
                    cls._bm25_retriever =  bm25s.BM25.load("data/processed/bm25_code_index")
                else:
                    cls._bm25_retriever =  bm25s.BM25.load("data/processed/bm25_all_index")
            else:
                cls._bm25_retriever =  bm25s.BM25.load("data/processed/bm25_all_index")
        return cls._bm25_retriever

    @classmethod
    def get_chunks_data(cls, type: str) -> list[dict[str, Any]] | None:
        if not cls._chunks_data:
            if configs.SPLIT_CHUNKS == True:
                if type == "docs":
                    cls._chunks_data = json.load(open("data/processed/docs_chunks", "r"))
                elif type == "code":
                    cls._chunks_data = json.load(open("data/processed/code_chunks", "r"))
                else:
                    cls._chunks_data = json.load(open("data/processed/all_chunks", "r"))
            else:
                cls._chunks_data = json.load(open("data/processed/all_chunks", "r"))
        return cls._chunks_data
    
    @classmethod
    def get_chromadb_client(cls) -> Any:
        if not cls._chromadb_client:
            cls._chromadb_client = chromadb.PersistentClient(path="data/processed/chromadb_index")
        return cls._chromadb_client