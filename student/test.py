import chromadb
client = chromadb.PersistentClient(path="data/processed/chromadb_index")
for name in ["all_chunks", "docs_chunks", "code_chunks"]:
    col = client.get_or_create_collection(name=name)
    print(f"{name}: {col.count()} documents")