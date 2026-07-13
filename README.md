# RAG Against The Machines

A RAG (Retrieval-Augmented Generation) pipeline built over the vLLM codebase as a learning project for 42 School. The goal was to understand how RAG works end-to-end by experimenting with different retrieval strategies and observing their impact on recall.

## Overview

The pipeline takes a natural language question about vLLM, retrieves relevant chunks from the codebase, and generates an answer using a local LLM. The project focused primarily on the retrieval side, experimenting with different methods to maximize Recall@5.

## How It Works

1. **Indexing**: The vLLM codebase is chunked and indexed into BM25 and/or ChromaDB, using LangChain text splitters with python code and text types. (see notes.txt on how this works internally)
2. **Retrieval**: Given a query, relevant chunks are retrieved using BM25 (keyword matching), ChromaDB (semantic similarity), or both fused together with Reciprocal Rank Fusion (RRF), alongside other methods like multiquery or hyde.
3. **Generation**: Retrieved chunks are passed as context to Qwen3-0.6B, which generates an answer

## Retrieval Methods Implemented

**BM25** scores documents by **matching exact keywords** between the query and chunks. It works well when the query and document use similar vocabulary.

**ChromaDB** converts both the query and chunks into **embedding vectors** and finds the closest matches by meaning rather than exact words.

**Hybrid (RRF)** runs both BM25 and ChromaDB, then combines their ranked results using Reciprocal Rank Fusion. Each chunk gets a score based on its rank position across both lists, and the top-k are selected from the merged ranking.

**Multi-Query Rewriting** uses the LLM to generate two alternative phrasings of the original question. All three queries (original + two rewrites) are searched independently, and the results are fused with RRF.

**HyDE (Hypothetical Document Embeddings)** asks the LLM to generate a hypothetical answer to the question, then uses that answer as the search query instead of the original question. The rationale is that a statement query (drafted answer) is closer in embedding space than the original question.

## Results (Recall@5, Docs Only)

| Retrieval Method | All Dataset | Split Dataset | Multi-Query | HyDE |
|------------------|-------------|---------------|-------------|------|
| BM25             | 0.84        | 0.90          | 0.91        | 0.81 |
| ChromaDB         | 0.54        | 0.57          | 0.56        | 0.53 |
| Hybrid           | 0.80        | 0.86          | 0.84        | 0.81 |

## Observations

**BM25 consistently outperformed ChromaDB.** The vLLM documentation uses specific technical terms that match well with keyword-based retrieval. ChromaDB's semantic matching didn't add much value here since the questions and documents already share similar vocabulary.

**Multi-query helped BM25 but hurt ChromaDB.** Different phrasings introduce different keywords, giving BM25 more chances to match. But for ChromaDB, the rewrites from a 0.6B model often drifted in meaning, pulling in irrelevant chunks that diluted the good results.

**HyDE made everything worse.** Qwen3-0.6B doesn't know anything about vLLM, so its hypothetical answers were hallucinated and didn't resemble the actual documents. For BM25, the fake answers introduced irrelevant keywords. For ChromaDB, the embeddings pointed to the wrong neighborhood in vector space. HyDE needs a model that can generate better drafted answers to be effective.

**Hybrid didn't always beat the best individual method.** When BM25 is already strong and ChromaDB is weak, fusing them with equal RRF weights can actually drag BM25's results down. Weighted RRF (giving BM25 higher weight) would likely help but was not implemented.

## Usage

Index the dataset, then search or generate answers. Configuration is done via the Makefile variables and `student/classes/configs.py`.

```bash
# Index the vLLM codebase into chunks
make index

# Search with a single query
make search QUERY="What versioning scheme does vLLM use?" k=10

# Run search over the full evaluation dataset
make search_dataset

# Evaluate search results with the moulinette
make momo_search

# Index, search, and evaluate in one step
make index_and_search

# Generate an answer for a single query
make answer QUERY="What versioning scheme does vLLM use?"

# Generate answers for the full dataset
make answer_dataset

# Clean indexed data or output
make clean_index
make clean_output
```

**Makefile Variables**

| Variable | Default | Description |
|----------|---------|-------------|
| `TYPE` | `docs` | Dataset type to process and evaluate (`docs` or `code`) |
| `MAX_CHUNK_SIZE` | `2000` | Maximum number of characters per chunk |
| `OVERLAP` | `200` | Number of overlapping characters between consecutive chunks |
| `QUERY` | `What versioning scheme does vLLM use?` | Query string for single search or answer commands |
| `k` | `10` | Number of top results to retrieve |
| `THRESHOLD` | `0.8` (docs) / `0.5` (code) | Minimum recall threshold for moulinette evaluation |