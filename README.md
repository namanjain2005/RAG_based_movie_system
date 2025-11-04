# Movie Search System with RAG

This repository contains a movie search system implemented using Retrieval-Augmented Generation (RAG) with various search capabilities.

## CLI Tools

The system provides several command-line interface (CLI) tools for different types of searches and operations:

### Semantic Search
```bash
python cli/semantic_search_cli.py <query>
```
Performs semantic search on the movie dataset using embeddings.

### Keyword Search
```bash
python cli/keyword_search_cli.py <query>
```
Performs traditional keyword-based search on the movie dataset.

### Hybrid Search
```bash
python cli/hybrid_search_cli.py <query>
```
Combines both semantic and keyword search capabilities for improved results.

### Multimodal Search
```bash
python cli/multimodal_search_cli.py --image <image_path> --query <text_query>
```
Performs search using both image and text inputs.

### Augmented Generation
```bash
python cli/augmented_generation_cli.py <prompt>
```
Generates augmented responses using the RAG system.

### Evaluation
```bash
python cli/evaluation_cli.py
```
Runs evaluation metrics on the search system.

### Chunked Semantic Search
```bash
python cli/chunked_semantic_search.py <query>
```
Performs semantic search on chunked text data.

### Image Description
```bash
python cli/describe_image_cli.py <image_path>
```
Generates descriptions for images using the system.

## Data Files

- `data/course-rag-movies.json`: Contains the movie dataset
- `data/golden_dataset.json`: Dataset for evaluation
- `data/stopwords.txt`: List of stopwords for text processing

## Cache

The system maintains several cache files for improved performance:
- `cache/chunk_embeddings.npy`
- `cache/chunk_metadata.json`
- `cache/movie_embeddings.npy`
- `cache/text_embedding.pt`

## Directory Structure

- `cli/`: Contains all CLI tools
- `utils/`: Utility functions and preprocessing tools
- `data/`: Data files and datasets
- `cache/`: Cached embeddings and metadata
