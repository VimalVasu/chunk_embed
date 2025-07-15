## Product Requirements Document: Chunking & Embedding Service (Localhost Development)

### 1. Overview

The **Chunking & Embedding Service** is a local development tool for the Reverbia platform that takes raw meeting transcripts, segments them into meaningful chunks, generates semantic embeddings, and persists both the chunks and embeddings along with metadata into a local ChromaDB instance.

**Purpose**: Enable localhost development and testing of semantic retrieval and RAG workflows by transforming linear transcript data into vectorized, queryable units.

**Scope**: This PRD covers localhost deployment with ingestion of transcript JSON, chunk segmentation, metadata extraction, embedding generation (local models preferred), and storage in local ChromaDB.

---

### 2. Goals & Objectives

* **Accurate segmentation** of transcripts into topic- or time-based chunks for local development.
* **Simple chunking strategies** (fixed window, speaker boundaries) suitable for localhost testing.
* **Local embeddings** prioritizing Ollama models (Gemma 3, Llama 2) with optional OpenAI for testing.
* **Local ChromaDB storage** with file-based persistence for development workflows.
* **CLI interface** for easy local testing and development iteration.

---

### 3. Key Features

1. **Transcript Loader**

   * Input: JSON transcript file with text, speaker labels, and timestamps.
   * Validate structure and basic integrity.

2. **Chunking Engine**

   * **Fixed-window**: Split by configurable time window (default 60s).
   * **Speaker-based**: Split at speaker turns or topic-change heuristics.
   * **Topic-based (future)**: NLP-driven segmentation by discourse topics.

3. **Metadata Extraction**

   * Capture start/end timestamps, speaker IDs/names, chunk index.
   * Optionally tag tentative topics via keyword heuristics.

4. **Embedding Generator**

   * Primary provider: **OpenAI**: `text-embedding-ada-002` API for development.
   * Simple batching for localhost development (smaller batches, basic rate limiting).

5. **Persistence Layer**

   * Connect to local ChromaDB instance (file-based storage).
   * Upsert vector records: `chunk_id`, `embedding`, and `metadata`.
   * Simple file-based persistence in `./data/` directory.

6. **Interfaces**

   * **CLI**: `pipeline.py ingest --input <file> --window 60 --model openai`.
   * **Local web UI** (optional): simple Flask interface for development testing.

7. **Monitoring & Logging**

   * Simple console logging for development debugging.
   * Basic error handling with simple retries for API timeouts.

---

### 4. Technical Architecture

* **Language & Framework**: Python 3.9+, Click for CLI, simple project structure.
* **Chunking Module**: `chunker.py` with basic strategies (fixed window, speaker-based).
* **Embedding Module**: `embedder.py` focused on OpenAI API integration.
* **Storage Module**: `store.py` for local ChromaDB operations.
* **Configuration**: Simple JSON config for localhost development (API keys, file paths).

**Diagram**:

```
Transcript JSON → Chunker → [Chunk Objects + Metadata] → Embedder → [Vectors] → Store → Local ChromaDB
```

---

### 5. Performance & Scalability (Localhost)

* **Target**: Process small test transcripts (1-5 minutes) for development iteration.
* **Throughput**: Simple sequential processing, small batches (10-20 chunks).
* **Concurrency**: Basic synchronous processing suitable for localhost development.

---

### 6. Error Handling & Resilience (Development)

* **Validation Errors**: Clear error messages for malformed transcript JSON.
* **API Failures**: Simple retry logic (2-3 attempts) for OpenAI API calls.
* **Local DB Errors**: Basic ChromaDB exception handling for development.

---

### 7. Security & Privacy (Localhost)

* **API Keys**: Load OpenAI keys from `.env` file (not committed to git).
* **Data Protection**: Local file storage only, no external data transmission except OpenAI API.
* **Development Safety**: Clear separation of test data from production data.

---

### 8. Usage Examples (Localhost Development)

```bash
# Basic ingestion with OpenAI embeddings
python pipeline.py ingest --input test_data/sample_transcript.json --window 60

# Speaker-based chunking for development testing
python pipeline.py ingest --input test_data/meeting1.json --strategy speaker

# Development with verbose logging
python pipeline.py ingest --input test_data/short_meeting.json --verbose
```

---

### 9. Development Setup Requirements

* **Python Environment**: Virtual environment with Python 3.9+
* **Dependencies**: OpenAI API client, ChromaDB, Click for CLI
* **Local Storage**: `./data/` directory for ChromaDB persistence
* **Configuration**: `.env` file for API keys (not tracked in git)
* **Test Data**: Sample transcript JSON files in `./test_data/`

---

### 10. Future Enhancements (Post-Development)

* **Topic-based segmentation** for improved chunking accuracy.
* **Web interface** for easier development testing.
* **Export functionality** for processed chunks and embeddings.
* **Production deployment** considerations after localhost validation.

---

**This PRD defines the localhost development requirements for the Chunking & Embedding Service prototype.**
