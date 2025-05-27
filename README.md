
# 🧠 Retrieval-Augmented Generation (RAG) Pipeline

This project implements a **RAG (Retrieval-Augmented Generation)** system that:

- Processes and chunks PDF documents
- Embeds and stores those chunks in a **Milvus** vector database
- Answers user questions using **LLM: LLaMA 3.2-1B** (by using [Ollama](https://ollama.com))
- Uses **Snowflake Arctic Embed-s** for embedding
- Provides a FastAPI interface and evaluation pipeline

---
## 📦 Environment Setup

To run this project, you will need to download software like Ollama and docker; and configure your enviroment to run properly.

### ✅ Requirements

- Python: `3.8+`
- OS: Linux/macOS/Windows
- Docker Desktop for Windows
- Follow the installation recommendations below:
  - **Docker Linux**: https://milvus.io/docs/install_standalone-docker.md
  - **Docker Compose (Linux)**: https://milvus.io/docs/install_standalone-docker-compose.md
  - **Docker Desktop (Windows)**: https://milvus.io/docs/install_standalone-windows.md
- Download Ollama: https://ollama.com/download

### 📥 Docker Desktop (Windows):
- _Install Docker Desktop_: https://docs.docker.com/desktop/setup/install/windows-install/
- _Install Windows Subsystem for Linux 2 (WSL 2)_: https://learn.microsoft.com/en-us/windows/wsl/install#install-wsl-command
- _Install Python 3.8+_.

### 1. On Windows (Command Prompt):
DOS
```dos
  set PYTHONUTF8=1
```

### 2. On Windows (PowerShell):
PowerShell
```PowerShell
  $env:PYTHONUTF8=1
```
### 3. On Linux/macOS:
Bash
```bash
  export PYTHONUTF8=1
```
---
## 🧰 Project Installation

### 1. Clone the project

```bash
  #  Clone the project
  git clone https://github.com/hsepulvedaj85/llm-case-study.git
  cd llm-case-study
```
### 2. Install virtual enviroment
```bash
  pip install virtualenv
```
### 3. Create virtual enviroment
```bash
  cd ll-case-study
  python3.12 -m venv venv
  source venv/bin/activate
```
### On Windows use

```bash
  cd ll-case-study
  virtualenv venv
  venv\Scripts\activate
```
## 🔧 Dependency Installation

All dependencies are listed in requirements.txt.

```bash
  pip install -r requirements.txt
```
### 🗃️ Vector Store (Milvus) Setup: https://milvus.io/docs/quickstart.md
#### ✅ Requirements: https://milvus.io/docs/prerequisite-docker.md

With the docker-compose.yml file in the root of the project first run dockerdesktop and then make sure you have the virtual environment `(venv)`  active to run:

```bash
  docker-compose up -d
```
### 🚀 Run the RAG Pipeline
In CMD or PowerShell (after activating venv):

```bash
  python pipeline.py
```

## 🔍 API Reference
To activate in virtual enviroment after running first pipeline "pipeline.py"

```bash
  uvicorn app:app --reload --port 8000
```

#### Get an answer

```http
  POST /query
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `Content-Type` | `application/json` | **Required**. question |

#### Example:

```json
  {
  "question": "What significant technological advancement is Veridia known for in agriculture?"
  }
```

#### 🎯 Retrieves and approximates an answer in JSON Format
```json
  {
  "answer": "Veridia is known for innovations in hydroharmonic farming technology that have revolutionized their yield."
  }
```

## 🧪 Running Tests

To run tests, run the following command

### 📄 Send POST request to:
```json
  $> POST http://localhost:8000/query
  {
    "question": "What is the official language of Veridia?"
  }
```

### Test via curl:
```bash
  curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"question": "What is the capital of Veridia?"}'
```

## 📊 Evaluation Pipeline

### 📁 Files Needed:
```
data/questions.txt (One question per line)

data/answers.txt (One answer per line, same order)
```

### 📈 Run Evaluation:
```bash
  python pipe_eval.py
```

This script will:
- Generate predictions using the same pipeline

- Embed both expected and generated answers

- Compute cosine similarity

- Save results to evaluation_report.txt
## 📁 Project Structure

The project is organized as follows:

```php
  llm_case_study/
    ├── data/
    │   ├── dr_voss_diary.pdf      # The document to process
    │   ├── questions.txt          # Questions to answer
    │   ├── answers.txt            # Answers to the questions, for evaluation/testing purposes
    │   └── evaluation_report.txt  # Questions, answers and predicted answers with COSINE similarity 
    ├── scripts/
    │   ├── prepare_data.py        # Load and preprocess PDF file
    │   └── eval.py                # Evaluation pipeline
    ├── src/                    
    │   ├── chunker.py             # PDF Load, prepare and split into chunks
    |   ├── embedder.py            # Embedding logic 
    |   ├── final_answer.py        # Generates answer to a query based
    |   ├── milvus_client.py       # Milvus setup & ingestion
    |   └── search.py              # Searches Milvus collection based on a given query embedding.
    ├── test/
    |   └── search_reranker.py     # test with re-rankink Cross Encoder 'ms-marco-TinyBERT-L-2-v2'
    ├── app.py                     # FastAPI server implementation
    ├── pipeline.py                # End-to-end runner Document processing pipeline to Milvus.
    ├── README.md                  # Project documentation
    ├── requirements.txt           # List of needed librarys
    └── docker-compose.yml         # file to run Milvus.  
```
---


## 🧪 Technical Discussion

### 1. Data Processing
🔸 Document Parsing
- Document loaded using PyPDFLoader from LangChain, which is reliable and efficient for structured PDF parsing.

- Extracts clean text and integrates well with downstream LangChain processing.

🔸 Chunking Strategy
- Applied using LangChain’s RecursiveCharacterTextSplitter:
    - `chunk_size = 250` characters
    - `chunk_overlap = 100` characters
    
🔸 Justification:
- Shorter chunks ensure the content stays within the embedding model's optimal input size.
- Overlap of 100 characters maintains semantic continuity between chunks.
- Character-based splitting prevents fragmentation of logical sentences while being tokenizer-agnostic.

### 2. 🔍 Model Selection
🔹 Embedding Model: snowflake-arctic-embed-s

**Justification**:

- Optimized for retrieval tasks like semantic similarity and dense passage retrieval.

- Low latency and lightweight embedding size (384 dimensions), which is ideal for storing in vector databases like Milvus.

- Open-source and freely accessible without API restrictions (compared to OpenAI or Cohere embeddings).

- Trained specifically on English, improving performance for this use case.

**Alternative Options Considered**:

- `all-MiniLM-L6-v2` (good but lower quality).
- `bge-small-en` (powerful, but Snowflake's embeddings outperform for long documents).

### 3. 🧠 Retrieval System
🔸 Vector Database: Milvus
**Why Milvus**:
- High-performance, vector search engine
- Seamless Docker-based deployment.
- Scales well for document-level retrieval and supports hybrid search.

**Collection Design**:

- Fields: 
    - `chunk_id` (primary key) Unique identifier to ensure traceability and avoid collisions.
    - `text` (chunk content) Holds the actual content chunked from the PDF
      - `max_length=65535`  to accommodate dense text.
    - `embedding` (float vector of dimension 384) Stores the 384-dimensional embedding vector.
    - `source` (document reference)

- Index: 
    - Type: `IVF_FLAT`
    - Parameters: `{"nlist":1024}`
    - Metric Type: `COSINE`
`nlist` determines the number of clustering centers. A higher value can lead to better accuracy with more compute cost.

`IVF_FLAT ` is a good choice for most small-medium-sized datasets. The vectors are partitioned into clusters (nlist controls how many). Good balance of speed vs. accuracy. Works well for up to a few million vectors. Slower than compressed indexes on very large datasets. and Memory usage is higher (stores full vectors).

`COSINE` The choice of similarity metric, -COSINE- is as important as the type of index in vector search systems like Milvus. and embeddings like snowflake-arctic-embed-s are normalized: the meaning of a text is encoded in its direction, not its size. so -1 (opposite) to 1 (identical), with 0 meaning orthogonal (unrelated). easy to evaluate using cosine similarity.

🔸 **Search Strategy**:

🔎 Vector Search
```python
search_params = {"metric_type": "COSINE", "params": {"nprobe": 1024}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=limit,
        output_fields=["text", "source"]
    )
```
Top-K `k=5` using cosine similarity to generate context.

`nprobe=1024` Searches 64 out of the 1024  for balance between speed and recall. 

- Query Embedding:
    - Generated from the user question using snowflake-arctic-embed-s.
- Search Process:
    - Milvus compares the query vector against the stored chunk vectors.
    - Searches within nprobe=1024 out of the nlist=1024 partitions — a good tradeoff between recall and performance.

🔢 Ranking
- Milvus handles initial scoring based on cosine similarity.
- Top-K results (e.g., 5) are returned with associated scores and metadata (`text`, `source`).
- Ranking is implicit — top result = most semantically similar based on vector proximity.

🧾 Context Construction for LLM
📋 Prompt Format
The retrieved documents are concatenated into a structured prompt:
```txt
Context:
---
{text_chunk_1}
---
{text_chunk_2}
...
Question: {user_question}
```
This format ensures:

- Each chunk is clearly separated (via ---).
- The LLM can easily distinguish context from the question.
- Works well with models like LLaMA 3.2-1B in Ollama, which expect natural language input and perform better with clear delimiters.

🧠 Why this works:
- Semantic grounding: The context provides anchor points so the LLM doesn’t hallucinate.
- Controlled input: Limited to only top-K relevant passages → reduces irrelevant noise.
- LLM flexibility: Works whether the model is hosted locally (via Ollama) or remotely via API.

### 4. 📊 Results and Analysis

🔹 Evaluation Results
- Evaluated 1-to-1 match of generated answers vs ground-truth from answers.txt
- Used cosine similarity on embedded answers to compute semantic closeness

🔸**Results**:

- Avg. cosine similarity: ~0.90
- Accuracy (>0.90 similarity): ~90%
- Some questions were matched perfectly; others had near misses or generic answers

🔹 Strengths
- Very fast, runs locally and offline
- Flexible: can switch models easily
- Good semantic retrieval from Milvus
- Answer generation grounded in context, reducing hallucinations

🔹 Weaknesses
- Small model (LLaMA 3.2-1B) struggles with complex synthesis or reasoning
- Some chunks are too large or too vague → retrieval mismatch
- Evaluation is similarity-based, not comprehension-based

### 🚀 Future Improvements

| Area | Suggestion |
| :-------- | :------- |
| LLM | Use `LLaMA 3-8B-Instruct` or `Mistral 7B` via Ollama for better QA |
| Embeddings | Try `bge-base-en` with better document-query alignment |
| Chunking | Apply semantic chunking (e.g., by paragraph/topic) |
| Indexing | Try `IVF_SQ8` for Milvus elasticity or `IVF_PQ` for more agressive compresion |
| Prompting | Use prompt templates like "You are a historian answering questions about..." |
| Ranking | Rerank retrieved chunks before answering using a cross-encoder model |
| Evaluation | Integrate human-in-the-loop assessment or BLEU/ROUGE scoring |

### 🧠 Summary
When to Use What
| Index Type | Speed        | Accuracy       | Memory  | Best For                     |
| ---------- | ------------ | -------------- | ------- | ---------------------------- |
| `IVF_FLAT` | 🟢 Medium    | 🟢 High        | 🔴 High | Balanced general use         |
| `IVF_SQ8`  | 🟢 Fast      | 🟡 Medium      | 🟢 Low  | Large datasets, lower memory |
| `IVF_PQ`   | 🟢 Fast      | 🟡 Medium      | 🟢 Low  | Massive datasets (10M+)      |

This RAG system is simple, local, and modular, and serves as a strong prototype for document-based QA. With a few targeted upgrades (mainly in LLM and retrieval tuning), it could evolve into a production-ready pipeline for customer support, research assistants, or legal/document QA.