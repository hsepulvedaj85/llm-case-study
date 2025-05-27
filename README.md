
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
  set PYTHONUTF8=1

### 2. On Windows (PowerShell):
  PowerShell
  $env:PYTHONUTF8=1

### 3. On Linux/macOS:
  Bash
  export PYTHONUTF8=1
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