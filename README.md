# 🛠️ BlueCloud LLM Engineer/Scientist Coding Solution

This soluctions bla bla bla blabla bla

---

## 📂 Project Structure

The project is organized as follows:

```
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

### 2. Application Step (`app.py`)
- Implement a FastAPI server with a single POST `/query` endpoint that:
  - Accepts a JSON payload with a `question` field
  - Retrieves relevant context from Milvus DB
  - Uses an LLM to return an answer based on the retrieved context

### 3. Eval Pipeline (`scripts/eval.py`)
- Implement an evaluation pipeline that:
  - Answers the questions in `data/questions.txt` using your RAG pipeline
  - Compares the answers with the expected answers in `data/answers.txt`
  - Reports the accuracy of the answers

---

## 🔧 Technical Requirements

- **Vector Database:** Milvus Lite (required)
- **API Framework:** FastAPI (required)
- **Models:**  
  - All models used in your pipeline (LLMs, embedding models, or any others) must be **open source**.  
  - We recommend using the following models:  
    - **LLM:** Llama-3.2-1B
    - **Embedding model:** snowflake-arctic-embed-s
  - You are free to use different models, but if you do, you must provide a justification for your choices.
- **Other Dependencies:** You're free to choose any additional utilities or packages you need.

---

## 📝 Report Guidelines

As part of your submission, you are required to **replace this README.md** with your own, documenting your approach. Your README should include the following content.

### 1. **Installation & Setup Instructions**  

Your README must include a step-by-step guide detailing everything required to install, set up, and run your pipeline. The guide should cover:  

- **Environment Setup:**  
  - Specify the Python version you used.  
  - If using a virtual environment (e.g., `venv`, `conda`, `poetry`), provide clear instructions on setting it up.  

- **Dependency Installation:**  
  - List all dependencies in a `requirements.txt` or `pyproject.toml`.  
  - Provide installation commands (e.g., `pip install -r requirements.txt`).  

- **Model Downloads & Setup (if required):**  
  - Include precise steps to download and configure any **LLM**, **embedding model** or **additional models** used.  
  - If external files or programs are required, provide instructions to obtain them or set them up.  

- **Running the Scripts & Application:**  
  - Explain how to run each script (`prepare_data.py`, `eval.py`), where to check for their outputs (if any) and how to interpret them.  
  - Provide commands to start the FastAPI server (`app.py`).  
  - If any configuration files or environment variables are needed, specify how to set them up.  

In short, your documentation should ensure that anyone following the steps can fully reproduce your setup and run the project without any additional guidance.

### 2. **Technical Discussion:**  
   Your report should include detailed discussions on the following topics:

   - **Model Selection:**
     - Choice of embedding model and rationale
     - Choice of LLM and reasoning behind the selection
   - **Data Processing:**
     - Document parsing and processing approach
     - Chunking strategy and its justification
   - **Retrieval System:**
     - Vector database design decisions
     - Retrieval and ranking approach
     - How context is prepared and fed to the LLM
   - **Results and Analysis:**
     - Evaluation results
     - Analysis of strengths and weaknesses
     - Potential improvements to enhance performance and make this solution production-ready

---

## ⭐ Bonus Points  

The following are entirely **optional** but can earn you extra points if implemented. Feel free to attempt them if you have the time and want to showcase additional skills!  

- **Image Understanding:**  
  - Extract images from `dr_voss_diary.pdf`.  
  - Use an **open-source multi-modal LLM** to generate text descriptions of the images.  
  - Provide a brief discussion on how these descriptions could be used in the main pipeline.  

- **Dockerization:**  
  - Set up your application to run inside a Docker container.  
  - Provide a `Dockerfile` for building the image.  
  - Include clear instructions on how to build the image, run the container, and use your application.  
---

## ✅ Evaluation Criteria

- **Development Quality:** Code readability, documentation, version control practices, and adherence to Python best practices
- **Functionality:** Correctness of each step and end-to-end pipeline execution
- **Efficiency:** Reasonable time and resource management for embeddings and search
- **Design Decisions:** Quality of your reasoning behind technical choices, such as chunking strategy, vector DB indexing & search parameters, embedding model and LLM choice, etc.
- **Analysis:** Depth and clarity of your evaluation of your solution's performance, limitations, and potential improvements

---

## 📦 Submission Format

To submit your solution:  
1. **Create a Git repository locally** and track your work using version control best practices.  
2. When you're done, **zip your entire repository (including the `.git` folder)**.  
   > 🚫 Make sure the zip file **does not** include anything listed in your `.gitignore`.
3. Send us the zip file as an email attachment.


**Good luck, and happy coding! 🚀**