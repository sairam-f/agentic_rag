

---

# Agentic RAG System (NotebookLM-inspired CLI)

A high-fidelity, command-line **Agentic Retrieval-Augmented Generation (RAG)** system. This project replicates the core functionality of NotebookLM—multi-source grounding and precise citations—built from the ground up without using external RAG platforms or managed services.

## 🚀 Key Features

* **Multi-Format Ingestion:** Seamlessly processes `.pdf`, `.docx`, `.txt`, and `.md`.
* **Custom Vector Store:** Implements a local, persistent vector database using **NumPy** (no dependency on Pinecone, Chroma, or Weaviate).
* **Agentic Reasoning:** Features a validation layer that evaluates context before answering. If the context is insufficient, the system requests clarification instead of hallucinating.
* **Precise Citations:** Every response is explicitly grounded with source file names and page numbers: `[source, page]`.
* **Zero-UI CLI:** A pure terminal-based interface designed for speed and reliability.

---

## 🏗️ System Architecture

The following diagram illustrates the data flow from raw document ingestion to the final grounded response.

1. **Ingestion:** Documents are parsed and split into semantic chunks with overlapping windows to preserve context.
2. **Embedding:** Text chunks are converted into high-dimensional vectors via the **Google Gemini Embedding API**.
3. **Storage:** Vectors are stored in a local `.npy` matrix, with a corresponding `.jsonl` file for metadata and text.
4. **Retrieval:** The system uses **Cosine Similarity** to find the most relevant Top-K chunks.
5. **Agentic Logic:** The Gemini LLM acts as a reasoning agent, verifying if the retrieved chunks actually contain the answer before generating a response.

---

## 🛠️ Technology Stack

| Component | Technology |
| --- | --- |
| **Language** | Python 3.14 |
| **LLM** | Google Gemini (Pro/Flash) |
| **Embeddings** | Gemini Embedding API |
| **Vector Store** | Custom NumPy-based persistent index |
| **File Parsing** | PyPDF, python-docx |
| **Environment** | Python-dotenv / Virtualenv |

---

## 📁 Project Structure

```text
agentic_rag/
├── app.py              # Main CLI entry point (Command Router)
├── ingest.py           # Document processing & embedding pipeline
├── rag_agent.py        # Agentic RAG logic & grounding protocols
├── vectordb.py         # Custom NumPy Vector Store implementation
├── loaders.py          # PDF/DOCX/TXT extraction logic
├── chunking.py         # Semantic text splitting logic
├── requirements.txt    # Project dependencies
├── .env                # API Keys (Gitignored)
└── data/
    ├── raw/            # Input folder for your documents
    └── vdb/            # Local vector index (npy + jsonl)

```

---

## ⚙️ Installation & Setup

1. **Clone & Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

```


2. **Configure API Key**
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_api_key_here

```



---

## 💻 Usage (CLI)

### 1. Ingest Documents

Place your source files in `data/raw/` and run the ingestion script:

```bash
python app.py ingest

```

*This triggers the extraction, chunking, and creation of `docs_emb.npy` and `docs_meta.jsonl`.*

### 2. Query the Agent

Start the interactive chat session:

```bash
python app.py chat

```

**Example Interaction:**

> **Ask:** Who is "Red-eyed devil" referred to?
> **Output:** The phrase "red-eyed devil" refers to Buck.
> `[The_Call_of_the_Wild-Jack_London.pdf, page 8]`

---

## 🛡️ Agentic Behavior: Hallucination Mitigation

Unlike standard RAG systems that might "guess" an answer, this system enforces **Strict Grounding**.

**Example of Safe Refusal:**

> "The indexed documents do not mention this phrase. To answer this, a document explaining the term is required. Could you add a relevant source?"

---

## 📊 Execution Evidence

* **Screenshot 1:** Successful ingestion showing vector creation in the terminal.
* **Screenshot 2:** Verified CLI output showing a document-grounded answer with citations.

---

## 📝 Author

Developed as a technical implementation of Agentic RAG systems to demonstrate local persistence and semantic retrieval architectures.

---
