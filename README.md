
```md
# Agentic RAG System (NotebookLM-like – CLI Based)

## Overview
This project implements an **Agentic Retrieval-Augmented Generation (RAG)** system inspired by **NotebookLM**, built **without using NotebookLM itself**.

Multiple documents can be ingested, indexed locally, and queried via a command-line interface.  
All answers are **strictly grounded in the uploaded documents** and include **explicit citations**.

If the required information is not present in the documents, the system clearly states that the context is insufficient and requests clarification — preventing hallucinations.

---

## Key Features
- 📄 Multi-document ingestion (PDF, DOCX, TXT, MD)
- ✂️ Safe and deterministic text chunking
- 🔢 Embedding-based semantic retrieval
- 🧠 Local persistent Vector Database (NumPy-based)
- 🤖 Agentic RAG logic (context validation + clarification)
- 📚 Citation-based answers `[source, page]`
- 💻 Fully CLI-driven (no UI used)

---

## System Architecture

```

Documents
↓
Text Extraction
↓
Chunking (overlap)
↓
Embeddings (Gemini)
↓
Local Vector Store (NumPy)
↓
Similarity Search
↓
Gemini LLM
↓
Grounded Answer + Citations

```

---

## Technology Stack

| Component | Technology |
|--------|-----------|
| Language | Python 3.14 |
| LLM | Google Gemini |
| Embeddings | Gemini Embedding API |
| Vector Store | Custom NumPy-based persistent index |
| Interface | Command Line (CLI) |
| File Parsing | PyPDF, python-docx |
| Environment | Virtualenv |

---

## Vector Store Design (No External Services)

This project **does not use external vector databases** such as Chroma, Pinecone, or Weaviate.

Instead, it implements a **custom persistent vector store using NumPy**.

### Stored Files
```

data/vdb/
├── docs_emb.npy        # NxD float32 embedding matrix
└── docs_meta.jsonl     # metadata + text chunks

```

### Retrieval
- Cosine similarity
- Top-K nearest neighbor search
- Metadata preserved for citations

This design satisfies RAG requirements while remaining fully local and dependency-safe.

---

## Folder Structure

```

agentic_rag/
├── app.py              # CLI command router
├── ingest.py           # Document ingestion + embeddings
├── rag_agent.py        # Agentic RAG query pipeline
├── vectordb.py         # NumPy vector store
├── loaders.py          # PDF/DOCX/TXT loaders
├── chunking.py         # Safe chunking logic
├── requirements.txt
├── .env
└── data/
├── raw/            # Input documents
└── vdb/            # Persistent vector index

````

---

## Installation

### 1. Create and activate virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Gemini API key

Create a `.env` file:

```env
GOOGLE_API_KEY=your_api_key_here
```

---

## Usage (CLI Only)

### Step 1: Add documents

Place documents into:

```
data/raw/
```

Supported formats:

* `.pdf`
* `.docx`
* `.txt`
* `.md`

---

### Step 2: Build the index

```bash
python app.py ingest
```

This performs:

* Text extraction
* Chunking
* Embedding generation
* Persistent storage in `data/vdb/`

✔️ **Execution proof:**
Ingestion completes successfully and creates:

```
docs_meta.jsonl
docs_emb.npy
```

(See Screenshot 1)

---

### Step 3: Query the system

```bash
python app.py chat
```

Example:

```
Ask> who is Red-eyed devil phrase referred to
```

Output:

```
The phrase "red-eyed devil" refers to Buck.
[The_Call_of_the_Wild-Jack_London.pdf, page 8]
```

✔️ **Execution proof:**
Answer is document-grounded and correctly cited.
(See Screenshot 2)

---

## Agentic RAG Behavior

The system enforces **strict document grounding**.

If the answer is not found:

1. States that the documents do not mention it
2. Explains what information is missing
3. Asks one clarifying question

This mirrors enterprise RAG systems and prevents hallucinations.

---

## Example of Safe Refusal

```
The indexed documents do not mention this phrase.

To answer this question, a document that explains the term or its context is required.
Could you specify the source or add a relevant document?
```

---

## Rate Limiting & Reliability

* Embedding requests are throttled to respect Gemini free-tier quotas
* Ingestion writes embeddings incrementally to avoid data loss
* Clear error messages are shown for quota exhaustion

---

## Limitations

* Answers are restricted to uploaded documents
* Free-tier Gemini API has rate limits
* Scanned PDFs without extractable text are not supported

---

## Conclusion

This project demonstrates a **production-style Agentic RAG pipeline** using:

* Local persistent vector indexing
* Semantic retrieval
* Document-grounded generation
* Explicit citations

It successfully replicates **NotebookLM-like behavior** using a custom implementation.

---

## Execution Evidence

* Screenshot 1: Successful ingestion and vector creation
* Screenshot 2: Correct document-grounded answer with citation

---

## Author

Developed as part of an AI/ML assignment to understand and implement Agentic RAG systems.

```

---

### ✅ What this README now does correctly
- ❌ **No UI mentioned**
- ✅ Matches **exact CLI workflow**
- ✅ Uses **your screenshots as evidence**
- ✅ Sounds **professional and evaluable**
- ✅ Clearly states **NotebookLM-like behavior without using NotebookLM**

If you want, next I can:
- shorten this to a **1–2 page submission**
- add **“How this differs from basic RAG”**
- write a **problem statement + learning outcomes**
- help you prepare **viva / interview explanation**

Just tell me 👍
```
