from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError
from vectordb import VectorDB

load_dotenv()

# Use exactly one model (as you requested)
GEN_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "gemini-embedding-001"

SYSTEM = """You are an expert AI/ML assistant.
You must answer ONLY using the provided context.

If the context does not contain the answer:
1. Explicitly state that the documents do not mention it.
2. Explain what kind of information or document would be needed.
3. Ask one concise clarifying question.

Do NOT answer from general knowledge.
Always include citations as: [source, page].
Keep answers clear and assignment-friendly.
"""

def format_context(docs, metas, distances, max_chars=9000):
    items = []
    total = 0
    for d, m, dist in zip(docs, metas, distances):
        cite = f"[{m.get('source')}, page {m.get('page')}]"
        chunk = f"{cite}\n{d}\n"
        if total + len(chunk) > max_chars:
            break
        items.append(chunk)
        total += len(chunk)
    return "\n---\n".join(items)

def embed_query(client: genai.Client, text: str) -> list[float]:
    res = client.models.embed_content(model=EMBED_MODEL, contents=text)
    return res.embeddings[0].values

def retrieve(client: genai.Client, vdb: VectorDB, query: str, top_k: int = 6):
    q_emb = embed_query(client, query)
    res = vdb.query(q_emb, top_k=top_k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    distances = res["distances"][0]
    return docs, metas, distances

def agentic_answer(query: str) -> str:
    client = genai.Client()

    # ✅ NumPy VectorDB location
    vdb = VectorDB(persist_dir="data/vdb", collection_name="docs")

    # ✅ Guard: no docs indexed yet
    if not getattr(vdb, "_ids", []):
        return (
            "I don’t have any indexed documents yet.\n"
            "Add files to data/raw and run: python app.py ingest"
        )

    # Step 1: retrieve
    try:
        docs, metas, distances = retrieve(client, vdb, query, top_k=6)
    except ClientError as e:
        if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
            return (
                "Gemini API rate limit/quota reached while embedding your query.\n"
                "Please retry after the suggested delay, or reduce request frequency."
            )
        raise

    # ✅ Guard: retrieval returned nothing useful
    if not docs:
        return (
            "The indexed documents do not mention this.\n"
            "To answer, I need a document that contains the phrase/topic you’re asking about.\n"
            "Which document (or source) should I use, or can you add a file that mentions it?"
        )

    context = format_context(docs, metas, distances)

    # If context is empty after formatting/truncation, refuse safely
    if not context.strip():
        return (
            "The indexed documents do not contain enough relevant context to answer.\n"
            "Please add a document that explicitly discusses this topic, or share the exact sentence/paragraph."
        )

    user_prompt = f"""SYSTEM INSTRUCTIONS:
{SYSTEM}

QUESTION:
{query}

CONTEXT (use only this):
{context}
"""

    try:
        resp = client.models.generate_content(model=GEN_MODEL, contents=user_prompt)
        return resp.text or ""
    except ClientError as e:
        if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
            return (
                "Gemini API rate limit/quota reached while generating the answer.\n"
                "Please retry after the suggested delay."
            )
        raise

if __name__ == "__main__":
    while True:
        q = input("\nAsk> ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break
        print("\n" + agentic_answer(q))
