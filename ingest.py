import os
import hashlib
import time
from dotenv import load_dotenv
from tqdm import tqdm
from google import genai
from google.genai.errors import ClientError

from loaders import load_document
from chunking import chunk_docs
from vectordb import VectorDB

load_dotenv()

EMBED_MODEL = "gemini-embedding-001"

# ---- Free-tier safe limiter (embed items per minute) ----
ITEMS_PER_MIN_LIMIT = 95   # keep buffer under 100/min
WINDOW_SECONDS = 60
BATCH_SIZE = 5             # 5 items/request -> easier to stay under 100/min

def stable_id(source: str, page: int | None, text: str) -> str:
    h = hashlib.sha256((source + str(page) + text).encode("utf-8", errors="ignore")).hexdigest()
    return h[:24]

def embed_batch(client: genai.Client, texts: list[str]) -> list[list[float]]:
    print(f"[embed] embedding {len(texts)} chunks...")
    t0 = time.time()
    res = client.models.embed_content(model=EMBED_MODEL, contents=texts)
    out = [e.values for e in res.embeddings]
    print(f"[embed] done in {time.time() - t0:.2f}s")
    return out

def main():
    client = genai.Client()
    vdb = VectorDB(persist_dir="data/vdb", collection_name="docs")

    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)

    files = [
        os.path.join(raw_dir, f)
        for f in os.listdir(raw_dir)
        if os.path.isfile(os.path.join(raw_dir, f))
    ]

    if not files:
        print("No files found in data/raw. Add pdf/txt/docx/md files and re-run ingest.")
        return

    all_chunks = []
    for fp in files:
        pages = load_document(fp)
        print(f"Loaded {os.path.basename(fp)}: {len(pages)} page(s)")

        chunks = chunk_docs(pages, chunk_size=2000, overlap=200)
        print(f"Chunked {os.path.basename(fp)}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")
    if not all_chunks:
        print("No text extracted. If PDFs are scanned images, try a .txt file first.")
        return

    # Resume-safe: skip IDs already stored (VectorDB loads these internally)
    existing_ids = set(getattr(vdb, "_ids", []))

    items_in_window = 0
    window_start = time.time()

    def throttle_if_needed(about_to_send: int):
        nonlocal items_in_window, window_start
        now = time.time()

        # reset window
        if now - window_start >= WINDOW_SECONDS:
            window_start = now
            items_in_window = 0

        # sleep until next window if limit would be exceeded
        if items_in_window + about_to_send > ITEMS_PER_MIN_LIMIT:
            sleep_s = WINDOW_SECONDS - (now - window_start)
            if sleep_s > 0:
                print(f"[rate-limit] sleeping {sleep_s:.1f}s to respect embed limit...")
                time.sleep(sleep_s)
            window_start = time.time()
            items_in_window = 0

    # Process + write PER BATCH (so you never lose progress)
    for i in tqdm(range(0, len(all_chunks), BATCH_SIZE)):
        batch = all_chunks[i:i + BATCH_SIZE]

        batch_ids = []
        batch_docs = []
        batch_metas = []

        for b in batch:
            sid = stable_id(b["metadata"].get("source"), b["metadata"].get("page"), b["text"])
            if sid in existing_ids:
                continue
            batch_ids.append(sid)
            batch_docs.append(b["text"])
            batch_metas.append(b["metadata"])

        if not batch_docs:
            continue

        throttle_if_needed(len(batch_docs))

        try:
            batch_embs = embed_batch(client, batch_docs)
            items_in_window += len(batch_docs)
        except ClientError as e:
            # If rate-limited, wait and retry once
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                print("[embed] 429 hit. Waiting ~55s and retrying once...")
                time.sleep(55)
                batch_embs = embed_batch(client, batch_docs)
                items_in_window += len(batch_docs)
            else:
                print("\n❌ Gemini embeddings failed:")
                print(e)
                raise

        vdb.add(
            ids=batch_ids,
            embeddings=batch_embs,
            documents=batch_docs,
            metadatas=batch_metas
        )
        existing_ids.update(batch_ids)

    print("✅ Ingestion complete. VectorDB updated.")
    print("✅ Files written to: data/vdb/ (docs_meta.jsonl, docs_emb.npy)")

if __name__ == "__main__":
    main()
