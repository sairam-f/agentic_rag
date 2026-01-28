from typing import List, Dict, Any

def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    text = (text or "").replace("\r", "").strip()
    if not text:
        return []

    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break  # ✅ end reached safely

        next_start = end - overlap

        # ✅ guarantee progress (prevents infinite loop)
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks

def chunk_docs(pages: List[Dict[str, Any]], chunk_size: int = 2000, overlap: int = 200):
    out = []
    for p in pages:
        txt = p.get("text") or ""
        # safety cap against insanely large text blobs
        if len(txt) > 2_000_000:
            txt = txt[:2_000_000]

        for c in chunk_text(txt, chunk_size=chunk_size, overlap=overlap):
            out.append({"text": c, "metadata": p.get("metadata", {})})
    return out
