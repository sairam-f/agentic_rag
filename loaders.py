import os
from typing import List, Dict, Any
from pypdf import PdfReader
import docx

def load_txt(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [{"text": text, "metadata": {"source": os.path.basename(path), "page": None}}]

def load_docx(path: str) -> List[Dict[str, Any]]:
    d = docx.Document(path)
    text = "\n".join([p.text for p in d.paragraphs if p.text.strip()])
    return [{"text": text, "metadata": {"source": os.path.basename(path), "page": None}}]

def load_pdf(path: str) -> List[Dict[str, Any]]:
    reader = PdfReader(path)
    out = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            out.append({
                "text": text,
                "metadata": {"source": os.path.basename(path), "page": i + 1}
            })
    return out

def load_document(path: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".txt", ".md"]:
        return load_txt(path)
    if ext == ".docx":
        return load_docx(path)
    if ext == ".pdf":
        return load_pdf(path)
    raise ValueError(f"Unsupported file type: {ext}")
