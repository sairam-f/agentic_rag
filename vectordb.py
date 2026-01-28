import os
import json
import numpy as np
from typing import List, Dict, Any, Optional

def _cosine_sim_matrix(query_vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    # query_vec: (d,), mat: (n,d)
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    m = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    return m @ q  # (n,)

class VectorDB:
    """
    Simple persistent vector store using:
      - data/vdb_meta.jsonl (one JSON per chunk: id, document, metadata)
      - data/vdb_emb.npy     (NxD float32 matrix)
    Interface matches your earlier Chroma VectorDB: add() and query()
    """
    def __init__(self, persist_dir: str = "data/vdb", collection_name: str = "docs"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        os.makedirs(self.persist_dir, exist_ok=True)

        self.meta_path = os.path.join(self.persist_dir, f"{collection_name}_meta.jsonl")
        self.emb_path = os.path.join(self.persist_dir, f"{collection_name}_emb.npy")

        self._metas: List[Dict[str, Any]] = []
        self._docs: List[str] = []
        self._ids: List[str] = []
        self._emb: Optional[np.ndarray] = None  # shape (n,d)

        self._load()

    def _load(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    self._ids.append(rec["id"])
                    self._docs.append(rec["document"])
                    self._metas.append(rec["metadata"])

        if os.path.exists(self.emb_path):
            self._emb = np.load(self.emb_path)
        else:
            self._emb = None

        # sanity: if meta exists but emb missing, reset meta to avoid mismatch
        if (self._emb is None and len(self._ids) > 0) or (self._emb is not None and self._emb.shape[0] != len(self._ids)):
            # reset everything if mismatch
            self._ids, self._docs, self._metas, self._emb = [], [], [], None
            if os.path.exists(self.meta_path):
                os.remove(self.meta_path)
            if os.path.exists(self.emb_path):
                os.remove(self.emb_path)

    def add(self, ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[Dict[str, Any]]):
        emb_new = np.array(embeddings, dtype=np.float32)

        # append metadata
        with open(self.meta_path, "a", encoding="utf-8") as f:
            for i, doc, meta in zip(ids, documents, metadatas):
                f.write(json.dumps({"id": i, "document": doc, "metadata": meta}, ensure_ascii=False) + "\n")

        self._ids.extend(ids)
        self._docs.extend(documents)
        self._metas.extend(metadatas)

        # append embeddings
        if self._emb is None:
            self._emb = emb_new
        else:
            self._emb = np.vstack([self._emb, emb_new])

        np.save(self.emb_path, self._emb)

    def query(self, embedding: List[float], top_k: int = 6):
        if self._emb is None or self._emb.shape[0] == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        q = np.array(embedding, dtype=np.float32)
        sims = _cosine_sim_matrix(q, self._emb)  # higher is better
        idx = np.argsort(-sims)[:top_k]

        docs = [self._docs[i] for i in idx]
        metas = [self._metas[i] for i in idx]
        distances = [float(1.0 - sims[i]) for i in idx]  # convert sim -> distance-like

        return {"documents": [docs], "metadatas": [metas], "distances": [distances]}
