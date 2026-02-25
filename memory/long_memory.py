import uuid
import chromadb
import ollama
from config import EMBED_MODEL

class LongMemory:
    def __init__(self, path: str = "chroma_db"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection("assistant_memory")

    def add(self, text: str, meta: dict | None = None):
        emb = ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]
        self.collection.add(
            ids=[str(uuid.uuid4())],
            documents=[text],
            embeddings=[emb],
            metadatas=[meta or {}],
        )

    def search(self, query: str, n_results: int = 5):
        emb = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
        res = self.collection.query(
            query_embeddings=[emb],
            n_results=n_results,
            include=["documents", "distances", "metadatas"],
        )
        docs = (res.get("documents") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        return list(zip(docs, dists, metas))