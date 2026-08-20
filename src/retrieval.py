import json
from pathlib import Path

import faiss
import numpy as np

from src.embeddings import EmbeddingModel


class VectorRetriever:
    """Semantic search over a collection of documents."""

    def __init__(self, model_name: str):
        self.embedding_model = EmbeddingModel(model_name)
        self.documents = []
        self.index = None

    def load_documents(self, path: str):
        path = Path(path)

        with open(path, "r", encoding="utf-8") as file:
            self.documents = json.load(file)

    def build_index(self):
        texts = [document["text"] for document in self.documents]

        embeddings = self.embedding_model.encode(texts)

        embeddings = np.asarray(embeddings, dtype="float32")

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 3):
        query_embedding = self.embedding_model.encode([query])

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            document = self.documents[index]

            results.append({
                "id": document["id"],
                "title": document["title"],
                "text": document["text"],
                "score": float(score)
            })

        return results