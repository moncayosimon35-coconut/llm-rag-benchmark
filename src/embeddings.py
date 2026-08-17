from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Wrapper around a Sentence Transformers embedding model."""

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(
            texts,
            normalize_embeddings=True
        )