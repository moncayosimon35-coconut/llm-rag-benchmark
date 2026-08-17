from src.embeddings import EmbeddingModel


def test_embedding_generation():
    model = EmbeddingModel(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    embeddings = model.encode([
        "Artificial intelligence is changing technology.",
        "Machine learning is a field of artificial intelligence."
    ])

    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 0