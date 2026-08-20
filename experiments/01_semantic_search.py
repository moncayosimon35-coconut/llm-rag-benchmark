from src.retrieval import VectorRetriever


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    retriever = VectorRetriever(MODEL_NAME)

    retriever.load_documents(
        "data/documents.json"
    )

    retriever.build_index()

    query = "How does RAG help language models answer questions?"

    results = retriever.search(
        query,
        top_k=3
    )

    print("\nQUERY:")
    print(query)

    print("\nTOP RESULTS:\n")

    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result['title']}")
        print(f"Score: {result['score']:.4f}")
        print(result["text"])
        print("-" * 80)


if __name__ == "__main__":
    main()