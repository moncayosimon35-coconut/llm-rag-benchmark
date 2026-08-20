import json

from src.retrieval import VectorRetriever


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_questions(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def calculate_recall_at_k(results, relevant_document, k):
    retrieved_ids = [result["id"] for result in results[:k]]

    return int(relevant_document in retrieved_ids)

def calculate_reciprocal_rank(results, relevant_document):
    for rank, result in enumerate(results, start=1):
        if result["id"] == relevant_document:
            return 1 / rank

    return 0.0


def main():
    retriever = VectorRetriever(MODEL_NAME)

    retriever.load_documents("data/documents.json")
    retriever.build_index()

    questions = load_questions(
        "data/evaluation_questions.json"
    )

    recall_at_1 = []
    recall_at_3 = []
    recall_at_5 = []
    reciprocal_ranks = []

    for item in questions:
        results = retriever.search(
            item["question"],
            top_k=5
        )
    

        relevant_document = item["relevant_document"]
        relevant_rank = None

        for rank, result in enumerate(results, start=1):
            if result["id"] == relevant_document:
                relevant_rank = rank
                break



            if relevant_rank != 1:
                print("\nRETRIEVAL ERROR")
                print("-" * 40)
                print(f"Question: {item['question']}")
                print(f"Expected document: {relevant_document}")
                print(f"Retrieved rank: {relevant_rank}")

                print("\nRetrieved documents:")

        for rank, result in enumerate(results, start=1):
            print(
                f"{rank}. "
                f"{result['id']} - "
                f"{result['title']} "
                f"(score={result['score']:.4f})"
            )

         

        recall_at_1.append(
            calculate_recall_at_k(
                results,
                relevant_document,
                1
            )
        )

        recall_at_3.append(
            calculate_recall_at_k(
                results,
                relevant_document,
                3
            )
        )

        recall_at_5.append(
            calculate_recall_at_k(
                results,
                relevant_document,
                5
            )
        )
        reciprocal_ranks.append(
            calculate_reciprocal_rank(
            results,
            relevant_document
            )
        )

    recall_1 = sum(recall_at_1) / len(recall_at_1)
    recall_3 = sum(recall_at_3) / len(recall_at_3)
    recall_5 = sum(recall_at_5) / len(recall_at_5)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    print("\nRETRIEVAL BENCHMARK")
    print("=" * 40)

    print(f"Questions evaluated: {len(questions)}")
    print(f"Recall@1: {recall_1:.2%}")
    print(f"Recall@3: {recall_3:.2%}")
    print(f"Recall@5: {recall_5:.2%}")
    print(f"MRR: {mrr:.4f}")

    


if __name__ == "__main__":
    main()