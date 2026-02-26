import json
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunking import chunk_documents
from src.vectorstore.chroma_store import VectorStore
from src.retrieval.retriever import Retriever

# Ground truth: queries mapped to expected relevant pages
EVAL_SET = [
    {
        "query": "What programming languages are important for ML roles?",
        "relevant_pages": [1, 2]
    },
    {
        "query": "What are the expectations for internship candidates?",
        "relevant_pages": [3]
    },
    {
        "query": "What software engineering skills are needed?",
        "relevant_pages": [2]
    },
    {
        "query": "What machine learning frameworks are mentioned?",
        "relevant_pages": [1]
    },
    {
        "query": "What behavioral traits do companies look for?",
        "relevant_pages": [3]
    }
]


def precision_at_k(retrieved_pages, relevant_pages, k):
    top_k = retrieved_pages[:k]
    relevant_count = sum(1 for p in top_k if p in relevant_pages)
    return relevant_count / k


def recall_at_k(retrieved_pages, relevant_pages, k):
    top_k = set(retrieved_pages[:k])
    relevant_count = sum(1 for p in top_k if p in relevant_pages)
    return relevant_count / len(relevant_pages) if relevant_pages else 0


def mrr(retrieved_pages, relevant_pages):
    for i, page in enumerate(retrieved_pages):
        if page in relevant_pages:
            return 1.0 / (i + 1)
    return 0.0


def run_evaluation():
    # Index documents
    docs = load_pdf("test.pdf")
    chunks = chunk_documents(docs)
    store = VectorStore()
    store.add_documents(chunks)
    retriever = Retriever(store)

    results = []
    total_p1, total_p3, total_r3, total_mrr = 0, 0, 0, 0

    for item in EVAL_SET:
        retrieved = retriever.retrieve(item["query"], top_k=5)
        retrieved_pages = [r["metadata"]["page"] for r in retrieved]

        p1 = precision_at_k(retrieved_pages, item["relevant_pages"], 1)
        p3 = precision_at_k(retrieved_pages, item["relevant_pages"], 3)
        r3 = recall_at_k(retrieved_pages, item["relevant_pages"], 3)
        m = mrr(retrieved_pages, item["relevant_pages"])

        total_p1 += p1
        total_p3 += p3
        total_r3 += r3
        total_mrr += m

        results.append({
            "query": item["query"],
            "expected_pages": item["relevant_pages"],
            "retrieved_pages": retrieved_pages,
            "precision@1": p1,
            "precision@3": p3,
            "recall@3": r3,
            "mrr": m
        })

    n = len(EVAL_SET)
    summary = {
        "mean_precision@1": total_p1 / n,
        "mean_precision@3": total_p3 / n,
        "mean_recall@3": total_r3 / n,
        "mean_mrr": total_mrr / n
    }

    print("\n=== RETRIEVAL EVALUATION ===\n")
    for r in results:
        print(f"Query: {r['query']}")
        print(f"  Expected: {r['expected_pages']} | Retrieved: {r['retrieved_pages']}")
        print(f"  P@1={r['precision@1']:.2f}  P@3={r['precision@3']:.2f}  R@3={r['recall@3']:.2f}  MRR={r['mrr']:.2f}")
        print()

    print("=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}")

    with open("evaluation/results.json", "w") as f:
        json.dump({"queries": results, "summary": summary}, f, indent=2)
    print("\nResults saved to evaluation/results.json")


if __name__ == "__main__":
    run_evaluation()