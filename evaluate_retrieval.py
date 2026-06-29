"""
Evaluates retrieval quality on the 'tickets' Chroma collection.

Metric: Recall@3
For each test question, check whether the expected ticket ID
appears in the top-3 retrieved results.

Recall@3 = (number of hits) / (total test cases)

Usage:
    python evaluate_retrieval.py

No LLM is called — this tests the retriever only.
"""
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR  = "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
K           = 3   # top-k results to check

# ── 10 hand-crafted (question, expected_ticket_id) pairs ─────────────────────
# Replace the ticket IDs below with real IDs from your tickets.db
TEST_PAIRS = [
    ("My mobile data is very slow and pages won't load",          "TK-001"),
    ("Calls are dropping every few minutes while driving",        "TK-002"),
    ("I can't send or receive SMS messages",                      "TK-003"),
    ("Phone shows no service after inserting new SIM card",       "TK-004"),
    ("I was billed twice for the same month",                     "TK-005"),
    ("International roaming is not working in Europe",            "TK-006"),
    ("Wi-Fi calling option is missing from my phone settings",    "TK-007"),
    ("My account balance is wrong after topping up",              "TK-008"),
    ("Phone is locked to another network after purchase",         "TK-009"),
    ("Data bundle expired before the month ended unexpectedly",   "TK-010"),
]


def run_evaluation():
    print("=== Telecom RAG — Retrieval Quality Evaluation ===\n")
    print(f"Metric   : Recall@{K}")
    print(f"Collection: tickets")
    print(f"Test cases: {len(TEST_PAIRS)}\n")
    print("-" * 60)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    tickets_store = Chroma(
        collection_name="tickets",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    hits = 0
    for i, (question, expected_id) in enumerate(TEST_PAIRS, start=1):
        results = tickets_store.similarity_search_with_score(question, k=K)

        retrieved_ids = [
            doc.metadata.get("ticket_id", "") for doc, _ in results
        ]
        scores = [round(score, 4) for _, score in results]

        hit = expected_id in retrieved_ids
        if hit:
            hits += 1

        status = "✅ HIT " if hit else "❌ MISS"
        print(f"[{i:02d}] {status} | Expected: {expected_id}")
        print(f"       Question : {question}")
        print(f"       Retrieved: {retrieved_ids}")
        print(f"       Scores   : {scores}")
        print()

    print("-" * 60)
    recall = hits / len(TEST_PAIRS)
    print(f"\nRecall@{K} = {hits}/{len(TEST_PAIRS)} = {recall:.2%}\n")

    if recall >= 0.8:
        print("✅ Retrieval quality is GOOD (≥ 80%)")
    elif recall >= 0.5:
        print("⚠️  Retrieval quality is MODERATE (50–79%) — consider tuning k or embeddings")
    else:
        print("❌ Retrieval quality is POOR (< 50%) — review your data or embedding model")


if __name__ == "__main__":
    run_evaluation()