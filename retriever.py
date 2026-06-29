"""
Builds a merged retriever across all three Chroma collections:
- faq     : FAQ entries (no chunking — 1 row = 1 doc)
- tickets : resolved support tickets (no chunking — 1 ticket = 1 doc)
- guides  : PDF guide chunks (RecursiveCharacterTextSplitter applied at ingest)

Uses similarity_search_with_score() for confidence filtering.
Chroma returns L2 distances: lower = more similar.
SIMILARITY_THRESHOLD controls the cutoff — docs with distance above
this value are considered low-confidence and discarded.
If NO doc from ANY collection passes the threshold, retrieve() returns
an empty list — the chain in rag_chain.py then skips the LLM entirely
and returns a canned fallback message.
"""
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

CHROMA_DIR  = "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# L2 distance threshold — docs with distance > this are discarded.
# Lower value = stricter. 1.0 is a reasonable starting point for MiniLM.
SIMILARITY_THRESHOLD = 1.0

FALLBACK_MESSAGE = (
    "I'm sorry, I don't have enough information in my knowledge base to "
    "answer your question confidently. Please call **611** or use the "
    "**MyTelecom app** for further assistance."
)


def build_retriever(
    k_faq: int = 3,
    k_tickets: int = 3,
    k_guides: int = 3,
) -> RunnableLambda:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    faq_store = Chroma(
        collection_name="faq",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    tickets_store = Chroma(
        collection_name="tickets",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    guides_store = Chroma(
        collection_name="guides",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    def _filter(results: list[tuple[Document, float]]) -> list[Document]:
        """Keep only docs whose L2 distance is below the threshold."""
        return [doc for doc, score in results if score <= SIMILARITY_THRESHOLD]

    def retrieve(query: str) -> list[Document]:
        faq_results    = faq_store.similarity_search_with_score(query, k=k_faq)
        ticket_results = tickets_store.similarity_search_with_score(query, k=k_tickets)
        guide_results  = guides_store.similarity_search_with_score(query, k=k_guides)

        docs = (
            _filter(faq_results)
            + _filter(ticket_results)
            + _filter(guide_results)
        )
        # Return empty list if nothing passes — triggers fallback in rag_chain.py
        return docs

    return RunnableLambda(retrieve)