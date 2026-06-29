"""
Builds the RAG chain:
merged retriever → prompt → Qwen3-32B on Groq → string output
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from retriever import build_retriever, FALLBACK_MESSAGE

SYSTEM_PROMPT = """You are a helpful and professional telecom customer care assistant.
Your job is to help customers resolve technical issues with their mobile service.

Use ONLY the context below to answer the customer's question.
The context comes from three sources:
- FAQ entries (general policy and how-to information)
- Past support tickets (real resolved cases with step-by-step resolutions)
- Telecom guide (official PDF documentation)

After each piece of information you use, cite the source in parentheses like these examples:
(Source: FAQ-12, Category: billing)
(Source: Ticket TK-045, Category: connectivity)
(Source: Guide Page 3, Chunk 7)

If the context does not contain enough information to answer confidently, say so clearly \
and suggest the customer call 611 or use the MyTelecom app.

Context:
{context}
"""


def _format_docs(docs: list[Document]) -> str:
    sections = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")

        if source == "faq":
            faq_id   = doc.metadata.get("faq_id", "?")
            category = doc.metadata.get("category", "?")
            label = f"[FAQ | ID: {faq_id} | Category: {category}]"

        elif source == "ticket":
            ticket_id = doc.metadata.get("ticket_id", "?")
            category  = doc.metadata.get("category", "?")
            label = f"[TICKET | ID: {ticket_id} | Category: {category}]"

        elif source == "guide":
            page  = doc.metadata.get("page", "?")
            chunk = doc.metadata.get("chunk_index", "?")
            label = f"[GUIDE | Page: {page} | Chunk: {chunk}]"

        else:
            label = f"[{source.upper()}]"

        sections.append(f"{label}\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


def build_chain():
    retriever = build_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    llm = ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0,
        max_tokens=None,
        reasoning_format="parsed",
        timeout=None,
        max_retries=2,
    )

    def chain_with_fallback(question: str):
        """Run retrieval first; skip LLM if no docs pass the confidence threshold."""
        docs = retriever.invoke(question)
        if not docs:
            yield FALLBACK_MESSAGE
            return
        context = _format_docs(docs)
        formatted = prompt.format_messages(context=context, question=question)
        for chunk in llm.stream(formatted):
            yield chunk.content

    return chain_with_fallback