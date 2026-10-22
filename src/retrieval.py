from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
from src.ingestion import get_vectorstore


# ---------- Sparse (BM25) ----------

def get_all_chunks(vectorstore):
    """Pull every stored chunk back out of the FAISS index."""
    return list(vectorstore.docstore._dict.values())


def get_bm25_retriever(vectorstore, k=4):
    chunks = get_all_chunks(vectorstore)
    return BM25Retriever.from_documents(chunks, k=k)


# ---------- Hybrid (dense + BM25) ----------

def get_hybrid_retriever(vectorstore, k=4):
    dense = vectorstore.as_retriever(search_kwargs={"k": k})
    bm25 = get_bm25_retriever(vectorstore, k=k)
    return EnsembleRetriever(
        retrievers=[dense, bm25],
        weights=[0.5, 0.5],
    )


# ---------- Re-ranking ----------

def get_reranked_retriever(vectorstore, candidates=8, top_n=3):
    hybrid = get_hybrid_retriever(vectorstore, k=candidates)
    cross_encoder = HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n)
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=hybrid,
    )


# ---------- Multi-query ----------

def get_multiquery_retriever(vectorstore, k=4):
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatGroq(model="openai/gpt-oss-20b")
    return MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)


# ---------- HyDE ----------

HYDE_PROMPT = ChatPromptTemplate.from_template(
    """Write a short hypothetical passage that would answer this question,
as if it came from a document. Do not mention that it's hypothetical.

Question: {question}

Passage:"""
)


def get_hyde_retriever(vectorstore, k=4):
    llm = ChatGroq(model="openai/gpt-oss-20b")
    hyde_chain = HYDE_PROMPT | llm | StrOutputParser()

    def retrieve(question):
        hypothetical_doc = hyde_chain.invoke({"question": question})
        return vectorstore.similarity_search(hypothetical_doc, k=k)

    return retrieve  # a plain function, not a LangChain Retriever object


# ---------- Router (automatic strategy selection) ----------

ROUTER_PROMPT = ChatPromptTemplate.from_template(
    """Classify the retrieval strategy that best fits this question. Reply with
exactly one word: "standard", "multiquery", or "hyde".

- "standard": simple, direct questions with clear keywords
- "multiquery": vague, broad, or ambiguous questions that could be phrased
  several different ways
- "hyde": questions asking for a specific fact, definition, or explanation
  where a hypothetical answer would closely resemble the source text

Question: {question}

Strategy:"""
)


def get_router():
    llm = ChatGroq(model="openai/gpt-oss-20b")
    return ROUTER_PROMPT | llm | StrOutputParser()


def route_and_retrieve(question, vectorstore, router, hybrid_retriever, k=4):
    strategy = router.invoke({"question": question}).strip().lower()

    if "multiquery" in strategy:
        docs = get_multiquery_retriever(vectorstore, k=k).invoke(question)
    elif "hyde" in strategy:
        docs = get_hyde_retriever(vectorstore, k=k)(question)
    else:
        strategy = "standard"
        docs = hybrid_retriever.invoke(question)

    return docs, strategy


# ---------- Quick test ----------

if __name__ == "__main__":
    vectorstore = get_vectorstore()
    router = get_router()
    hybrid = get_hybrid_retriever(vectorstore)

    questions = [
        "What do teams collect in distributed approaches to digital ecosystems?",
        "Tell me about how teams handle information over time.",
        "What is a useful process for separating facts from assumptions?",
    ]

    for q in questions:
        docs, strategy = route_and_retrieve(q, vectorstore, router, hybrid)
        print(f"Q: {q}")
        print(f"Strategy chosen: {strategy}")
        print(f"Top result: {docs[0].metadata.get('source')} -> {docs[0].page_content[:100]}\n")