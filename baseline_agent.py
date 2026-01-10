"""
baseline_agent.py
Baseline RAG pipeline using LangGraph.
Graph: START -> input -> retriever -> generator -> output -> END

Standard semantic retrieval with no ontology expansion.
"""

from typing import TypedDict
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from gemini_client import get_llm, TOP_K_RETRIEVAL

_vectorstore = None


def set_vectorstore(vs):
    global _vectorstore
    _vectorstore = vs


class BaselineState(TypedDict):
    query: str
    retrieved_docs: list
    context: str
    answer: str


def input_node(state: BaselineState) -> dict:
    """Validate and forward the query."""
    return {"query": state["query"]}


def retriever_node(state: BaselineState) -> dict:
    """Standard semantic retrieval using the original query."""
    if _vectorstore is None:
        raise RuntimeError("Vectorstore not initialized. Call set_vectorstore() first.")
    docs = _vectorstore.similarity_search(state["query"], k=TOP_K_RETRIEVAL)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_docs": docs, "context": context}


def generator_node(state: BaselineState) -> dict:
    """Send retrieved context + query to Gemini via LangChain."""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "Based on the following context, answer the question accurately "
        "and comprehensively.\n\n"
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Answer:"
    )
    chain = prompt | llm
    response = chain.invoke({"context": state["context"], "query": state["query"]})
    return {"answer": response.content}


def output_node(state: BaselineState) -> dict:
    return {"answer": state["answer"]}


def build_baseline_graph():
    """
    Build and compile the baseline RAG pipeline.

    Graph structure:
        START -> input -> retriever -> generator -> output -> END
    """
    graph = StateGraph(BaselineState)

    graph.add_node("input", input_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("generator", generator_node)
    graph.add_node("output", output_node)

    graph.add_edge(START, "input")
    graph.add_edge("input", "retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", "output")
    graph.add_edge("output", END)

    return graph.compile()


def run_baseline(query: str) -> str:
    """Run a single query through the baseline pipeline."""
    app = build_baseline_graph()
    result = app.invoke({"query": query, "retrieved_docs": [], "context": "", "answer": ""})
    return result["answer"]
