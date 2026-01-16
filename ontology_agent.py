"""
ontology_agent.py
Ontology-Enhanced RAG pipeline using LangGraph.
Graph: START -> input -> ontology_reasoning -> retriever -> generator -> output -> END

The ontology_reasoning node expands the query using:
  - rdfs:subClassOf  (subclass hierarchy)
  - owl:equivalentClass  (synonym / lay-clinical mapping)
  - med:hasSymptom, med:treatedBy, med:diagnosedBy, med:relatedCondition  (semantic neighbors)
"""

import re
import os
from typing import TypedDict

import rdflib
from rdflib import Namespace, RDF, RDFS, OWL
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from gemini_client import get_llm, TOP_K_RETRIEVAL

MED = Namespace("http://example.org/medical#")

# Module-level shared resources
_vectorstore = None
_reasoner = None


def set_vectorstore(vs):
    global _vectorstore
    _vectorstore = vs


# ── Ontology Reasoner ──
class OntologyReasoner:
    """Loads a Turtle ontology and provides query expansion via graph traversal."""

    def __init__(self, ttl_path: str = None):
        if ttl_path is None:
            ttl_path = os.path.join(os.path.dirname(__file__), "ontology", "domain.ttl")
        self.graph = rdflib.Graph()
        self.graph.parse(ttl_path, format="turtle")
        self.med = MED

    # ── public API ──
    def get_expansion_terms(self, query: str) -> tuple[list[str], str]:
        """Return (expanded_terms, reasoning_log) for a query."""
        concepts = self._find_matching_concepts(query)
        if not concepts:
            return [], "No ontology concepts matched the query."

        expansion = self._expand_concepts(concepts)

        all_terms: set[str] = set()
        for terms in expansion.values():
            all_terms.update(terms)

        log_parts = [f"Matched concepts: {[self._local(c) for c in concepts]}"]
        for key, terms in expansion.items():
            if terms:
                log_parts.append(f"  {key}: {sorted(terms)}")
        reasoning_log = "\n".join(log_parts)

        return sorted(all_terms), reasoning_log

    # ── internals ──
    def _find_matching_concepts(self, query: str) -> list:
        """Match query words against ontology class local names."""
        query_lower = query.lower()
        matched = []
        for s in self.graph.subjects(RDF.type, OWL.Class):
            local = self._local(s)
            readable = self._camel_to_words(local).lower()
            if readable in query_lower or local.lower() in query_lower:
                matched.append(s)
        return matched

    def _expand_concepts(self, concept_uris: list) -> dict[str, set[str]]:
        """Traverse ontology relations for each concept."""
        expansion: dict[str, set[str]] = {
            "equivalent": set(),
            "subclasses": set(),
            "superclasses": set(),
            "symptoms": set(),
            "treatments": set(),
            "related": set(),
            "diagnostics": set(),
        }
        for concept in concept_uris:
            # equivalentClass (bidirectional)
            for obj in self.graph.objects(concept, OWL.equivalentClass):
                expansion["equivalent"].add(self._local(obj))
            for subj in self.graph.subjects(OWL.equivalentClass, concept):
                expansion["equivalent"].add(self._local(subj))

            # subClassOf children
            for subj in self.graph.subjects(RDFS.subClassOf, concept):
                expansion["subclasses"].add(self._local(subj))

            # subClassOf parents
            for obj in self.graph.objects(concept, RDFS.subClassOf):
                expansion["superclasses"].add(self._local(obj))

            # custom properties
            for obj in self.graph.objects(concept, MED.hasSymptom):
                expansion["symptoms"].add(self._local(obj))
            for obj in self.graph.objects(concept, MED.treatedBy):
                expansion["treatments"].add(self._local(obj))
            for obj in self.graph.objects(concept, MED.diagnosedBy):
                expansion["diagnostics"].add(self._local(obj))
            for obj in self.graph.objects(concept, MED.relatedCondition):
                expansion["related"].add(self._local(obj))

        return expansion

    @staticmethod
    def _local(uri) -> str:
        return str(uri).split("#")[-1]

    @staticmethod
    def _camel_to_words(name: str) -> str:
        return re.sub(r"(?<!^)(?=[A-Z])", " ", name)


def _get_reasoner() -> OntologyReasoner:
    global _reasoner
    if _reasoner is None:
        _reasoner = OntologyReasoner()
    return _reasoner


# ── State ──
class OntologyState(TypedDict):
    query: str
    expanded_terms: list
    expanded_query: str
    retrieved_docs: list
    context: str
    answer: str
    ontology_reasoning_log: str


# ── Nodes ──
def input_node(state: OntologyState) -> dict:
    """Validate and forward the query."""
    return {"query": state["query"]}


def ontology_reasoning_node(state: OntologyState) -> dict:
    """
    Expand the query using ontology relations.
    Traverses subClassOf, equivalentClass, and custom properties.
    """
    reasoner = _get_reasoner()
    expanded_terms, reasoning_log = reasoner.get_expansion_terms(state["query"])

    if expanded_terms:
        readable = " ".join(
            [OntologyReasoner._camel_to_words(t) for t in expanded_terms]
        )
        expanded_query = f"{state['query']} {readable}"
    else:
        expanded_query = state["query"]

    return {
        "expanded_terms": expanded_terms,
        "expanded_query": expanded_query,
        "ontology_reasoning_log": reasoning_log,
    }


def retriever_node(state: OntologyState) -> dict:
    """Ontology-guided retrieval using the expanded query."""
    if _vectorstore is None:
        raise RuntimeError("Vectorstore not initialized. Call set_vectorstore() first.")
    docs = _vectorstore.similarity_search(state["expanded_query"], k=TOP_K_RETRIEVAL)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_docs": docs, "context": context}


def generator_node(state: OntologyState) -> dict:
    """Send retrieved context + original query to Gemini via LangChain."""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "Based on the following context, answer the question accurately "
        "and comprehensively.\n\n"
        "Context:\n{context}\n\n"
        "Additional ontology context: {ontology_log}\n\n"
        "Question: {query}\n\n"
        "Answer:"
    )
    chain = prompt | llm
    response = chain.invoke({
        "context": state["context"],
        "query": state["query"],
        "ontology_log": state["ontology_reasoning_log"],
    })
    return {"answer": response.content}


def output_node(state: OntologyState) -> dict:
    """Return the final answer."""
    return {"answer": state["answer"]}


# ── Graph Construction ──
def build_ontology_graph():
    """
    Build and compile the ontology-enhanced RAG pipeline.

    Graph structure:
        START -> input -> ontology_reasoning -> retriever -> generator -> output -> END
    """
    graph = StateGraph(OntologyState)

    graph.add_node("input", input_node)
    graph.add_node("ontology_reasoning", ontology_reasoning_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("generator", generator_node)
    graph.add_node("output", output_node)

    graph.add_edge(START, "input")
    graph.add_edge("input", "ontology_reasoning")
    graph.add_edge("ontology_reasoning", "retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", "output")
    graph.add_edge("output", END)

    return graph.compile()


def run_ontology_enhanced(query: str) -> str:
    """Run a single query through the ontology-enhanced pipeline."""
    app = build_ontology_graph()
    result = app.invoke({
        "query": query,
        "expanded_terms": [],
        "expanded_query": "",
        "retrieved_docs": [],
        "context": "",
        "answer": "",
        "ontology_reasoning_log": "",
    })
    return result["answer"]
