"""
graph_pipeline.py
Unified pipeline runner that initializes shared resources and
runs both baseline and ontology-enhanced pipelines on test queries.
"""

import time
from utils import load_corpus, build_vectorstore, get_test_queries
import baseline_agent
import ontology_agent


class PipelineRunner:
    """Orchestrates corpus loading, vectorstore creation, and pipeline execution."""

    def __init__(self, corpus_path: str = None):
        print("Loading corpus...")
        self.corpus_df = load_corpus(corpus_path)
        print(f"  Loaded {len(self.corpus_df)} documents")

        print("Building vector store (embedding corpus)...")
        self.vectorstore = build_vectorstore(self.corpus_df, force_rebuild=True)
        print("  Vector store ready")

        # Share the SAME vectorstore instance with both pipelines
        baseline_agent.set_vectorstore(self.vectorstore)
        ontology_agent.set_vectorstore(self.vectorstore)

        # Pre-compile graphs
        self.baseline_app = baseline_agent.build_baseline_graph()
        self.ontology_app = ontology_agent.build_ontology_graph()

    def run_single_query(self, query: str, pipeline: str = "baseline") -> dict:
        """Run one query through the specified pipeline. Returns full state."""
        start = time.time()

        if pipeline == "baseline":
            result = self.baseline_app.invoke({
                "query": query,
                "retrieved_docs": [],
                "context": "",
                "answer": "",
            })
        elif pipeline == "ontology":
            result = self.ontology_app.invoke({
                "query": query,
                "expanded_terms": [],
                "expanded_query": "",
                "retrieved_docs": [],
                "context": "",
                "answer": "",
                "ontology_reasoning_log": "",
            })
        else:
            raise ValueError(f"Unknown pipeline: {pipeline}")

        result["latency_seconds"] = time.time() - start
        return result

    def run_all_queries(self, delay: float = 2.0) -> dict:
        """
        Run all 20 test queries through both pipelines.
        Returns {"baseline_results": [...], "ontology_results": [...]}.
        """
        queries = get_test_queries()
        baseline_results = []
        ontology_results = []

        for i, q in enumerate(queries):
            short = q["query"][:55] + "..." if len(q["query"]) > 55 else q["query"]
            print(f"  [{i + 1:2d}/20] {short}")

            # Baseline
            b = self.run_single_query(q["query"], "baseline")
            b["query_info"] = q
            baseline_results.append(b)
            time.sleep(delay)

            # Ontology-enhanced
            o = self.run_single_query(q["query"], "ontology")
            o["query_info"] = q
            ontology_results.append(o)
            time.sleep(delay)

        return {
            "baseline_results": baseline_results,
            "ontology_results": ontology_results,
        }
