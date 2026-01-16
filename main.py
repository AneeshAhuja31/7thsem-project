import os
import sys
import time
from dotenv import load_dotenv


def main():
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Set a valid GOOGLE_API_KEY in your .env file.")
        sys.exit(1)

    print("=" * 72)
    print("  Ontology-Enhanced Agentic Retrieval")
    print("  Using LangGraph and Gemini API")
    print("=" * 72)

    # Step 1: Initialise pipeline runner (corpus + vectorstore)
    from graph_pipeline import PipelineRunner

    runner = PipelineRunner()

    # Step 2: Run all 20 queries through both pipelines
    print("\nRunning both pipelines on 20 test queries...")
    t0 = time.time()
    results = runner.run_all_queries(delay=2.0)
    elapsed = time.time() - t0
    print(f"\nPipeline execution completed in {elapsed:.1f}s")

    # Step 3: Evaluate with LLM-as-judge
    print("\nEvaluating results (LLM-as-judge)...")
    from evaluate import run_evaluation, compute_aggregate_metrics, print_results

    evaluation = run_evaluation(results, delay=1.5)
    metrics = compute_aggregate_metrics(evaluation)

    # Step 4: Print results
    print_results(evaluation, metrics)

    # Step 5: Show sample ontology expansions
    print("\n" + "=" * 72)
    print("  SAMPLE ONTOLOGY EXPANSIONS")
    print("=" * 72)
    for r in results["ontology_results"][:5]:
        print(f"\n  Query: {r['query']}")
        print(f"  Expanded terms: {r.get('expanded_terms', [])}")
        log = r.get("ontology_reasoning_log", "N/A")
        for line in log.split("\n"):
            print(f"    {line}")


if __name__ == "__main__":
    main()
