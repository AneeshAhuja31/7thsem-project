"""
evaluate.py
Evaluation framework: LLM-as-judge correctness, concept coverage metrics
(Accuracy, Precision, Recall, F1), and paired t-test for statistical validation.
"""

import time
import numpy as np
from scipy import stats
from gemini_client import get_llm


# ── LLM-as-Judge ──
def llm_judge_correctness(
    query: str,
    expected_answer: str,
    actual_answer: str,
    llm=None,
) -> dict:
    """
    Use Gemini to judge whether the actual answer is correct.
    Returns {"correct": bool, "reasoning": str}.
    """
    if llm is None:
        llm = get_llm()

    judge_prompt = (
        "You are an expert medical knowledge evaluator.\n\n"
        "Given a question, the expected correct answer, and an actual answer, "
        "determine if the actual answer is correct.\n\n"
        f"Question: {query}\n\n"
        f"Expected Answer: {expected_answer}\n\n"
        f"Actual Answer: {actual_answer}\n\n"
        "Evaluate whether the actual answer conveys the same key medical facts "
        "as the expected answer. Minor wording differences are acceptable. "
        "The answer must be factually correct and address the question.\n\n"
        "Respond with EXACTLY one of these two words on the first line, "
        "followed by a brief explanation:\n"
        "CORRECT\n"
        "INCORRECT\n\n"
        "Your response:"
    )
    response = llm.invoke(judge_prompt)
    text = response.content.strip()
    is_correct = text.upper().startswith("CORRECT")
    return {"correct": is_correct, "reasoning": text}


def evaluate_concept_coverage(
    expected_concepts: list[str],
    actual_answer: str,
    llm=None,
) -> dict:
    """
    Use Gemini to check which expected concepts are covered in the answer.
    Returns {"mentioned": [...], "precision": float, "recall": float, "f1": float}.
    """
    if llm is None:
        llm = get_llm()

    if not expected_concepts:
        return {"mentioned": [], "precision": 1.0, "recall": 1.0, "f1": 1.0}

    concept_list = ", ".join(expected_concepts)
    judge_prompt = (
        "Given this medical answer and a list of medical concepts, "
        "identify which concepts are mentioned or substantively addressed.\n\n"
        f"Answer: {actual_answer}\n\n"
        f"Concepts to check: {concept_list}\n\n"
        "For each concept respond on its own line:\n"
        "ConceptName: YES\n"
        "ConceptName: NO\n"
    )
    response = llm.invoke(judge_prompt)
    resp_text = response.content

    mentioned = []
    for concept in expected_concepts:
        if f"{concept}: YES" in resp_text or concept.lower() in actual_answer.lower():
            mentioned.append(concept)

    tp = len(mentioned)
    recall = tp / len(expected_concepts) if expected_concepts else 1.0
    precision = tp / max(tp, 1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"mentioned": mentioned, "precision": precision, "recall": recall, "f1": f1}


# ── Full Evaluation Pipeline ──
def run_evaluation(pipeline_results: dict, delay: float = 1.5) -> dict:
    """
    Evaluate both pipelines' results using LLM-as-judge.
    Returns {"baseline": [...], "ontology": [...]}, each entry has per-query scores.
    """
    llm = get_llm()
    evaluation: dict[str, list] = {"baseline": [], "ontology": []}

    for pipeline_name in ["baseline", "ontology"]:
        key = f"{pipeline_name}_results"
        for i, result in enumerate(pipeline_results[key]):
            qi = result["query_info"]

            correctness = llm_judge_correctness(
                query=qi["query"],
                expected_answer=qi["expected_answer"],
                actual_answer=result["answer"],
                llm=llm,
            )
            time.sleep(delay)

            coverage = evaluate_concept_coverage(
                expected_concepts=qi["expected_concepts"],
                actual_answer=result["answer"],
                llm=llm,
            )
            time.sleep(delay)

            evaluation[pipeline_name].append({
                "query": qi["query"],
                "correct": correctness["correct"],
                "precision": coverage["precision"],
                "recall": coverage["recall"],
                "f1": coverage["f1"],
                "judge_reasoning": correctness["reasoning"],
            })
            print(f"    Evaluated {pipeline_name} Q{i + 1}: "
                  f"{'CORRECT' if correctness['correct'] else 'WRONG'}")

    return evaluation


def compute_aggregate_metrics(evaluation: dict) -> dict:
    """Compute aggregate metrics and paired t-test."""
    metrics: dict = {}

    for name in ["baseline", "ontology"]:
        scores = evaluation[name]
        metrics[name] = {
            "accuracy": float(np.mean([s["correct"] for s in scores])),
            "precision": float(np.mean([s["precision"] for s in scores])),
            "recall": float(np.mean([s["recall"] for s in scores])),
            "f1": float(np.mean([s["f1"] for s in scores])),
        }

    # Paired t-test on binary correctness scores
    b_scores = [float(s["correct"]) for s in evaluation["baseline"]]
    o_scores = [float(s["correct"]) for s in evaluation["ontology"]]
    t_acc, p_acc = stats.ttest_rel(b_scores, o_scores)

    # Paired t-test on F1 scores
    b_f1 = [s["f1"] for s in evaluation["baseline"]]
    o_f1 = [s["f1"] for s in evaluation["ontology"]]
    t_f1, p_f1 = stats.ttest_rel(b_f1, o_f1)

    metrics["statistical_tests"] = {
        "accuracy_ttest": {"t_statistic": float(t_acc), "p_value": float(p_acc)},
        "f1_ttest": {"t_statistic": float(t_f1), "p_value": float(p_f1)},
    }
    return metrics


def print_results(evaluation: dict, metrics: dict) -> None:
    """Print formatted comparison results."""
    print("\n" + "=" * 72)
    print("  EVALUATION RESULTS: Ontology-Enhanced vs Baseline RAG")
    print("=" * 72)

    # Per-query table
    print(f"\n{'#':<4} {'Query':<48} {'Baseline':>10} {'Ontology':>10}")
    print("-" * 72)
    for i in range(len(evaluation["baseline"])):
        q = evaluation["baseline"][i]["query"]
        q_short = (q[:45] + "...") if len(q) > 45 else q
        b = "CORRECT" if evaluation["baseline"][i]["correct"] else "WRONG"
        o = "CORRECT" if evaluation["ontology"][i]["correct"] else "WRONG"
        print(f"{i + 1:<4} {q_short:<48} {b:>10} {o:>10}")

    # Aggregate metrics
    print("\n" + "=" * 72)
    print("  AGGREGATE METRICS")
    print("=" * 72)
    print(f"{'Metric':<20} {'Baseline':>12} {'Ontology':>12} {'Delta':>10}")
    print("-" * 54)
    for m in ["accuracy", "precision", "recall", "f1"]:
        bv = metrics["baseline"][m]
        ov = metrics["ontology"][m]
        delta = ov - bv
        print(f"{m.upper():<20} {bv:>12.4f} {ov:>12.4f} {delta:>+10.4f}")

    # Required output lines
    b_acc = metrics["baseline"]["accuracy"]
    o_acc = metrics["ontology"]["accuracy"]
    improvement = ((o_acc - b_acc) / b_acc * 100) if b_acc > 0 else 0.0

    print(f"\nBaseline Accuracy = {b_acc:.4f}")
    print(f"Ontology Accuracy = {o_acc:.4f}")
    print(f"Improvement = {improvement:.2f}%")

    # Statistical significance
    print("\n" + "=" * 72)
    print("  STATISTICAL SIGNIFICANCE (Paired t-test)")
    print("=" * 72)
    for test_name, vals in metrics["statistical_tests"].items():
        sig = "YES (p < 0.05)" if vals["p_value"] < 0.05 else "NO (p >= 0.05)"
        print(f"  {test_name}: t = {vals['t_statistic']:.4f}, "
              f"p-value = {vals['p_value']:.4f} -> Significant: {sig}")

    p_val = metrics["statistical_tests"]["accuracy_ttest"]["p_value"]
    print(f"\np-value = {p_val:.6f}")

    # Final required line
    sig_text = "with statistical significance" if p_val < 0.05 else "without statistical significance"
    print(f"\nOntology-guided retrieval improved performance by "
          f"{improvement:.2f}% over baseline {sig_text}")
