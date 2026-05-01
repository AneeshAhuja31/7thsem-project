# Ontology-Enhanced Agentic Retrieval

A research project that compares two Retrieval-Augmented Generation (RAG) pipelines in the medical domain:

- **Baseline RAG** – standard semantic retrieval with no query expansion.
- **Ontology-Enhanced RAG** – query expansion driven by a custom OWL/Turtle healthcare ontology before retrieval.

Both pipelines are implemented as [LangGraph](https://github.com/langchain-ai/langgraph) state-machine graphs, share the same ChromaDB vector store, and use Google Gemini as the LLM and embedding model. Evaluation uses an *LLM-as-judge* strategy (Gemini) and paired t-tests for statistical significance.

---

## Features

- **Two comparable RAG pipelines** built with LangGraph, ensuring a fair A/B comparison.
- **Healthcare ontology** (`ontology/domain.ttl`) modelled in OWL/Turtle with classes for diseases, treatments, symptoms, and diagnostics, plus custom object properties (`hasSymptom`, `treatedBy`, `diagnosedBy`, `relatedCondition`).
- **Ontology query expansion** – traverses `rdfs:subClassOf`, `owl:equivalentClass`, and domain-specific relations to enrich user queries with clinical synonyms and related concepts.
- **ChromaDB vector store** built from a medical corpus (`data/corpus.csv`) and embedded with Google Generative AI (`models/embedding-001`).
- **LLM-as-judge evaluation** for answer correctness and concept coverage (Precision, Recall, F1).
- **Statistical validation** via paired t-tests on accuracy and F1 scores.
- **20 curated test queries** across four categories: equivalence-dependent, subclass-dependent, semantic-neighbor, and neutral/control.

---

## Tech Stack

| Component | Library / Tool |
|-----------|----------------|
| Pipeline orchestration | `langgraph >= 0.2.0` |
| LLM & embeddings | `langchain-google-genai >= 2.0.0` (`gemini-2.0-flash` / `embedding-001`) |
| Vector store | `langchain-chroma >= 0.2.0`, `chromadb >= 0.5.0` |
| Ontology parsing | `rdflib >= 7.0.0` |
| LangChain core | `langchain >= 0.3.20`, `langchain-community >= 0.3.0` |
| Data / stats | `pandas`, `numpy`, `scipy`, `scikit-learn` |
| Configuration | `python-dotenv >= 1.0.0` |

---

## Repository Structure

```
7thsem-project/
├── main.py               # Entry point – runs both pipelines, evaluates, prints results
├── graph_pipeline.py     # PipelineRunner: initialises shared resources, runs all queries
├── baseline_agent.py     # Baseline RAG LangGraph pipeline
├── ontology_agent.py     # Ontology-Enhanced RAG LangGraph pipeline
├── evaluate.py           # LLM-as-judge evaluation & statistical tests
├── gemini_client.py      # Centralised Gemini LLM / embedding configuration
├── utils.py              # Corpus loading, vector store construction, test queries
├── data/
│   └── corpus.csv        # Medical document corpus (id, text, category, ground_truth_concepts)
├── ontology/
│   └── domain.ttl        # OWL/Turtle healthcare domain ontology
└── requirements.txt      # Python dependencies
```

---

## Setup

### Prerequisites

- Python 3.10 or later
- A [Google AI Studio](https://aistudio.google.com/) API key

### 1. Clone the repository

```bash
git clone https://github.com/AneeshAhuja31/7thsem-project.git
cd 7thsem-project
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

Create a `.env` file in the repository root:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

---

## How to Run

```bash
python main.py
```

The script performs the following steps automatically:

1. **Load corpus** – reads `data/corpus.csv` into memory.
2. **Build vector store** – embeds all documents into ChromaDB (rebuilt on each run).
3. **Run 20 test queries** – each query is executed through both the baseline and ontology-enhanced pipelines (a configurable `delay` prevents API rate-limiting).
4. **Evaluate** – uses Gemini as an LLM judge to score correctness and concept coverage.
5. **Print results** – displays a per-query comparison table, aggregate metrics, and statistical significance.

### Example output

```
========================================================================
  Ontology-Enhanced Agentic Retrieval
  Using LangGraph and Gemini API
========================================================================
Loading corpus...
  Loaded 50 documents
Building vector store (embedding corpus)...
  Vector store ready
Running both pipelines on 20 test queries...
  [ 1/20] What are the treatments for high blood pressure?
  ...

========================================================================
  EVALUATION RESULTS: Ontology-Enhanced vs Baseline RAG
========================================================================
#    Query                                             Baseline    Ontology
------------------------------------------------------------------------
1    What are the treatments for high blood pres...    CORRECT     CORRECT
...

========================================================================
  AGGREGATE METRICS
========================================================================
Metric                   Baseline      Ontology       Delta
------------------------------------------------------
ACCURACY                   0.7000        0.8500       +0.1500
...

Baseline Accuracy = 0.7000
Ontology Accuracy = 0.8500
Improvement = 21.43%

p-value = 0.031250
Ontology-guided retrieval improved performance by 21.43% over baseline with statistical significance
```

### Running a single query programmatically

```python
from dotenv import load_dotenv
load_dotenv()

from graph_pipeline import PipelineRunner

runner = PipelineRunner()

# Baseline
result = runner.run_single_query("What causes a heart attack?", pipeline="baseline")
print(result["answer"])

# Ontology-enhanced
result = runner.run_single_query("What causes a heart attack?", pipeline="ontology")
print(result["answer"])
print("Expanded terms:", result["expanded_terms"])
```

---

## Configuration

| Variable | Location | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | `.env` file | Google AI Studio API key (required) |
| `MODEL_NAME` | `gemini_client.py` | Gemini model used for generation (default: `gemini-2.0-flash`) |
| `EMBEDDING_MODEL` | `gemini_client.py` | Embedding model (default: `models/embedding-001`) |
| `TEMPERATURE` | `gemini_client.py` | LLM temperature for reproducibility (default: `0.0`) |
| `TOP_K_RETRIEVAL` | `gemini_client.py` | Number of documents retrieved per query (default: `5`) |

---

## Notes & Limitations

- **API costs** – every run makes roughly 120+ Gemini API calls (20 queries × 2 pipelines + evaluation). Monitor your usage quota.
- **Rate limiting** – a 2-second delay is applied between pipeline calls and a 1.5-second delay during evaluation to stay within free-tier limits. Adjust the `delay` parameters in `main.py` if needed.
- **Vector store rebuild** – ChromaDB is rebuilt from scratch on each run (`force_rebuild=True`). This ensures reproducibility but adds embedding time. Set `force_rebuild=False` in `graph_pipeline.py` to reuse an existing store.
- **Ontology scope** – `domain.ttl` covers a curated set of cardiovascular, respiratory, metabolic, neurological, and infectious disease concepts. Queries outside this domain will receive no ontology expansion.
- **Evaluation subjectivity** – the LLM-as-judge approach introduces non-determinism; results may vary slightly between runs even with `temperature=0.0`.
