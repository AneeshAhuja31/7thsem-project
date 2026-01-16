"""
utils.py
Shared utilities: corpus loading, vector store construction, and test queries.
"""

import os
import shutil
import pandas as pd
from langchain_chroma import Chroma
from langchain.schema import Document
from gemini_client import get_embeddings, TOP_K_RETRIEVAL

CHROMA_COLLECTION = "medical_corpus"
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")


def load_corpus(csv_path: str = None) -> pd.DataFrame:
    """Load the medical corpus CSV."""
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "data", "corpus.csv")
    return pd.read_csv(csv_path)


def build_vectorstore(corpus_df: pd.DataFrame, force_rebuild: bool = False) -> Chroma:
    """
    Build a ChromaDB vector store from the corpus DataFrame.
    Uses GoogleGenerativeAIEmbeddings for consistency across pipelines.
    """
    if force_rebuild and os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    documents = []
    for _, row in corpus_df.iterrows():
        doc = Document(
            page_content=row["text"],
            metadata={
                "id": str(row["id"]),
                "category": row["category"],
                "ground_truth_concepts": row["ground_truth_concepts"],
            },
        )
        documents.append(doc)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore


def get_test_queries() -> list[dict]:
    """
    Return 20 test queries grouped into four categories:
      - 5 equivalence-dependent  (lay term -> clinical term)
      - 5 subclass-dependent     (parent -> child expansion)
      - 5 semantic-neighbor      (symptom/treatment traversal)
      - 5 neutral / control      (exact clinical terms)
    """
    return [
        # ── Equivalence-dependent (1-5) ──
        {
            "query": "What are the treatments for high blood pressure?",
            "expected_answer": (
                "Hypertension is treated with antihypertensive medications including "
                "ACE inhibitors, beta-blockers, calcium channel blockers, and lifestyle "
                "modifications such as reduced sodium intake and regular exercise."
            ),
            "expected_concepts": [
                "Hypertension", "Antihypertensive", "ACEInhibitor", "BetaBlocker",
            ],
        },
        {
            "query": "What causes a heart attack and how is it treated?",
            "expected_answer": (
                "Myocardial infarction is caused by blocked blood flow to the heart "
                "muscle, typically due to atherosclerotic plaque rupture. Treatment "
                "includes anticoagulants, angioplasty, and bypass surgery."
            ),
            "expected_concepts": [
                "MyocardialInfarction", "Anticoagulant", "Angioplasty", "BypassSurgery",
            ],
        },
        {
            "query": "How do you manage shortness of breath in patients?",
            "expected_answer": (
                "Dyspnea management depends on the underlying cause. For asthma and "
                "COPD, bronchodilators are used. For heart failure, ACE inhibitors and "
                "diuretics are prescribed. Oxygen therapy may be needed."
            ),
            "expected_concepts": [
                "Dyspnea", "Asthma", "COPD", "HeartFailure", "Bronchodilator",
            ],
        },
        {
            "query": "What is high cholesterol and how is it diagnosed?",
            "expected_answer": (
                "Hyperlipidemia is diagnosed through a fasting lipid panel blood test "
                "measuring total cholesterol, LDL, HDL, and triglycerides. It is a "
                "risk factor for atherosclerosis and coronary artery disease."
            ),
            "expected_concepts": [
                "Hyperlipidemia", "LipidPanel", "Atherosclerosis", "Statin",
            ],
        },
        {
            "query": "What is the flu and what antiviral drugs treat it?",
            "expected_answer": (
                "Influenza is an acute respiratory infection caused by influenza viruses. "
                "Antiviral medications such as oseltamivir (Tamiflu) are effective when "
                "administered within 48 hours of symptom onset."
            ),
            "expected_concepts": [
                "Influenza", "Antiviral", "Fever", "Cough",
            ],
        },
        # ── Subclass-dependent (6-10) ──
        {
            "query": "What are the different types of cardiovascular diseases?",
            "expected_answer": (
                "Cardiovascular diseases include hypertension, coronary artery disease, "
                "heart failure, arrhythmia, atherosclerosis, myocardial infarction, "
                "and stroke."
            ),
            "expected_concepts": [
                "CardiovascularDisease", "Hypertension", "CoronaryArteryDisease",
                "HeartFailure", "Arrhythmia",
            ],
        },
        {
            "query": "What lung diseases exist and how are they diagnosed?",
            "expected_answer": (
                "Respiratory diseases include asthma, COPD, pneumonia, bronchitis, "
                "and lung cancer. Diagnosis uses spirometry, chest X-ray, CT scan, "
                "and pulmonary function tests."
            ),
            "expected_concepts": [
                "RespiratoryDisease", "Asthma", "COPD", "Pneumonia", "Spirometry",
            ],
        },
        {
            "query": "What metabolic diseases are related to obesity?",
            "expected_answer": (
                "Obesity is associated with type 2 diabetes, hypertension, "
                "hyperlipidemia, and metabolic syndrome. These conditions share "
                "common pathways including insulin resistance and inflammation."
            ),
            "expected_concepts": [
                "MetabolicDisease", "Obesity", "Type2Diabetes", "Hyperlipidemia",
                "MetabolicSyndrome",
            ],
        },
        {
            "query": "What types of medications are used for mental health?",
            "expected_answer": (
                "Mental health medications include antidepressants (SSRIs, SNRIs) for "
                "depression and anxiety, antipsychotics for schizophrenia and bipolar "
                "disorder, and anticonvulsants as mood stabilizers."
            ),
            "expected_concepts": [
                "Antidepressant", "Antipsychotic", "Anticonvulsant",
                "Depression", "Schizophrenia",
            ],
        },
        {
            "query": "What are the common infectious diseases and their treatments?",
            "expected_answer": (
                "Common infectious diseases include influenza, tuberculosis, COVID-19, "
                "hepatitis, and HIV. Treatments include antiviral medications and "
                "antibiotics depending on the pathogen."
            ),
            "expected_concepts": [
                "InfectiousDisease", "Influenza", "Tuberculosis", "COVID19",
                "Antiviral", "Antibiotic",
            ],
        },
        # ── Semantic-neighbor (11-15) ──
        {
            "query": "What conditions can cause dizziness?",
            "expected_answer": (
                "Dizziness can be caused by hypertension, cardiac arrhythmias, "
                "epilepsy, migraine, and stroke. It may also result from inner "
                "ear disorders and medication side effects."
            ),
            "expected_concepts": [
                "Dizziness", "Hypertension", "Arrhythmia", "Epilepsy", "Migraine",
            ],
        },
        {
            "query": "What diseases are related to hypertension?",
            "expected_answer": (
                "Hypertension is related to coronary artery disease, stroke, heart "
                "failure, atherosclerosis, chronic kidney disease, and is a risk "
                "factor for myocardial infarction."
            ),
            "expected_concepts": [
                "Hypertension", "CoronaryArteryDisease", "Stroke", "HeartFailure",
                "Atherosclerosis",
            ],
        },
        {
            "query": "What are the symptoms and treatments of type 2 diabetes?",
            "expected_answer": (
                "Type 2 diabetes symptoms include fatigue, weight loss, and increased "
                "thirst. Treatment involves insulin therapy, oral hypoglycemics, "
                "lifestyle modifications, and regular blood glucose monitoring."
            ),
            "expected_concepts": [
                "Type2Diabetes", "Fatigue", "Insulin", "BloodGlucoseTest",
            ],
        },
        {
            "query": "What diagnostic tests are used for heart problems?",
            "expected_answer": (
                "Cardiac diagnostic tests include ECG for arrhythmias and ischemia, "
                "echocardiogram for heart failure and valvular disease, blood tests "
                "for cardiac biomarkers, and cardiac catheterization."
            ),
            "expected_concepts": [
                "ECG", "Echocardiogram", "BloodTest", "HeartFailure", "Arrhythmia",
            ],
        },
        {
            "query": "How does depression relate to other medical conditions?",
            "expected_answer": (
                "Depression is commonly comorbid with cardiovascular disease, diabetes, "
                "and chronic pain. It is related to anxiety disorders and insomnia. "
                "Treatment includes antidepressants and cognitive behavioral therapy."
            ),
            "expected_concepts": [
                "Depression", "AnxietyDisorder", "Insomnia", "Antidepressant",
                "CognitiveBehavioralTherapy",
            ],
        },
        # ── Neutral / control (16-20) ──
        {
            "query": "How is pneumonia diagnosed and treated?",
            "expected_answer": (
                "Pneumonia is diagnosed with chest X-ray and clinical assessment. "
                "Treatment involves antibiotic therapy for bacterial pneumonia, "
                "supportive care, and oxygen therapy for severe cases."
            ),
            "expected_concepts": [
                "Pneumonia", "XRay", "Antibiotic", "Fever", "Cough",
            ],
        },
        {
            "query": "What is Alzheimer's disease and how is it managed?",
            "expected_answer": (
                "Alzheimer's disease is a progressive neurodegenerative disorder "
                "causing memory loss and cognitive decline. Management includes "
                "cholinesterase inhibitors, occupational therapy, and MRI for diagnosis."
            ),
            "expected_concepts": [
                "Alzheimers", "MemoryLoss", "MRI", "OccupationalTherapy",
            ],
        },
        {
            "query": "What is COPD and what medications are used?",
            "expected_answer": (
                "COPD is a chronic obstructive pulmonary disease characterized by "
                "persistent airflow limitation. Treatment includes bronchodilators, "
                "inhaled corticosteroids, and pulmonary rehabilitation."
            ),
            "expected_concepts": [
                "COPD", "Bronchodilator", "Spirometry", "Dyspnea",
            ],
        },
        {
            "query": "How is epilepsy treated with anticonvulsant medications?",
            "expected_answer": (
                "Epilepsy is treated with anticonvulsant medications including "
                "levetiracetam, lamotrigine, and valproic acid. MRI and EEG are "
                "used for diagnosis. Surgery is an option for refractory cases."
            ),
            "expected_concepts": [
                "Epilepsy", "Anticonvulsant", "MRI",
            ],
        },
        {
            "query": "What is coronary artery disease and how is it treated?",
            "expected_answer": (
                "Coronary artery disease results from atherosclerotic plaque buildup "
                "in coronary arteries. Treatment includes statins, anticoagulants, "
                "angioplasty, and coronary artery bypass surgery."
            ),
            "expected_concepts": [
                "CoronaryArteryDisease", "Statin", "Angioplasty", "BypassSurgery",
            ],
        },
        {
            "query": "What are the symptoms and treatment of heart failure?",
            "expected_answer": (
                "Heart failure symptoms include dyspnea, edema, and fatigue. "
                "Treatment involves ACE inhibitors, beta-blockers, diuretics, "
                "and in advanced cases cardiac devices or transplantation."
            ),
            "expected_concepts": [
                "HeartFailure", "Dyspnea", "Edema", "ACEInhibitor", "BetaBlocker",
            ],
        },
    ]
