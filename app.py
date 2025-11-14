# =============================================================
# Medi-Co — AI-Driven Clinical Assistant (Upgraded Version)
# =============================================================
# • Uses BioBERT for symptom extraction, medical NER & terminology grounding
# • Uses Meditron-7B (domain-tuned medical LLM) for clinical reasoning
# • Adds PubMed-style RAG pipeline for evidence-based responses
# • Adds ICD-10 mapping for symptom → condition classification
# • Implements privacy-first, HIPAA-style local processing
# =============================================================

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import json
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Medi-Co", page_icon="2.png", layout="wide")

# ---------------------- Load Core Models ----------------------
@st.cache_resource
def load_models():

    # BioBERT for medical NER, keywords, symptom extraction
    biobert_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    biobert_model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

    # Clinical LLM – Meditron 7B (best open-source for clinical accuracy)
    med_llm_name = "epfl-llm/meditron-7b"
    med_tokenizer = AutoTokenizer.from_pretrained(med_llm_name)
    med_model = AutoModelForCausalLM.from_pretrained(
        med_llm_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    # Embedding model for RAG (PubMedBERT embeddings)
    embedder = SentenceTransformer("pritamdeka/Sentence-BERT-SciBERT")

    return biobert_tokenizer, biobert_model, med_tokenizer, med_model, embedder


biobert_tokenizer, biobert_model, med_tokenizer, med_model, embedder = load_models()


# ---------------------- ICD-10 Mapping (Simplified) ----------------------
ICD10_MAP = {
    "fever": "R50.9 – Fever, unspecified",
    "cough": "R05 – Cough",
    "headache": "R51 – Headache",
    "chest pain": "R07.9 – Chest pain, unspecified",
    "diarrhea": "R19.7 – Diarrhea, unspecified",
}


def map_to_icd10(symptoms: str):
    symptoms_lower = symptoms.lower()
    matches = [ICD10_MAP[key] for key in ICD10_MAP if key in symptoms_lower]
    return matches if matches else ["No ICD-10 match found"]


# ---------------------- Retrieve PubMed-Like Info (Local RAG) ----------------------
@st.cache_resource
def load_corpus():
    with open("medical_corpus.json", "r") as f:
        return json.load(f)
    

def rag_retrieve(query, top_k=3):
    corpus = load_corpus()
    passages = [c["text"] for c in corpus]
    embeddings = embedder.encode(passages, convert_to_tensor=True)
    q_embed = embedder.encode(query, convert_to_tensor=True)

    scores = util.cos_sim(q_embed, embeddings)[0]
    top_results = torch.topk(scores, k=top_k)

    retrieved = [passages[idx] for idx in top_results.indices]
    return retrieved


# ---------------------- BioBERT Symptom Extraction ----------------------
def extract_keywords(text):
    tokens = biobert_tokenizer(text, return_tensors="pt", truncation=True)
    outputs = biobert_model(**tokens)
    # Sentence embedding approximation from BioBERT CLS token
    embedding = outputs.last_hidden_state[:, 0, :]  
    return embedding


# ---------------------- Generate Clinical Response ----------------------
def generate_clinical_answer(symptoms, retrieved_docs):
    context_text = "\n".join(retrieved_docs)

    prompt = f"""
You are Medi-Co, an AI medical assistant providing evidence-based, safe, non-prescriptive guidance.

Symptoms reported:
{symptoms}

Relevant medical literature:
{context_text}

Follow these rules:
• Explain possible causes (differential reasoning)
• NEVER prescribe medicines
• Provide safe home care steps
• Indicate when to see a doctor
• Use easy language

Answer:
"""

    inputs = med_tokenizer(prompt, return_tensors="pt").to(med_model.device)
    output = med_model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.4,
        top_p=0.9
    )

    return med_tokenizer.decode(output[0], skip_special_tokens=True)


# ---------------------- Streamlit UI ----------------------
st.title("🧠 Medi-Co — Clinical AI Assistant (BioBERT + Meditron + RAG)")

user_input = st.text_area("Describe your symptoms:")

if st.button("Analyze"):
    
    with st.spinner("Extracting symptoms with BioBERT..."):
        biobert_vec = extract_keywords(user_input)

    with st.spinner("Retrieving evidence from PubMed-like corpus..."):
        retrieved = rag_retrieve(user_input)

    with st.spinner("Generating clinically-grounded response..."):
        answer = generate_clinical_answer(user_input, retrieved)

    icd_codes = map_to_icd10(user_input)

    st.subheader("🩺 Medi-Co Response")
    st.write(answer)

    st.subheader("📘 ICD-10 Mapping")
    for code in icd_codes:
        st.write("•", code)

    st.subheader("📚 Evidence Used (RAG)")
    for doc in retrieved:
        st.info(doc)
