import streamlit as st
import numpy as np
from scipy.sparse import load_npz
from src.normalize_csv_text import normalize_text
from src.nb_model import NaiveBayes
from scripts.tf_idf import tfidf_vector
import csv
import joblib

# Carregue o vocabulário
def load_vocab(vocab_path):
    with open(vocab_path, encoding="utf-8") as f:
        vocab = [line.strip() for line in f if line.strip()]
    vocab_index = {w: i for i, w in enumerate(vocab)}
    return vocab, vocab_index

# Carregue o modelo treinado (ou treine rapidamente)
@st.cache_resource
def load_model():
    # Carregue a matriz TF-IDF e os rótulos
    tfidf = load_npz("data/processed/tfidf_sparse.npz").toarray()
    n_samples = tfidf.shape[0]
    labels = []
    with open("data/processed/emails_cleaned.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n_samples:
                break
            labels.append(int(row["spam"]))
    labels = np.array(labels)
    model = NaiveBayes()
    model.fit(tfidf, labels)
    return model

st.title("Detector de Spam")

email = st.text_area("Cole o texto do email:")

if st.button("Analisar"):
    if not email.strip():
        st.warning("Digite um email para analisar.")
    else:
        # Normaliza
        norm_text = normalize_text(email)
        # Carrega vocabulário
        vocab, vocab_index = load_vocab("data/processed/vocab.txt")
        # Carrega o idf_dict salvo
        import pickle
        with open("data/processed/idf_dict.pkl", "rb") as f:
            idf_dict = pickle.load(f)
        # transforma em vetor tf-idf
        tfidf_vec = tfidf_vector(norm_text, vocab, idf_dict)
        model = load_model()
        pred = model.predict([tfidf_vec])[0]
        st.success("SPAM" if pred == 1 else "NÃO É SPAM")