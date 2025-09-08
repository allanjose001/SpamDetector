import streamlit as st
import numpy as np
from scipy.sparse import load_npz
from src.normalize_csv_text import normalize_text
from src.nb_model import NaiveBayes
from scripts.tf_idf import tfidf_vector
import pickle
import csv
import pandas as pd

def load_vocab(vocab_path):
    with open(vocab_path, encoding="utf-8") as f:
        vocab = [line.strip() for line in f if line.strip()]
    vocab_index = {w: i for i, w in enumerate(vocab)}
    return vocab, vocab_index

@st.cache_resource
def load_model():
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
        norm_text = normalize_text(email)
        vocab, vocab_index = load_vocab("data/processed/vocab.txt")
        with open("data/processed/idf_dict.pkl", "rb") as f:
            idf_dict = pickle.load(f)
        tfidf_vec = tfidf_vector(norm_text, vocab, idf_dict)
        model = load_model()
        pred = model.predict([tfidf_vec])[0]
        st.success("SPAM" if pred == 1 else "NÃO É SPAM")

        mu0 = model.feature_params[0]["mean"]
        var0 = model.feature_params[0]["std"] ** 2
        mu1 = model.feature_params[1]["mean"]
        var1 = model.feature_params[1]["std"] ** 2

        # Carrega matriz TF-IDF e rótulos completos
        tfidf_matrix = load_npz("data/processed/tfidf_sparse.npz").toarray()
        labels = []
        with open("data/processed/emails_cleaned.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.append(int(row["spam"]))
        labels = np.array(labels)

        # --- ajuste para garantir tamanhos iguais ---
        min_len = min(len(labels), tfidf_matrix.shape[0])
        labels = labels[:min_len]
        tfidf_matrix = tfidf_matrix[:min_len, :]

        # Coleta os dados explicativos em uma lista
        explicativos = []
        n_features = tfidf_matrix.shape[1]
        for i, tfidf in enumerate(tfidf_vec):
            if i >= n_features:
                continue
            if tfidf > 0 and vocab[i]:
                # Calcula média e variância APENAS dos tf-idf não nulos na base de treinamento para cada classe
                tfidf_spam = tfidf_matrix[labels == 1, i]
                tfidf_spam_nz = tfidf_spam[tfidf_spam > 0]
                tfidf_ham = tfidf_matrix[labels == 0, i]
                tfidf_ham_nz = tfidf_ham[tfidf_ham > 0]

                # Se não houver valores não nulos, usa fallback dos parâmetros do modelo
                if pred == 1:
                    mu_pred = np.mean(tfidf_spam_nz) if len(tfidf_spam_nz) > 0 else mu1[i]
                    var_pred = np.var(tfidf_spam_nz) if len(tfidf_spam_nz) > 0 else var1[i]
                    mu_other = np.mean(tfidf_ham_nz) if len(tfidf_ham_nz) > 0 else mu0[i]
                    var_other = np.var(tfidf_ham_nz) if len(tfidf_ham_nz) > 0 else var0[i]
                else:
                    mu_pred = np.mean(tfidf_ham_nz) if len(tfidf_ham_nz) > 0 else mu0[i]
                    var_pred = np.var(tfidf_ham_nz) if len(tfidf_ham_nz) > 0 else var0[i]
                    mu_other = np.mean(tfidf_spam_nz) if len(tfidf_spam_nz) > 0 else mu1[i]
                    var_other = np.var(tfidf_spam_nz) if len(tfidf_spam_nz) > 0 else var1[i]

                var_pred = var_pred if var_pred > 0 else 1e-6
                var_other = var_other if var_other > 0 else 1e-6

                likelihood_pred = (1.0 / np.sqrt(2 * np.pi * var_pred)) * np.exp(-((tfidf - mu_pred) ** 2) / (2 * var_pred))
                likelihood_other = (1.0 / np.sqrt(2 * np.pi * var_other)) * np.exp(-((tfidf - mu_other) ** 2) / (2 * var_other))

                explicativos.append({
                    "Palavra": vocab[i],
                    "TF-IDF": round(tfidf, 4),
                    "Verossimilhança": round(likelihood_pred, 6),
                    "Diferença Verossimilhança": round(likelihood_pred - likelihood_other, 6)
                })

        # Ordena por diferença de verossimilhança conforme a classe prevista
        if pred == 1:
            explicativos.sort(key=lambda x: -x["Diferença Verossimilhança"])
        else:
            explicativos.sort(key=lambda x: x["Diferença Verossimilhança"])

        st.subheader("Palavras mais influentes para a decisão")
        if explicativos:
            df = pd.DataFrame(explicativos)
            st.dataframe(df)
        else:
            st.info("Nenhuma palavra explicativa encontrada para este email.")
