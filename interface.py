import streamlit as st
import numpy as np
from scipy.sparse import load_npz
from src.normalize_csv_text import normalize_text
from src.nb_model import NaiveBayes
from scripts.tf_idf import tfidf_vector
from scipy.stats import norm
from collections import Counter
import matplotlib.pyplot as plt
import pickle
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
        # Carrega vocabulário e idf_dict
        vocab, vocab_index = load_vocab("data/processed/vocab.txt")
        with open("data/processed/idf_dict.pkl", "rb") as f:
            idf_dict = pickle.load(f)
        # Calcula vetor TF-IDF
        tfidf_vec = tfidf_vector(norm_text, vocab, idf_dict)
        model = load_model()
        pred = model.predict([tfidf_vec])[0]
        explanation = model.explain(tfidf_vec, vocab)
        st.success("SPAM" if pred == 1 else "NÃO É SPAM")

        # --- Top 10 palavras mais influentes ---
        words = explanation['words']
        tfidfs = explanation['tfidf']
        classes = explanation['classes']
        chosen_class = int(pred)
        log_likelihoods = np.array(classes[chosen_class]['log_likelihoods'])

        # Ordena por influência (valor absoluto do log-likelihood)
        top_n = 10
        top_idx = np.argsort(np.abs(log_likelihoods))[::-1][:top_n]

        # Cálculo da contribuição percentual
        abs_log_likelihoods = np.abs(log_likelihoods)
        total_abs_log_likelihood = np.sum(abs_log_likelihoods)
        percent_contrib = 100 * abs_log_likelihoods / total_abs_log_likelihood if total_abs_log_likelihood > 0 else np.zeros_like(abs_log_likelihoods)

        word_counts = Counter(norm_text.split())

        st.subheader(f"Palavras mais influentes para a decisão ({'SPAM' if pred==1 else 'HAM'})")

        # Layout compacto: 2 colunas por linha
        cols = st.columns(2)
        for i, idx in enumerate(top_idx):
            word = words[idx]
            tfidf_value = tfidfs[idx]
            count = word_counts.get(word, 0)
            mu_spam = classes[1]['mean'][idx]
            std_spam = classes[1]['std'][idx]
            mu_ham = classes[0]['mean'][idx]
            std_ham = classes[0]['std'][idx]
            z_spam = (tfidf_value - mu_spam) / std_spam if std_spam > 1e-8 else 0
            z_ham = (tfidf_value - mu_ham) / std_ham if std_ham > 1e-8 else 0
            log_l = log_likelihoods[idx]
            pct = percent_contrib[idx]
            with cols[i % 2]:
                st.markdown(
                    f"**{word}**  \n"
                    f"- Contagem: `{count}`  \n"
                    f"- TF-IDF: `{tfidf_value:.4f}`  \n"
                    f"- Média (spam): `{mu_spam:.4f}` | σ: `{std_spam:.4f}`  \n"
                    f"- Média (ham): `{mu_ham:.4f}` | σ: `{std_ham:.4f}`  \n"
                    f"- Z-score (spam): `{z_spam:.2f}`  \n"
                    f"- Z-score (ham): `{z_ham:.2f}`  \n"
                    f"- Log-likelihood: `{log_l:.4f}`  \n"
                    f"- **Contribuição para decisão:** `{pct:.1f}%`"
                )

        # --- Gráfico para a palavra mais relevante ---
        if len(top_idx) > 0:
            idx = top_idx[0]
            word = words[idx]
            tfidf_value = tfidfs[idx]
            mu_spam = classes[1]['mean'][idx]
            std_spam = classes[1]['std'][idx]
            mu_ham = classes[0]['mean'][idx]
            std_ham = classes[0]['std'][idx]

            x_min = 0
            x_max = max(mu_spam + 3*std_spam, mu_ham + 3*std_ham, tfidf_value + 0.1)
            x = np.linspace(x_min, x_max, 200)
            likelihood_spam = norm.pdf(x, mu_spam, std_spam)
            likelihood_ham = norm.pdf(x, mu_ham, std_ham)
            ratio = np.divide(likelihood_spam, likelihood_ham, out=np.zeros_like(likelihood_spam), where=likelihood_ham!=0)

            with st.container():
                fig, ax1 = plt.subplots(figsize=(7, 4))
                ax1.plot(x, likelihood_spam, label=f"Spam (μ={mu_spam:.3f}, σ={std_spam:.3f})", color='red')
                ax1.plot(x, likelihood_ham, label=f"Ham (μ={mu_ham:.3f}, σ={std_ham:.3f})", color='blue')
                ax1.axvline(tfidf_value, color='green', linestyle='--', label=f"TF-IDF no email: {tfidf_value:.3f}")
                ax1.set_xlabel("TF-IDF")
                ax1.set_ylabel("Verossimilhança (PDF)")
                ax1.legend(loc="upper left")
                ax1.grid(True, linestyle='--', alpha=0.5)

                ax2 = ax1.twinx()
                ax2.plot(x, ratio, label="Spam/Ham", color='black', linestyle='--')
                ax2.set_ylabel("Razão Spam/Ham")
                ax2.set_yscale("log")
                ax2.legend(loc="upper right")

                plt.title(f"Verossimilhança para '{word}' (palavra mais relevante)")
                plt.tight_layout()
                st.pyplot(fig)