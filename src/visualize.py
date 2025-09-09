import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse import load_npz
import csv
from scipy.stats import norm

# Caminhos padrão (ajuste se necessário)
DEFAULT_TFIDF = r"../SpamDetector/data/processed/tfidf_sparse.npz"
DEFAULT_VOCAB = r"../SpamDetector/data/processed/vocab.txt"
DEFAULT_CSV = r"../SpamDetector/data/processed/emails_train.csv"

def load_labels(csv_path, label_field='spam'):
    labels = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                labels.append(int(row[label_field]))
            except (KeyError, ValueError):
                continue
    return np.array(labels, dtype=np.int8)

def load_vocab(vocab_path):
    vocab = []
    with open(vocab_path, 'r', encoding='utf-8') as vf:
        for line in vf:
            parts = line.strip().split('\t')
            if parts and parts[0]:
                vocab.append(parts[0])
    index = {term: i for i, term in enumerate(vocab)}
    return vocab, index

def plot_likelihood(word, tfidf_values_spam, tfidf_values_ham):
    tfidf_values_spam = tfidf_values_spam[tfidf_values_spam > 0]
    tfidf_values_ham = tfidf_values_ham[tfidf_values_ham > 0]

    if len(tfidf_values_spam) == 0 or len(tfidf_values_ham) == 0:
        print(f"[warn] Sem exemplos não nulos para '{word}' em ambas as classes.")
        return

    mu_spam, std_spam = np.mean(tfidf_values_spam), np.std(tfidf_values_spam)
    mu_ham, std_ham = np.mean(tfidf_values_ham), np.std(tfidf_values_ham)

    x_min = min(tfidf_values_spam.min(), tfidf_values_ham.min())
    x_max = max(tfidf_values_spam.max(), tfidf_values_ham.max())
    x = np.linspace(x_min, x_max, 200)

    likelihood_spam = norm.pdf(x, mu_spam, std_spam)
    likelihood_ham = norm.pdf(x, mu_ham, std_ham)
    ratio = np.divide(likelihood_spam, likelihood_ham, out=np.zeros_like(likelihood_spam), where=likelihood_ham!=0)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(x, likelihood_spam, label=f"Spam (μ={mu_spam:.3f}, σ={std_spam:.3f})", color='red')
    ax1.plot(x, likelihood_ham, label=f"Ham (μ={mu_ham:.3f}, σ={std_ham:.3f})", color='blue')
    ax1.set_xlabel("TF-IDF")
    ax1.set_ylabel("Verossimilhança (PDF)")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(x, ratio, label="Spam/Ham", color='black', linestyle='--')
    ax2.set_ylabel("Razão Spam/Ham")
    ax2.set_yscale("log")
    ax2.legend(loc="upper right")

    plt.title(f"Verossimilhança para a palavra '{word}' (TF-IDF > 0)")
    plt.tight_layout()
    plt.savefig(f"{word}_likelihood.png")  # <-- ALTERAÇÃO: salva o gráfico
    plt.close(fig)  # <-- ALTERAÇÃO: fecha o gráfico para liberar memória

def main(tfidf_path, vocab_path, csv_path, words, label_field='spam'):
    # Carrega recursos
    print("[load] carregando TF-IDF...")
    tfidf = load_npz(tfidf_path).tocsr()
    print(f"[load] TF-IDF shape: {tfidf.shape}")

    print("[load] carregando vocab...")
    vocab, vocab_index = load_vocab(vocab_path)
    print(f"[load] vocab len: {len(vocab)}")

    print("[load] carregando rótulos...")
    labels = load_labels(csv_path, label_field)
    print(f"[load] rótulos len: {len(labels)}, valores únicos: {np.unique(labels)}")

    for word in words:
        if word not in vocab_index:
            print(f"[warn] palavra não encontrada no vocab: {word}")
            continue
        idx = vocab_index[word]
        col = tfidf.getcol(idx).toarray().ravel()
        # Ajusta comprimento se necessário
        m = min(len(col), len(labels))
        col = col[:m]
        labels_cut = labels[:m]
        tfidf_spam = col[labels_cut == 1]
        tfidf_ham = col[labels_cut == 0]
        if len(tfidf_spam) == 0 or len(tfidf_ham) == 0:
            print(f"[warn] Sem exemplos suficientes para '{word}' em ambas as classes.")
            continue
        print(f"[info] '{word}': spam μ={np.mean(tfidf_spam):.4f}, σ={np.std(tfidf_spam):.4f} | ham μ={np.mean(tfidf_ham):.4f}, σ={np.std(tfidf_ham):.4f}")
        plot_likelihood(word, tfidf_spam, tfidf_ham)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analisa verossimilhança de palavras usando TF-IDF e distribuição normal")
    parser.add_argument("--tfidf", default=DEFAULT_TFIDF, help="Caminho .npz do TF-IDF")
    parser.add_argument("--vocab", default=DEFAULT_VOCAB, help="Caminho do vocab.txt")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="CSV com rótulos (campo 'spam')")
    parser.add_argument(
    "--words",
    default="money",
        help="Lista de palavras separadas por vírgula"
    )
    parser.add_argument("--label-field", default="spam", help="Nome do campo de rótulo no CSV")
    args = parser.parse_args()

    words = [w.strip() for w in args.words.split(",") if w.strip()]
    main(args.tfidf, args.vocab, args.csv, words, label_field=args.label_field)

    #free,win,offer