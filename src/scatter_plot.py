import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
import csv

# Caminhos dos arquivos
tfidf_path = r"../SpamDetector/data/processed/tfidf_sparse.npz"
vocab_path = r"../SpamDetector/data/processed/vocab.txt"
csv_path = r"../SpamDetector/data/processed/emails_cleaned.csv"

def load_vocab(vocab_path):
    vocab = []
    with open(vocab_path, 'r', encoding='utf-8') as vf:
        for line in vf:
            term = line.strip()
            if term:
                vocab.append(term)
    index = {term: i for i, term in enumerate(vocab)}
    return vocab, index

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

def plot_tfidf_bar_per_class(word):
    tfidf = load_npz(tfidf_path).toarray()
    vocab, vocab_index = load_vocab(vocab_path)
    labels = load_labels(csv_path)

    min_len = min(len(labels), tfidf.shape[0])
    labels = labels[:min_len]
    tfidf = tfidf[:min_len, :]

    if word not in vocab_index:
        print(f"Palavra '{word}' não encontrada no vocabulário.")
        return

    idx = vocab_index[word]
    tfidf_word = tfidf[:, idx]

    # Considera apenas linhas com TF-IDF não nulo para a palavra
    nonzero_idx = np.where(tfidf_word != 0)[0]
    labels_nonzero = labels[nonzero_idx]
    tfidf_word_nonzero = tfidf_word[nonzero_idx]

    spam_vals = tfidf_word_nonzero[labels_nonzero == 1]
    ham_vals = tfidf_word_nonzero[labels_nonzero == 0]

    bins = 30

    # Histograma para spam
    plt.figure(figsize=(8, 4))
    plt.hist(spam_vals, bins, color='red', alpha=0.7)
    plt.xlabel(f'TF-IDF de "{word}" (spam)')
    plt.ylabel('Frequência')
    plt.title(f'Histograma do TF-IDF da palavra "{word}" - Spam (apenas valores não nulos)')
    plt.tight_layout()
    plt.show()

    # Histograma para ham
    plt.figure(figsize=(8, 4))
    plt.hist(ham_vals, bins, color='blue', alpha=0.7)
    plt.xlabel(f'TF-IDF de "{word}" (ham)')
    plt.ylabel('Frequência')
    plt.title(f'Histograma do TF-IDF da palavra "{word}" - Ham (apenas valores não nulos)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    word = "money"
    plot_tfidf_bar_per_class(word)