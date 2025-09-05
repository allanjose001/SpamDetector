import argparse
import numpy as np
from pathlib import Path
from scipy.sparse import load_npz
import csv

DEFAULT_TFIDF = r"c:/Users/Diogenes/Desktop/SpamDetector/data/processed/tfidf_sparse.npz"
DEFAULT_VOCAB = r"c:/Users/Diogenes/Desktop/SpamDetector/data/processed/vocab.txt"
DEFAULT_CSV = r"c:/Users/Diogenes/Desktop/SpamDetector/data/processed/emails_cleaned.csv"
DEFAULT_OUT = r"c:/Users/Diogenes/Desktop/SpamDetector/data/processed/words_with_examples.txt"

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
            term = line.strip()
            if term:
                vocab.append(term)
    index = {term: i for i, term in enumerate(vocab)}
    return vocab, index

def main(tfidf_path, vocab_path, csv_path, out_path, label_field='spam'):
    print("[load] carregando TF-IDF...")
    tfidf = load_npz(tfidf_path).tocsr()
    print(f"[load] TF-IDF shape: {tfidf.shape}")

    print("[load] carregando vocab...")
    vocab, vocab_index = load_vocab(vocab_path)
    print(f"[load] vocab len: {len(vocab)}")

    print("[load] carregando rótulos...")
    labels = load_labels(csv_path, label_field)
    print(f"[load] rótulos len: {len(labels)}, valores únicos: {np.unique(labels)}")

    words_with_examples = []
    for word in vocab:
        idx = vocab_index[word]
        col = tfidf.getcol(idx).toarray().ravel()
        m = min(len(col), len(labels))
        col = col[:m]
        labels_cut = labels[:m]
        tfidf_spam = col[labels_cut == 1]
        tfidf_ham = col[labels_cut == 0]
        # Filtra apenas valores > 0
        tfidf_spam = tfidf_spam[tfidf_spam > 0]
        tfidf_ham = tfidf_ham[tfidf_ham > 0]
        if len(tfidf_spam) > 0 and len(tfidf_ham) > 0:
            freq_spam = len(tfidf_spam)
            freq_ham = len(tfidf_ham)
            words_with_examples.append((word, freq_spam, freq_ham))

    # Ordena pela diferença de frequência (spam - ham), do maior para o menor
    words_with_examples.sort(key=lambda x: x[1] - x[2], reverse=True)

    print(f"[result] {len(words_with_examples)} palavras com exemplos não nulos em ambas as classes.")
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["palavra", "freq_spam", "freq_ham", "diff_freq", "total_freq"])
        for w, freq_spam, freq_ham in words_with_examples:
            writer.writerow([w, freq_spam, freq_ham, freq_spam - freq_ham, freq_spam + freq_ham])
    print(f"[save] Lista salva em: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lista palavras com exemplos não nulos em ambas as classes e suas frequências")
    parser.add_argument("--tfidf", default=DEFAULT_TFIDF, help="Caminho .npz do TF-IDF")
    parser.add_argument("--vocab", default=DEFAULT_VOCAB, help="Caminho do vocab.txt")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="CSV com rótulos (campo 'spam')")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Arquivo de saída .txt")
    parser.add_argument("--label-field", default="spam", help="Nome do campo de rótulo no CSV")
    args = parser.parse_args()

    main(args.tfidf, args.vocab, args.csv, args.out, label_field=args.label_field)