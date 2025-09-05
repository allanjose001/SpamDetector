from pathlib import Path
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
from joblib import Parallel, delayed
import csv

# Calcula tf-idf para um documento usando contagem por documento (mais eficiente)
def tfidf_vector(doc, vocab, idf_dict):
    # Conta tokens no documento uma única vez
    counts = Counter(doc.split())
    vec = np.zeros(len(vocab), dtype=np.float32)
    for i, term in enumerate(vocab):
        c = counts.get(term, 0)
        if c > 0:
            tf_val = 1.0 + np.log(c)
            vec[i] = tf_val * idf_dict.get(term, 0.0)
    # Normaliza L2
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec

if __name__ == "__main__":
    import time

    # Parâmetros
    csv.field_size_limit(10**7)
    csv_path = r"c:/Users/Diogenes/Desktop/SpamDetector/data/processed/emails_cleaned.csv"
    max_lines = 3000        # processa até N linhas
    out_npz = r"c:/Users/Diogenes/Desktop/SpamDetector/data/processed/tfidf_sparse.npz"
    out_vocab = r"c:/Users/Diogenes/Desktop/SpamDetector/data/processed/vocab.txt"

    # Lê textos (até max_lines) com contador de progresso
    texts = []
    start = time.time()
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            texts.append(row.get('text', '') or '')
            if (i + 1) % 500 == 0:
                print(f"[read] {i+1} linhas lidas (máx {max_lines})...")
            if len(texts) >= max_lines:
                break
    print(f"[read] leitura concluída: {len(texts)} documentos em {time.time() - start:.1f}s")

    # Carrega vocabulário salvo se existir, caso contrário gera e salva
    vocab_path = Path(out_vocab)
    if vocab_path.exists():
        with open(vocab_path, 'r', encoding='utf-8') as vf:
            vocab = [line.strip() for line in vf if line.strip()]
        print(f"[vocab] carregado de: {vocab_path} (len={len(vocab)})")
    else:
        vocab = sorted({w for t in texts for w in t.split() if len(w) > 3})
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        with open(vocab_path, 'w', encoding='utf-8') as vf:
            for term in vocab:
                vf.write(term + '\n')
        print(f"[vocab] gerado e salvo em: {vocab_path} (len={len(vocab)})")

    # Calcula IDF de forma eficiente: uma passada pelos documentos para contar DF
    print(f"[idf] contando DF em {len(texts)} documentos...")
    t_idf_start = time.time()
    df_counter = Counter()
    for doc in texts:
        tokens = {tok.lower() for tok in doc.split()}  # conjunto por documento
        df_counter.update(tokens)
    n_docs = len(texts)
    idf_dict = {
        term: (np.log((n_docs + 1) / (df_counter.get(term, 0) + 1)) + 1.0) if n_docs > 0 else 0.0
        for term in vocab
    }
    print(f"[idf] concluído em {time.time() - t_idf_start:.1f}s")

    # Calcula vetores TF-IDF em batches com feedback de progresso
    print(f"[tfidf] calculando TF-IDF para {n_docs} documentos (vocab size={len(vocab)})...")
    t_tfidf_start = time.time()
    batch_size = 500   # ajuste para ter mais/menos granularidade no progresso
    rows = []
    for i in range(0, n_docs, batch_size):
        batch = texts[i:i + batch_size]
        t_batch_start = time.time()
        # Processa o batch em paralelo; verbose do joblib dará feedback interno
        batch_vecs = Parallel(n_jobs=-1, verbose=5)(
            delayed(tfidf_vector)(doc, vocab, idf_dict) for doc in batch
        )
        rows.extend(batch_vecs)
        t_batch = time.time() - t_batch_start
        print(f"[tfidf] processed {min(i + batch_size, n_docs)}/{n_docs} docs (batch {i//batch_size + 1}) in {t_batch:.1f}s")

    print(f"[tfidf] cálculo concluído em {time.time() - t_tfidf_start:.1f}s")

    # Converte para CSR e salva (usa float32 para reduzir memória/disk)
    print("[save] convertendo para CSR e salvando .npz ...")
    tfidf_array = np.vstack(rows).astype(np.float32)
    tfidf_sparse = csr_matrix(tfidf_array)
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    save_npz(out_npz, tfidf_sparse)
    print(f"[save] Saved TF-IDF matrix: {out_npz} (shape={tfidf_sparse.shape}, dtype={tfidf_sparse.dtype})")
    print(f"[save] Vocab used (len={len(vocab)}): {out_vocab}")