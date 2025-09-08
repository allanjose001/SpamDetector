import csv
from collections import Counter, defaultdict

csv.field_size_limit(10**7)
csv_path = "../SpamDetector/data/processed/emails_cleaned.csv"
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    all_texts = [(row['text'], int(row['spam'])) for row in reader]

# Conta em quantos documentos cada palavra aparece no total e por classe
doc_freq = Counter()
class_doc_freq = defaultdict(lambda: Counter())

for text, label in all_texts:
    tokens = set([w for w in text.split() if len(w) >= 3])
    doc_freq.update(tokens)
    class_doc_freq[label].update(tokens)

MIN_DOCS = 3  # mínimo de documentos totais
MIN_CLASS_DOCS = 2  # mínimo de documentos em cada classe

vocab = [
    w for w, freq in doc_freq.items()
    if freq >= MIN_DOCS
    and class_doc_freq[0][w] >= MIN_CLASS_DOCS
    and class_doc_freq[1][w] >= MIN_CLASS_DOCS
]

# Salva o vocabulário em um arquivo .txt
with open("../SpamDetector/data/processed/vocab.txt", "w", encoding="utf-8") as f:
    for word in vocab:
        f.write(word + "\n")