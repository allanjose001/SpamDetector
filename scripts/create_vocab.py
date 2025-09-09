import csv
from collections import Counter, defaultdict

csv.field_size_limit(10**7)
csv_path = "../SpamDetector/data/processed/emails_train.csv"
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    all_texts = [(row['text'], int(row['spam'])) for row in reader]

doc_freq = Counter()
class_doc_freq = defaultdict(lambda: Counter())
n_spam = 0
n_ham = 0

for text, label in all_texts:
    tokens = [w for w in text.split() if len(w) >= 3]
    bigrams = ['{}_{}'.format(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
    all_terms = set(tokens + bigrams)
    doc_freq.update(all_terms)
    class_doc_freq[label].update(all_terms)
    if label == 1:
        n_spam += 1
    else:
        n_ham += 1

cd_scores = {}
for w in doc_freq:
    p_spam = class_doc_freq[1][w] / n_spam if n_spam > 0 else 0
    p_ham = class_doc_freq[0][w] / n_ham if n_ham > 0 else 0
    cd_scores[w] = abs(p_spam - p_ham)

# Ordena por CD decrescente
sorted_terms = sorted(cd_scores.items(), key=lambda x: -x[1])

with open("../SpamDetector/data/processed/vocab_cd.txt", "w", encoding="utf-8") as f:
    for word, score in sorted_terms:
        if score > 0.016:
            f.write(f"{word},{score:.4f}\n")