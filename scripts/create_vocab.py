import csv

csv.field_size_limit(10**7)

csv_path = "c:/Users/Diogenes/Desktop/SpamDetector/data/processed/emails_cleaned.csv"
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    all_texts = [row['text'] for row in reader]
vocab = list(set([w for text in all_texts for w in text.split() if len(w) >= 3]))

# Salva o vocabulário em um arquivo .txt
with open("c:/Users/Diogenes/Desktop/SpamDetector/data/processed/vocab.txt", "w", encoding="utf-8") as f:
    for word in vocab:
        f.write(word + "\n")