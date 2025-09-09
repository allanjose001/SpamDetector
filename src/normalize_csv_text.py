import unicodedata
import re
import csv
import argparse
import random
from pathlib import Path

import nltk
from nltk.stem import SnowballStemmer
nltk.download('punkt', quiet=True)
stemmer = SnowballStemmer("english")

csv.field_size_limit(10**7)

# Compila expressões regulares usadas
RE_SUBJECT = re.compile(r'^\s*(subject|re|fw|fwd)[:\-\s]+', flags=re.I)
RE_URL = re.compile(r'https?://\S+|www\.\S+', flags=re.I)
RE_EMAIL = re.compile(r'\S+@\S+')
RE_HTML = re.compile(r'<[^>]+>')
RE_URL_LOOSE = re.compile(r'(?:https?[:\s]*[/\\]{0,2}|www[.\s])[\w\.\-\/\\\s]{5,}', flags=re.I)
RE_EMAIL_LOOSE = re.compile(r'[\w\.\-]+\s*@\s*[\w\.\-]+(?:\s*\.\s*\w{2,})+', flags=re.I)
RE_NON_ALNUM = re.compile(r"[^a-z0-9'\s]")   # mantém apóstrofo
RE_MULTI_SPACE = re.compile(r'\s+')
RE_REPEAT_CHARS = re.compile(r'(.)\1{2,}')   # colapsa repetições longas
RE_REPEAT_PUNCT = re.compile(r"(([!?\-])(\s*\2){1,})")
RE_EXACT_RE = re.compile(r'\sre\s:', flags=re.I)
RE_NAME = re.compile(r"\b(vince|larrissa|kaminski|sharma|joe|john|martin|carr|collins|stephen|bennett|norma|mike|roberts|jose|marquez|paul|bristow|ed|edward|krapels)\b", flags=re.I)
RE_FILE = re.compile(r"\b(pdf|doc|xls|letter|file|attach|enclo|attachment)\b", flags=re.I)
RE_DATE = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december|week|month|year|day|today|tomorrow|yesterday)\b", flags=re.I)

# Listas de palavras por categoria
TAG_WORDS = {
    "ARTDET": ["the", "a", "an"],
    "DEMDET": ["this", "that", "these", "those"],
    "PRONPERS": ["i", "you", "he", "she", "it", "we", "they"],
    "PRONOBJPOS": ["me", "you", "him", "her", "it", "us", "them", "my", "your", "his", "her", "its", "our", "their"],
    "PRONRELINT": ["who", "whom", "whose", "which", "what", "where", "when", "why", "how"],
    "PREP": ["in", "on", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "under", "over", "of"],
    "CONJ": ["and", "or", "but", "nor", "so", "yet", "for", "although", "because", "while", "if", "when"],
    "AUXVERB": ["be", "am", "is", "are", "was", "were", "been", "have", "has", "had", "do", "does", "did"],
    "MODAL": ["can", "could", "will", "would", "shall", "should", "may", "might", "must"],
    "NEG": ["not", "no", "neither"],
    "QUANT": ["some", "any", "many", "much", "few", "several", "all", "most", "none", "every", "each"],
    "ADV": ["very", "just", "now", "then", "still", "already", "soon", "often", "always", "never", "usually", "sometimes", "recently"],
    "FILLER": ["oh", "well", "uh", "um", "hmm", "aye", "hey"],
    "ADJ": ["other", "same", "new", "good", "bad", "different", "little", "big", "great", "small"]
}

def strip_accents(text: str) -> str:
    if not text:
        return text
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(ch for ch in nfkd if not unicodedata.combining(ch))

def normalize_text(text: str,
                   remove_urls=True,
                   remove_emails=True,
                   remove_html=True,
                   remove_numbers=False,
                   min_token_len=1,
                   collapse_repeats=True) -> str:
    if not text:
        return ''
    s = text

    s = RE_SUBJECT.sub('', s)
    if remove_html:
        s = RE_HTML.sub(' ', s)
    if remove_urls:
        s = RE_URL_LOOSE.sub(' URLTAG ', s)
        s = RE_URL.sub(' URLTAG ', s)
    if remove_emails:
        s = RE_EMAIL_LOOSE.sub(' EMAILTAG ', s)
        s = RE_EMAIL.sub(' EMAILTAG ', s)
    s = RE_REPEAT_PUNCT.sub(' PUNCTTAG ', s)
    s = RE_EXACT_RE.sub(' RETAG ', s)
    s = RE_SUBJECT.sub('', s)
    s = RE_NAME.sub("NAMETAG", s) 
    s = RE_FILE.sub("FILETAG", s)
    s = RE_DATE.sub("DATETAG", s)
    s = strip_accents(s)
    s = s.lower()
    if collapse_repeats:
        s = RE_REPEAT_CHARS.sub(r'\1\1', s)
    if remove_numbers:
        s = re.sub(r'\d+', ' ', s)
    s = RE_NON_ALNUM.sub(' ', s)
    s = RE_MULTI_SPACE.sub(' ', s).strip()

    tokens = s.split()
    tagged_tokens = []
    tag_word_set = set(w for words in TAG_WORDS.values() for w in words)
    tags_to_remove = {"PUNCTTAG", "EMAILTAG"}
    tags_to_keep = {"NAMETAG", "URLTAG"}

    for t in tokens:
        if t in tag_word_set:
            continue
        if t in tags_to_remove:
            continue
        if t in tags_to_keep:
            tagged_tokens.append(t)  # mantém a tag sem stemmatizar
        else:
            tagged_tokens.append(stemmer.stem(t))  # aplica stemmatização
    if min_token_len > 1:
        tagged_tokens = [tok for tok in tagged_tokens if len(tok) >= min_token_len]
    return ' '.join(tagged_tokens)

def normalize_and_split_csv(input_path: str, train_path: str, test_path: str,
                           text_field: str = 'text',
                           min_token_len: int = 1,
                           shuffle: bool = True,
                           seed: int = 42,
                           n_train: int = 5155):
    input_p = Path(input_path)
    train_p = Path(train_path)
    test_p = Path(test_path)
    train_p.parent.mkdir(parents=True, exist_ok=True)
    test_p.parent.mkdir(parents=True, exist_ok=True)

    # Lê todas as linhas
    with open(input_p, 'r', encoding='utf-8', newline='') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames or []
        if text_field not in fieldnames:
            raise SystemExit(f"Campo '{text_field}' não encontrado no cabeçalho: {fieldnames}")
        rows = [row for row in reader]

    # Embaralha antes de normalizar
    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(rows)

    # Divide para treino e teste
    train_rows = rows[:n_train]
    test_rows = rows[n_train:]

    # Normaliza e grava treino
    with open(train_p, 'w', encoding='utf-8', newline='') as trainfile:
        writer = csv.DictWriter(trainfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in train_rows:
            raw = row.get(text_field, '') or ''
            row[text_field] = normalize_text(raw, min_token_len=min_token_len, remove_numbers=True)
            writer.writerow(row)

    # Normaliza e grava teste
    with open(test_p, 'w', encoding='utf-8', newline='') as testfile:
        writer = csv.DictWriter(testfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in test_rows:
            raw = row.get(text_field, '') or ''
            row[text_field] = normalize_text(raw, min_token_len=min_token_len, remove_numbers=True)
            writer.writerow(row)

if __name__ == "__main__":
    exemplos = [
        "running runs runner",
        "better best good",
        "studies studying studied",
        "NAMETAG went to the URLTAG",
        "the quick brown fox jumps over the lazy dog"
    ]
    for texto in exemplos:
        print("Original:", texto)
        print("Normalizado:", normalize_text(texto))
        print("---")

    N_TRAIN = 5729-1000  # quantidade para treino

    parser = argparse.ArgumentParser(description="Normaliza o campo 'text' de um CSV e divide em treino/teste")
    parser.add_argument(
        "input_path",
        nargs="?",
        default=r"../SpamDetector/data/raw/emails.csv",
        help="Caminho do CSV de entrada (padrão: data/raw/emails.csv)"
    )
    parser.add_argument(
        "--train_path",
        default=r"../SpamDetector/data/processed/emails_train.csv",
        help="Caminho do CSV de treino (padrão: data/processed/emails_train.csv)"
    )
    parser.add_argument(
        "--test_path",
        default=r"../SpamDetector/data/processed/emails_test.csv",
        help="Caminho do CSV de teste (padrão: data/processed/emails_test.csv)"
    )
    parser.add_argument('--min-token-len', type=int, default=1,
                        help="Remove tokens menores que este tamanho (padrão 1)")
    parser.add_argument('--no-shuffle', action='store_true',
                        help="Desativa embaralhamento (mantém ordem original)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Semente para embaralhamento reprodutível (padrão 42)")
    parser.add_argument('--n-train', type=int, default=N_TRAIN,
                        help="Número de linhas para treino (restante vai para teste)")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    train_path = Path(args.train_path)
    test_path = Path(args.test_path)

    if not input_path.exists():
        print(f"Arquivo de entrada não encontrado: {input_path}")
        raise SystemExit(1)

    normalize_and_split_csv(str(input_path), str(train_path), str(test_path),
                           min_token_len=args.min_token_len,
                           shuffle=not args.no_shuffle,
                           seed=args.seed,
                           n_train=args.n_train)
    print(f"Treino salvo em: {train_path}")
    print(f"Teste salvo em: {test_path}")

    