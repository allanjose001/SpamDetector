import unicodedata
import re
import csv
import argparse
import random
from pathlib import Path

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

# Função: remove acentuação unicode
def strip_accents(text: str) -> str:
    if not text:
        return text
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(ch for ch in nfkd if not unicodedata.combining(ch))

# Função: normaliza um texto aplicando várias etapas e insere tags
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
    s = strip_accents(s)
    s = s.lower()
    if collapse_repeats:
        s = RE_REPEAT_CHARS.sub(r'\1\1', s)
    if remove_numbers:
        s = re.sub(r'\d+', ' ', s)
    s = RE_NON_ALNUM.sub(' ', s)
    s = RE_MULTI_SPACE.sub(' ', s).strip()

    # Adiciona tags para cada categoria
    tokens = s.split()
    tagged_tokens = []
    for t in tokens:
        tagged = False
        for tag, words in TAG_WORDS.items():
            if t in words:
                tagged_tokens.append(f"{tag}TAG")
                tagged = True
                break
        if not tagged:
            tagged_tokens.append(t)
    # Remove tokens curtos se necessário
    if min_token_len > 1:
        tagged_tokens = [tok for tok in tagged_tokens if len(tok) >= min_token_len]
    return ' '.join(tagged_tokens)

# Função: processa CSV e grava arquivo normalizado
def normalize_csv(input_path: str, output_path: str,
                  text_field: str = 'text',
                  min_token_len: int = 1,
                  shuffle: bool = True,
                  seed: int = 42,
                  max_lines: int = None):
    input_p = Path(input_path)
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)

    # Lê todas as linhas primeiro (para poder embaralhar mantendo rótulos)
    with open(input_p, 'r', encoding='utf-8', newline='') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames or []
        if text_field not in fieldnames:
            raise SystemExit(f"Campo '{text_field}' não encontrado no cabeçalho: {fieldnames}")
        rows = []
        for i, row in enumerate(reader):
            if max_lines is not None and i >= max_lines:
                break
            rows.append(row)

    # Embaralha se solicitado (reprodutível via seed)
    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(rows)

    # Normaliza e grava
    with open(output_p, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in rows:
            raw = row.get(text_field, '') or ''
            row[text_field] = normalize_text(raw, min_token_len=min_token_len, remove_numbers=True)
            writer.writerow(row)

if __name__ == "__main__":
    MAX_EMAILS = 3000  # ajuste para o valor desejado

    parser = argparse.ArgumentParser(description="Normaliza o campo 'text' de um CSV com tags linguísticas")
    parser.add_argument(
        "input_path",
        nargs="?",
        default=r"../SpamDetector/data/raw/emails.csv",
        help="Caminho do CSV de entrada (padrão: data/raw/emails.csv)"
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        default=r"../SpamDetector/data/processed/emails_cleaned.csv",
        help="Caminho do CSV de saída (padrão: data/processed/emails_cleaned.csv)"
    )
    parser.add_argument('--min-token-len', type=int, default=1,
                        help="Remove tokens menores que este tamanho (padrão 1)")
    parser.add_argument('--no-shuffle', action='store_true',
                        help="Desativa embaralhamento (mantém ordem original)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Semente para embaralhamento reprodutível (padrão 42)")
    # parser.add_argument('--max-lines', type=int, default=None,
    #                     help="Limita o número de linhas a serem normalizadas")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        print(f"Arquivo de entrada não encontrado: {input_path}")
        print(r'Exemplo: python src/normalize_csv_text.py ../SpamDetector/data/raw/emails.csv ../SpamDetector/data/processed/emails_cleaned.csv')
        raise SystemExit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    normalize_csv(str(input_path), str(output_path),
                  min_token_len=args.min_token_len,
                  shuffle=not args.no_shuffle,
                  seed=args.seed,
                  max_lines=MAX_EMAILS)
    print(f"CSV normalizado salvo em: {output_path}")