import unicodedata
import re
import csv
import argparse
import random
from pathlib import Path

# Ajusta limite para campos grandes
csv.field_size_limit(10**7)

# Compila expressões regulares usadas
RE_SUBJECT = re.compile(r'^\s*(subject|re|fw|fwd)[:\-\s]+', flags=re.I)
RE_URL = re.compile(r'https?://\S+|www\.\S+', flags=re.I)
RE_EMAIL = re.compile(r'\S+@\S+')
RE_HTML = re.compile(r'<[^>]+>')
# Padrões mais permissivos para capturar URLs/emails com espaços/pontos estranhos
RE_URL_LOOSE = re.compile(r'(?:https?[:\s]*[/\\]{0,2}|www[.\s])[\w\.\-\/\\\s]{5,}', flags=re.I)
RE_EMAIL_LOOSE = re.compile(r'[\w\.\-]+\s*@\s*[\w\.\-]+(?:\s*\.\s*\w{2,})+', flags=re.I)
RE_NON_ALNUM = re.compile(r"[^a-z0-9'\s]")   # mantém apóstrofo
RE_MULTI_SPACE = re.compile(r'\s+')
RE_REPEAT_CHARS = re.compile(r'(.)\1{2,}')   # colapsa repetições longas
RE_REPEAT_PUNCT = re.compile(r"(([!?\-])(\s*\2){1,})")

RE_SUBJECT = re.compile(r'^\s*(subject|re|fw|fwd)[:\-\s]+', flags=re.I)
RE_EXACT_RE = re.compile(r'\sre\s:', flags=re.I)

# Função: remove acentuação unicode
def strip_accents(text: str) -> str:
    # Normaliza acentos usando NFKD
    if not text:
        return text
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(ch for ch in nfkd if not unicodedata.combining(ch))

# Função: normaliza um texto aplicando várias etapas
def normalize_text(text: str,
                   remove_urls=True,
                   remove_emails=True,
                   remove_html=True,
                   remove_numbers=False,
                   min_token_len=1,
                   collapse_repeats=True) -> str:
    # Lida com entradas vazias
    if not text:
        return ''
    s = text

    # Remove prefixos de assunto no começo
    s = RE_SUBJECT.sub('', s)

    # Remove HTML
    if remove_html:
        s = RE_HTML.sub(' ', s)

    # Substitui formas "soltas" e formas normais de URLs/emails por tags
    # (faz antes da remoção de pontuação para capturar "ramsey @ goldengraphix . com" etc.)
    if remove_urls:
        s = RE_URL_LOOSE.sub(' URLTAG ', s)  # captura www . verticallv . com etc.
        s = RE_URL.sub(' URLTAG ', s)        # captura https://...
    if remove_emails:
        s = RE_EMAIL_LOOSE.sub(' EMAILTAG ', s)  # captura ramsey @ goldengraphix . com
        s = RE_EMAIL.sub(' EMAILTAG ', s)        # captura email@dominio.com

    s = RE_REPEAT_PUNCT.sub(' PUNCTTAG ', s)

    s = RE_EXACT_RE.sub(' RETAG ', s)
    s = RE_SUBJECT.sub('', s)
    
    # Remove acentuação
    s = strip_accents(s)

    # Converte para minúsculas
    s = s.lower()

    # Colapsa repetições longas de caracteres (loooove -> loove)
    if collapse_repeats:
        s = RE_REPEAT_CHARS.sub(r'\1\1', s)

    # Remove números se solicitado
    if remove_numbers:
        s = re.sub(r'\d+', ' ', s)

    # Remove caracteres não alfanuméricos (mantém apóstrofo)
    s = RE_NON_ALNUM.sub(' ', s)

    # Colapsa espaços múltiplos
    s = RE_MULTI_SPACE.sub(' ', s).strip()

    # Remove tokens curtos se necessário
    if min_token_len > 1:
        tokens = [t for t in s.split() if len(t) >= min_token_len]
        s = ' '.join(tokens)

    return s

# Função: processa CSV e grava arquivo normalizado
def normalize_csv(input_path: str, output_path: str,
                  text_field: str = 'text',
                  min_token_len: int = 1,
                  shuffle: bool = True,
                  seed: int = 42):
    input_p = Path(input_path)
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)

    # Lê todas as linhas primeiro (para poder embaralhar mantendo rótulos)
    with open(input_p, 'r', encoding='utf-8', newline='') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames or []
        if text_field not in fieldnames:
            raise SystemExit(f"Campo '{text_field}' não encontrado no cabeçalho: {fieldnames}")
        rows = list(reader)

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

# Bloco principal: argumentos de linha de comando e verificação de caminhos
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normaliza o campo 'text' de um CSV")
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
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    # Verifica existência do arquivo de entrada
    if not input_path.exists():
        print(f"Arquivo de entrada não encontrado: {input_path}")
        print(r'Exemplo: python src/normalize_csv_text.py ../SpamDetector/data/raw/emails.csv ../SpamDetector/data/processed/emails_cleaned.csv')
        raise SystemExit(1)

    # Garante pasta de saída existente
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Executa normalização com embaralhamento por padrão
    normalize_csv(str(input_path), str(output_path),
                  min_token_len=args.min_token_len,
                  shuffle=not args.no_shuffle,
                  seed=args.seed)
    print(f"CSV normalizado salvo em: {output_path}")