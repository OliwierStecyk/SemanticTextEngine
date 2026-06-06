import os
import argparse
from collections import Counter
import re
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

try:
    import nltk
    nltk.data.find('tokenizers/punkt')
except Exception:
    nltk = None

try:
    import spacy
    _spacy_pl = None
    try:
        _spacy_pl = spacy.load('pl_core_news_sm')
    except Exception:
        _spacy_pl = None
except Exception:
    spacy = None
    _spacy_pl = None

try:
    import textstat
except Exception:
    textstat = None

try:
    import gensim
    from gensim import corpora
    from gensim.models import LdaModel
except Exception:
    gensim = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None

try:
    from sentence_transformers import SentenceTransformer
    _sbert = SentenceTransformer('all-MiniLM-L6-v2')
except Exception:
    _sbert = None

try:
    from scipy.spatial.distance import jensenshannon
    import numpy as np
except Exception:
    jensenshannon = None
    import numpy as np

try:
    import pyphen
except Exception:
    pyphen = None

try:
    import syllapy
except Exception:
    syllapy = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
except Exception:
    _vader = None

try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None

RE_NON_WORD = re.compile(r"[^\w\sąćęłńóśżźĄĆĘŁŃÓŚŻŹ-]")
RE_MULTISPACE = re.compile(r"\s+")


def read_author_files(paths: List[str]) -> List[str]:
    texts = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for fn in files:
                    if fn.lower().endswith('.txt'):
                        with open(os.path.join(root, fn), 'r', encoding='utf-8') as f:
                            texts.append(f.read())
        elif os.path.isfile(p):
            with open(p, 'r', encoding='utf-8') as f:
                texts.append(f.read())
    return texts


def clean_text(text: str) -> str:
    if not text:
        return ''
    text = RE_NON_WORD.sub(' ', text)
    text = RE_MULTISPACE.sub(' ', text).strip()
    return text.lower()


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    if nltk:
        try:
            return nltk.word_tokenize(text, language='polish')
        except Exception:
            pass
    return text.split()


def aggregate_stats(texts: List[str]) -> dict:
    all_text = '\n'.join(texts)
    cleaned = clean_text(all_text)
    tokens = tokenize(cleaned)
    total_words = len(tokens)
    unique_words = len(set(tokens))
    avg_word_len = sum(len(t) for t in tokens) / total_words if total_words else 0
    char_count = len(cleaned)
    vowels = sum(1 for ch in cleaned if ch in 'aeiouyąęółóżćńś')
    consonants = sum(1 for ch in cleaned if ch.isalpha() and ch not in 'aeiouyąęółóżćńś')
    top_words = Counter(tokens).most_common(30)
    readability = get_readability(all_text)
    # sentence-level stats
    sentences = []
    if nltk:
        try:
            sentences = nltk.sent_tokenize(all_text, language='polish')
        except Exception:
            sentences = [s for s in re.split(r'[\.!?]+', all_text) if s.strip()]
    else:
        sentences = [s for s in re.split(r'[\.!?]+', all_text) if s.strip()]
    sent_lens = [len(tokenize(clean_text(s))) for s in sentences] if sentences else []
    avg_sent_len = sum(sent_lens) / len(sent_lens) if sent_lens else 0

    # lexical richness
    type_token_ratio = unique_words / total_words if total_words else 0
    hapax = sum(1 for w, c in Counter(tokens).items() if c == 1)

    # syllable & rhythm approximation: count vowel groups per token
    def syllables_in_word(w: str) -> int:
        groups = re.findall(r'[aeiouyąęółóżćńś]+', w)
        return max(1, len(groups)) if w else 0
    syllables = sum(syllables_in_word(t) for t in tokens)
    avg_syll_per_word = syllables / total_words if total_words else 0

    # punctuation counts
    punct_counts = Counter(ch for ch in all_text if ch in ",.;:!?\"'")

    # POS distribution if spaCy available
    pos_counts = {}
    if _spacy_pl:
        try:
            doc = _spacy_pl(all_text)
            pos_counts = Counter([tok.pos_ for tok in doc])
        except Exception:
            pos_counts = {}
    return {
        'total_words': total_words,
        'unique_words': unique_words,
        'avg_word_len': avg_word_len,
        'char_count': char_count,
        'vowels': vowels,
        'consonants': consonants,
        'sentences': len(sentences),
        'avg_sent_len': avg_sent_len,
        'type_token_ratio': type_token_ratio,
        'hapax': hapax,
        'syllables': syllables,
        'avg_syll_per_word': avg_syll_per_word,
        'punct_counts': dict(punct_counts),
        'pos_counts': dict(pos_counts),
        'top_words': top_words,
        'readability': readability,
        'raw_text': all_text,
    }


def find_ngrams(texts: List[str], n: int = 3, top_k: int = 30):
    tokens = []
    for t in texts:
        tokens.extend(tokenize(clean_text(t)))
    ngrams = Counter()
    for i in range(len(tokens) - n + 1):
        ngram = ' '.join(tokens[i:i+n])
        ngrams[ngram] += 1
    return ngrams.most_common(top_k)


def detect_chorus(texts: List[str], min_repeats: int = 2):
    # Find exact repeated lines across all texts
    lines = []
    for t in texts:
        for l in t.splitlines():
            l2 = clean_text(l)
            if l2:
                lines.append(l2)
    c = Counter(lines)
    chorus_lines = [(ln, ct) for ln, ct in c.items() if ct >= min_repeats]
    chorus_lines.sort(key=lambda x: -x[1])
    return chorus_lines


def export_csv(path: str, rows: List[dict], fieldnames: List[str]):
    import csv
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def compute_tfidf_similarity(texts_a: List[str], texts_b: List[str]):
    if not TfidfVectorizer:
        return None
    docs = ['\n'.join(texts_a), '\n'.join(texts_b)]
    vec = TfidfVectorizer(stop_words='english')
    X = vec.fit_transform(docs)
    sim = cosine_similarity(X[0:1], X[1:2])[0][0]
    # get top tf-idf features per doc
    feature_names = vec.get_feature_names_out()
    top_a = sorted(zip(feature_names, X[0].toarray()[0]), key=lambda x: -x[1])[:30]
    top_b = sorted(zip(feature_names, X[1].toarray()[0]), key=lambda x: -x[1])[:30]
    return {'cosine': float(sim), 'top_a': top_a, 'top_b': top_b}


def compute_pairwise_tfidf_matrix(paths: List[str]):
    """Compute TF-IDF matrix for each file under given paths (flattened). Returns (files, matrix)."""
    docs = []
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, fns in os.walk(p):
                for fn in fns:
                    if fn.lower().endswith('.txt'):
                        fp = os.path.join(root, fn)
                        with open(fp, 'r', encoding='utf-8') as f:
                            docs.append(f.read())
                        files.append(os.path.relpath(fp))
        elif os.path.isfile(p):
            with open(p, 'r', encoding='utf-8') as f:
                docs.append(f.read())
            files.append(os.path.relpath(p))
    if not docs or not TfidfVectorizer:
        return files, None
    vec = TfidfVectorizer(stop_words='english')
    X = vec.fit_transform(docs)
    mat = cosine_similarity(X)
    return files, mat


def compute_embeddings(texts: List[str]):
    if not _sbert:
        return None
    docs = ['\n'.join(texts)]
    try:
        emb = _sbert.encode(docs, convert_to_numpy=True)
        return emb[0]
    except Exception:
        return None


def compute_per_file_embeddings(paths: List[str]):
    files = []
    docs = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, fns in os.walk(p):
                for fn in fns:
                    if fn.lower().endswith('.txt'):
                        fp = os.path.join(root, fn)
                        with open(fp, 'r', encoding='utf-8') as f:
                            docs.append(f.read())
                        files.append(os.path.relpath(fp))
        elif os.path.isfile(p):
            with open(p, 'r', encoding='utf-8') as f:
                docs.append(f.read())
            files.append(os.path.relpath(p))
    if not docs or not _sbert:
        return files, None
    try:
        embs = _sbert.encode(docs, convert_to_numpy=True)
        return files, embs
    except Exception:
        return files, None


def jensen_shannon_divergence(freq_a: Counter, freq_b: Counter):
    # Build probability vectors over union of keys
    keys = sorted(set(list(freq_a.keys()) + list(freq_b.keys())))
    pa = np.array([freq_a.get(k, 0) for k in keys], dtype=float)
    pb = np.array([freq_b.get(k, 0) for k in keys], dtype=float)
    if pa.sum() == 0 or pb.sum() == 0:
        return None
    pa = pa / pa.sum()
    pb = pb / pb.sum()
    if jensenshannon:
        return float(jensenshannon(pa, pb))
    # fallback: compute sqrt(JS) manually using scipy not available
    m = 0.5 * (pa + pb)
    def kl(p, q):
        nz = p > 0
        return np.sum(p[nz] * np.log(p[nz] / q[nz]))
    return float(0.5 * (kl(pa, m) + kl(pb, m)))


def repetition_score(tokens: List[str], coverage: float = 0.5):
    if not tokens:
        return 0.0
    c = Counter(tokens)
    total = len(tokens)
    cum = 0
    for i, (_, cnt) in enumerate(c.most_common()):
        cum += cnt
        if cum / total >= coverage:
            return (i+1) / len(c)
    return 1.0


def syllable_stats(tokens: List[str]):
    if syllapy:
        syl_counts = [syllapy.count(t) for t in tokens]
    elif pyphen:
        dic = pyphen.Pyphen(lang='pl')
        syl_counts = [max(1, dic.inserted(t).count('-')+1) for t in tokens]
    else:
        # fallback: vowel groups heuristic
        syl_counts = [max(1, len(re.findall(r'[aeiouyąęółóżćńś]+', t))) for t in tokens]
    total = sum(syl_counts)
    avg = total / len(syl_counts) if syl_counts else 0
    return {'total_syllables': total, 'avg_syllables_per_word': avg}


def syllables_in_word(w: str) -> int:
    if not w:
        return 0
    if syllapy:
        try:
            return max(1, syllapy.count(w))
        except Exception:
            pass
    if pyphen:
        try:
            dic = pyphen.Pyphen(lang='pl')
            return max(1, dic.inserted(w).count('-')+1)
        except Exception:
            pass
    groups = re.findall(r'[aeiouyąęółóżćńś]+', w)
    return max(1, len(groups))


def rhyme_scheme(lines: List[str], suffix_len: int = 3):
    # Simplified rhyme detection based on line endings
    endings = [clean_text(l).split()[-1] if clean_text(l).split() else '' for l in lines]
    endings = [e[-suffix_len:] if len(e) >= suffix_len else e for e in endings]
    scheme = []
    mapping = {}
    next_label = ord('A')
    for e in endings:
        if e in mapping:
            scheme.append(mapping[e])
        else:
            mapping[e] = chr(next_label)
            scheme.append(mapping[e])
            next_label += 1
    return ''.join(scheme)


def rhythm_metrics_for_text(text: str):
    # syllables per line, variance, avg
    lines = [l for l in text.splitlines() if l.strip()]
    syls = []
    for l in lines:
        toks = tokenize(clean_text(l))
        s = sum(syllables_in_word(t) for t in toks)
        syls.append(s)
    import numpy as np
    if not syls:
        return {'lines':0,'avg_syl_per_line':0,'var_syl_per_line':0,'median_syl_per_line':0}
    return {'lines': len(syls), 'avg_syl_per_line': float(np.mean(syls)), 'var_syl_per_line': float(np.var(syls)), 'median_syl_per_line': float(np.median(syls))}


def rhyme_density(text: str, suffix_len: int = 3):
    # proportion of lines that share an ending with at least one other line
    lines = [clean_text(l) for l in text.splitlines() if l.strip()]
    endings = [ (l.split()[-1][-suffix_len:] if l.split() else '') for l in lines]
    c = Counter(endings)
    shared = sum(1 for e in endings if c.get(e,0) > 1)
    return shared / len(lines) if lines else 0


def plot_rhythm_and_rhyme(per_a, per_b, texts_a, texts_b, out_prefix):
    try:
        import pandas as pd
        rows = []
        for path_row, text in zip(per_a, texts_a):
            rm = rhythm_metrics_for_text(text)
            rd = rhyme_density(text)
            rows.append({'file': path_row.get('file',''), 'avg_syl_per_line': rm['avg_syl_per_line'], 'var_syl_per_line': rm['var_syl_per_line'], 'rhyme_density': rd})
        for path_row, text in zip(per_b, texts_b):
            rm = rhythm_metrics_for_text(text)
            rd = rhyme_density(text)
            rows.append({'file': path_row.get('file',''), 'avg_syl_per_line': rm['avg_syl_per_line'], 'var_syl_per_line': rm['var_syl_per_line'], 'rhyme_density': rd})
        df = pd.DataFrame(rows)
        if not df.empty:
            plt.figure(figsize=(10,6))
            sns.scatterplot(x='var_syl_per_line', y='rhyme_density', data=df)
            plt.title('Rhythm variance vs Rhyme density per file')
            plt.tight_layout(); plt.savefig(f'{out_prefix}_rhythm_rhyme_scatter.png'); plt.close()
    except Exception:
        pass


def sentiment_score(texts: List[str]):
    all_text = '\n'.join(texts)
    if _vader:
        return _vader.polarity_scores(all_text)
    return {}


def per_file_stats(paths: List[str]):
    rows = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for fn in files:
                    if fn.lower().endswith('.txt'):
                        fp = os.path.join(root, fn)
                        with open(fp, 'r', encoding='utf-8') as f:
                            txt = f.read()
                        stats = aggregate_stats([txt])
                        rows.append({'file': os.path.relpath(fp), 'words': stats['total_words'], 'unique': stats['unique_words'], 'avg_word_len': stats['avg_word_len'], 'avg_sent_len': stats.get('avg_sent_len', ''), 'ttr': stats.get('type_token_ratio', '')})
        elif os.path.isfile(p):
            with open(p, 'r', encoding='utf-8') as f:
                txt = f.read()
            stats = aggregate_stats([txt])
            rows.append({'file': os.path.relpath(p), 'words': stats['total_words'], 'unique': stats['unique_words'], 'avg_word_len': stats['avg_word_len'], 'avg_sent_len': stats.get('avg_sent_len', ''), 'ttr': stats.get('type_token_ratio', '')})
    return rows


def get_readability(text: str) -> dict:
    if not text or not textstat:
        return {}
    try:
        return {
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'smog_index': textstat.smog_index(text),
        }
    except Exception:
        return {}


def plot_comparison(stats_a: dict, stats_b: dict, author_a: str, author_b: str, out_prefix: str = 'compare'):
    sns.set(style='whitegrid')
    metrics = ['total_words', 'unique_words', 'avg_word_len', 'char_count', 'vowels', 'consonants']
    labels = ['Total words', 'Unique words', 'Avg word len', 'Chars', 'Vowels', 'Consonants']
    a_vals = [stats_a[m] for m in metrics]
    b_vals = [stats_b[m] for m in metrics]

    x = range(len(metrics))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar([i - width/2 for i in x], a_vals, width=width, label=author_a)
    plt.bar([i + width/2 for i in x], b_vals, width=width, label=author_b)
    plt.xticks(x, labels, rotation=20)
    plt.legend()
    plt.tight_layout()
    # ensure output directory exists
    out_dir = os.path.dirname(out_prefix)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f'{out_prefix}_metrics.png')
    plt.close()

    # Top words comparison (plot top 10 each)
    top_a = dict(stats_a['top_words'][:10])
    top_b = dict(stats_b['top_words'][:10])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(x=list(top_a.values()), y=list(top_a.keys()), ax=axes[0])
    axes[0].set_title(f'Top words: {author_a}')
    sns.barplot(x=list(top_b.values()), y=list(top_b.keys()), ax=axes[1])
    axes[1].set_title(f'Top words: {author_b}')
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_top_words.png')
    plt.close()


def generate_wordcloud(text: str, out_path: str):
    if not WordCloud:
        return
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    wc.to_file(out_path)


def plot_additional(stats_a, stats_b, texts_a, texts_b, per_a, per_b, tfidf, ngrams_a, ngrams_b, out_prefix):
    out_dir = os.path.dirname(out_prefix)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # per-file scatter: words vs unique
    try:
        import pandas as pd
        df_a = pd.DataFrame(per_a)
        df_b = pd.DataFrame(per_b)
        plt.figure(figsize=(8,6))
        plt.scatter(df_a['words'], df_a['unique'], label='A', alpha=0.7)
        plt.scatter(df_b['words'], df_b['unique'], label='B', alpha=0.7)
        plt.xlabel('Words'); plt.ylabel('Unique words'); plt.legend()
        plt.title('Per-file: words vs unique')
        plt.tight_layout(); plt.savefig(f'{out_prefix}_perfile_scatter.png'); plt.close()
    except Exception:
        pass

    # word length histogram
    try:
        tokens_a = tokenize(clean_text('\n'.join(texts_a)))
        tokens_b = tokenize(clean_text('\n'.join(texts_b)))
        lens_a = [len(t) for t in tokens_a]
        lens_b = [len(t) for t in tokens_b]
        plt.figure(figsize=(10,6))
        sns.histplot(lens_a, color='C0', label='A', stat='density', kde=True)
        sns.histplot(lens_b, color='C1', label='B', stat='density', kde=True)
        plt.legend(); plt.title('Word length distribution'); plt.xlabel('Chars'); plt.savefig(f'{out_prefix}_wordlen_hist.png'); plt.close()
    except Exception:
        pass

    # sentence length histogram
    try:
        if nltk:
            sents_a = nltk.sent_tokenize('\n'.join(texts_a), language='polish')
            sents_b = nltk.sent_tokenize('\n'.join(texts_b), language='polish')
        else:
            sents_a = [s for s in re.split(r'[\.!?]+', '\n'.join(texts_a)) if s.strip()]
            sents_b = [s for s in re.split(r'[\.!?]+', '\n'.join(texts_b)) if s.strip()]
        slens_a = [len(tokenize(clean_text(s))) for s in sents_a]
        slens_b = [len(tokenize(clean_text(s))) for s in sents_b]
        plt.figure(figsize=(10,6))
        sns.kdeplot(slens_a, label='A'); sns.kdeplot(slens_b, label='B')
        plt.legend(); plt.title('Sentence length (words) distribution'); plt.savefig(f'{out_prefix}_sentlen_kde.png'); plt.close()
    except Exception:
        pass

    # syllable distribution
    try:
        syls_a = [syllables_in_word(t) for t in tokenize(clean_text('\n'.join(texts_a)))]
        syls_b = [syllables_in_word(t) for t in tokenize(clean_text('\n'.join(texts_b)))]
        plt.figure(figsize=(10,6))
        sns.kdeplot(syls_a, label='A'); sns.kdeplot(syls_b, label='B')
        plt.legend(); plt.title('Syllables per word distribution'); plt.savefig(f'{out_prefix}_syl_kde.png'); plt.close()
    except Exception:
        pass

    # top n-grams bar (10)
    try:
        ng_a = ngrams_a[:10]
        ng_b = ngrams_b[:10]
        fig, axes = plt.subplots(1,2,figsize=(12,6))
        sns.barplot(x=[c for _,c in ng_a], y=[n for n,_ in ng_a], ax=axes[0])
        axes[0].set_title('Top trigrams A')
        sns.barplot(x=[c for _,c in ng_b], y=[n for n,_ in ng_b], ax=axes[1])
        axes[1].set_title('Top trigrams B')
        plt.tight_layout(); plt.savefig(f'{out_prefix}_trigrams.png'); plt.close()
    except Exception:
        pass

    # TF-IDF top features
    try:
        if tfidf and 'top_a' in tfidf:
            fa = tfidf['top_a'][:15]
            fb = tfidf['top_b'][:15]
            fig, axes = plt.subplots(1,2,figsize=(12,6))
            sns.barplot(x=[v for _,v in fa], y=[w for w,_ in fa], ax=axes[0])
            axes[0].set_title('TF-IDF top A')
            sns.barplot(x=[v for _,v in fb], y=[w for w,_ in fb], ax=axes[1])
            axes[1].set_title('TF-IDF top B')
            plt.tight_layout(); plt.savefig(f'{out_prefix}_tfidf_top.png'); plt.close()
    except Exception:
        pass

    # cumulative distribution of word lengths
    try:
        import numpy as np
        all_lens = sorted(lens_a + lens_b)
        vals, bins = np.histogram(all_lens, bins=30, density=True)
        cdf = np.cumsum(vals) / np.sum(vals)
        plt.figure(figsize=(8,5))
        plt.plot(bins[1:], cdf, label='combined')
        plt.title('Cumulative distribution of word lengths')
        plt.xlabel('word length'); plt.ylabel('CDF')
        plt.savefig(f'{out_prefix}_wordlen_cdf.png'); plt.close()
    except Exception:
        pass

    # readability comparison
    try:
        r_a = stats_a.get('readability', {})
        r_b = stats_b.get('readability', {})
        if r_a or r_b:
            keys = sorted(set(list(r_a.keys()) + list(r_b.keys())))
            a_vals = [r_a.get(k, 0) for k in keys]
            b_vals = [r_b.get(k, 0) for k in keys]
            x = range(len(keys))
            width = 0.35
            plt.figure(figsize=(8,5))
            plt.bar([i-width/2 for i in x], a_vals, width=width, label='A')
            plt.bar([i+width/2 for i in x], b_vals, width=width, label='B')
            plt.xticks(x, keys, rotation=20); plt.legend()
            plt.title('Readability metrics comparison')
            plt.tight_layout(); plt.savefig(f'{out_prefix}_readability.png'); plt.close()
    except Exception:
        pass

    # POS distribution comparison (if available)
    try:
        pa = stats_a.get('pos_counts', {})
        pb = stats_b.get('pos_counts', {})
        if pa or pb:
            keys = sorted(set(list(pa.keys()) + list(pb.keys())))
            a_vals = [pa.get(k, 0) for k in keys]
            b_vals = [pb.get(k, 0) for k in keys]
            fig, axes = plt.subplots(1,2,figsize=(12,6))
            sns.barplot(x=a_vals, y=keys, ax=axes[0]); axes[0].set_title('POS A')
            sns.barplot(x=b_vals, y=keys, ax=axes[1]); axes[1].set_title('POS B')
            plt.tight_layout(); plt.savefig(f'{out_prefix}_pos_dist.png'); plt.close()
    except Exception:
        pass

    # combined JSON report
    try:
        import json
        report = {
            'stats_a': stats_a,
            'stats_b': stats_b,
            'tfidf': tfidf,
            'top_ngrams_a': ngrams_a[:20],
            'top_ngrams_b': ngrams_b[:20],
        }
        out_dir = os.path.dirname(out_prefix)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(f'{out_prefix}_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def plot_zipf(tokens: List[str], out_path: str):
    if not tokens:
        return
    freq = Counter(tokens)
    freqs = sorted(freq.values(), reverse=True)
    ranks = range(1, len(freqs)+1)
    plt.figure(figsize=(8,5))
    plt.loglog(ranks, freqs, marker='.')
    plt.title('Zipf plot (rank vs frequency)')
    plt.xlabel('Rank (log)'); plt.ylabel('Frequency (log)')
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def jaccard_matrix(tokens_a: List[str], tokens_b: List[str], out_path: str):
    sa = set(tokens_a)
    sb = set(tokens_b)
    inter = len(sa & sb)
    union = len(sa | sb)
    j = inter / union if union else 0
    # simple bar showing Jaccard
    plt.figure(figsize=(4,3))
    plt.bar(['Jaccard'], [j])
    plt.ylim(0,1)
    plt.title('Jaccard overlap between authors')
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def rolling_ttr_plot(per_rows: List[dict], out_path: str, window: int = 3):
    try:
        import pandas as pd
        df = pd.DataFrame(per_rows)
        if df.empty:
            return
        df = df.sort_values('file')
        df['ttr'] = df['ttr'] if 'ttr' in df.columns else (df['unique'] / df['words']).fillna(0)
        df['ttr_roll'] = df['ttr'].rolling(window=min(window, len(df)), center=False).mean()
        plt.figure(figsize=(10,4))
        plt.plot(df['file'], df['ttr'], marker='o', label='TTR')
        plt.plot(df['file'], df['ttr_roll'], marker='o', label=f'Rolling TTR (window={window})')
        plt.xticks(rotation=60)
        plt.legend(); plt.title('TTR per file with rolling average')
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        plt.tight_layout(); plt.savefig(out_path); plt.close()
    except Exception:
        pass


def find_author_paths(root_data: str, language: str, author_name: str) -> List[str]:
    # Search for directories matching author_name under language folder
    base = os.path.join(root_data, language)
    matches = []
    if not os.path.isdir(base):
        return matches
    for root, dirs, files in os.walk(base):
        for d in dirs:
            if author_name.lower() in d.lower():
                matches.append(os.path.join(root, d))
    return matches


def build_topic_model(texts: List[str], num_topics: int = 6):
    if not gensim:
        return None
    docs = [tokenize(clean_text(t)) for t in texts]
    dictionary = corpora.Dictionary(docs)
    corpus = [dictionary.doc2bow(d) for d in docs]
    if not corpus:
        return None
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42, passes=5)
    topics = lda.print_topics(num_words=8)
    return {'model': lda, 'dictionary': dictionary, 'corpus': corpus, 'topics': topics}


def train_classifier(paths_a: List[str], paths_b: List[str], out_prefix: str):
    # Build dataset per-file and train simple logistic regression on TF-IDF
    docs = []
    labels = []
    files = []
    for p in paths_a:
        if os.path.isdir(p):
            for root, _, fns in os.walk(p):
                for fn in fns:
                    if fn.lower().endswith('.txt'):
                        fp = os.path.join(root, fn)
                        with open(fp, 'r', encoding='utf-8') as f:
                            docs.append(f.read())
                        labels.append(0)
                        files.append(os.path.relpath(fp))
        elif os.path.isfile(p):
            with open(p, 'r', encoding='utf-8') as f:
                docs.append(f.read())
            labels.append(0)
            files.append(os.path.relpath(p))
    for p in paths_b:
        if os.path.isdir(p):
            for root, _, fns in os.walk(p):
                for fn in fns:
                    if fn.lower().endswith('.txt'):
                        fp = os.path.join(root, fn)
                        with open(fp, 'r', encoding='utf-8') as f:
                            docs.append(f.read())
                        labels.append(1)
                        files.append(os.path.relpath(fp))
        elif os.path.isfile(p):
            with open(p, 'r', encoding='utf-8') as f:
                docs.append(f.read())
            labels.append(1)
            files.append(os.path.relpath(p))
    if not docs or not TfidfVectorizer:
        return None
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    vec = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vec.fit_transform(docs)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42, stratify=labels if len(set(labels))>1 else None)
    clf = LogisticRegression(max_iter=1000)
    try:
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        report = classification_report(y_test, preds, output_dict=True)
        # save report
        import json
        with open(f'{out_prefix}_classifier_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return {'model': clf, 'vectorizer': vec, 'report': report}
    except Exception:
        return None


def build_fingerprint(stats: dict):
    # select numeric features and normalize simply
    keys = ['total_words','unique_words','avg_word_len','avg_sent_len','type_token_ratio','avg_syll_per_word']
    vec = [float(stats.get(k,0)) for k in keys]
    # simple scaling
    smax = max(vec) if vec else 1.0
    if smax == 0:
        smax = 1.0
    norm = [v / smax for v in vec]
    return dict(zip(keys, norm))


def plot_radar(fingerprint_a: dict, fingerprint_b: dict, labels: List[str], out_path: str):
    try:
        import numpy as np
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        vals_a = [fingerprint_a.get(l,0) for l in labels]
        vals_b = [fingerprint_b.get(l,0) for l in labels]
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        vals_a += vals_a[:1]
        vals_b += vals_b[:1]
        angles += angles[:1]
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, vals_a, label='A')
        ax.fill(angles, vals_a, alpha=0.25)
        ax.plot(angles, vals_b, label='B')
        ax.fill(angles, vals_b, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_ylim(0,1)
        plt.legend()
        plt.tight_layout(); plt.savefig(out_path); plt.close()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description='Compare two authors (directories or files)')
    parser.add_argument('author_a', help='path to author A (dir/file) or author name')
    parser.add_argument('author_b', help='path to author B (dir/file) or author name')
    parser.add_argument('--data-root', default='data', help='root data folder')
    parser.add_argument('--language', default='PL', help='language folder to search when names provided')
    parser.add_argument('--out-prefix', default='compare', help='output prefix for plots')
    parser.add_argument('--sample-files', type=int, default=5, help='number of files per author to compare (max 5)')
    args = parser.parse_args()

    def resolve(arg):
        if os.path.exists(arg):
            return [arg]
        # try searching in data-root/language
        found = find_author_paths(args.data_root, args.language, arg)
        if found:
            return found
        # fallback: treat as relative path
        return [arg]

    paths_a = resolve(args.author_a)
    paths_b = resolve(args.author_b)

    texts_a = read_author_files(paths_a)
    texts_b = read_author_files(paths_b)

    stats_a = aggregate_stats(texts_a)
    stats_b = aggregate_stats(texts_b)

    # additional analyses
    ngrams_a = find_ngrams(texts_a, n=3, top_k=30)
    ngrams_b = find_ngrams(texts_b, n=3, top_k=30)
    chorus_a = detect_chorus(texts_a, min_repeats=2)
    chorus_b = detect_chorus(texts_b, min_repeats=2)
    tfidf = compute_tfidf_similarity(texts_a, texts_b)
    sent_a = sentiment_score(texts_a)
    sent_b = sentiment_score(texts_b)

    # export CSVs
    export_csv(f"{args.out_prefix}_metrics.csv", [
        {'author': args.author_a, **{k: stats_a.get(k, '') for k in ['total_words','unique_words','avg_word_len','avg_sent_len','type_token_ratio','hapax','avg_syll_per_word']}},
        {'author': args.author_b, **{k: stats_b.get(k, '') for k in ['total_words','unique_words','avg_word_len','avg_sent_len','type_token_ratio','hapax','avg_syll_per_word']}}
    ], fieldnames=['author','total_words','unique_words','avg_word_len','avg_sent_len','type_token_ratio','hapax','avg_syll_per_word'])

    # export n-grams
    export_csv(f"{args.out_prefix}_ngrams_a.csv", [{'ngram':n,'count':c} for n,c in ngrams_a], fieldnames=['ngram','count'])
    export_csv(f"{args.out_prefix}_ngrams_b.csv", [{'ngram':n,'count':c} for n,c in ngrams_b], fieldnames=['ngram','count'])

    # per-file stats CSV
    per_a = per_file_stats(paths_a)
    per_b = per_file_stats(paths_b)
    export_csv(f"{args.out_prefix}_per_file_a.csv", per_a, fieldnames=['file','words','unique','avg_word_len','avg_sent_len','ttr'])
    export_csv(f"{args.out_prefix}_per_file_b.csv", per_b, fieldnames=['file','words','unique','avg_word_len','avg_sent_len','ttr'])


def top_ngrams_for_file(fp: str, n: int = 2, top_k: int = 20):
    try:
        with open(fp, 'r', encoding='utf-8') as f:
            txt = f.read()
    except Exception:
        return []
    toks = tokenize(clean_text(txt))
    c = Counter()
    for i in range(len(toks)-n+1):
        c[' '.join(toks[i:i+n])] += 1
    return c.most_common(top_k)


def plot_multiple_ngrams_for_author(paths: List[str], sample_n: int, out_prefix: str, author_label: str):
    # gather files
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, fns in os.walk(p):
                for fn in fns:
                    if fn.lower().endswith('.txt'):
                        files.append(os.path.join(root, fn))
        elif os.path.isfile(p):
            files.append(p)
    files = sorted(files)[:max(0, min(sample_n, len(files)))]
    # cap at 5
    files = files[:5]
    idx = 0
    for fp in files:
        base = os.path.splitext(os.path.basename(fp))[0]
        # bigrams
        bg = top_ngrams_for_file(fp, n=2, top_k=15)
        if bg:
            labels = [t for t,_ in bg]
            vals = [c for _,c in bg]
            plt.figure(figsize=(8,5))
            sns.barplot(x=vals, y=labels)
            plt.title(f'{author_label} - {base} top bigrams')
            plt.tight_layout(); plt.savefig(f'{out_prefix}_{author_label}_file{idx}_bigrams.png'); plt.close()
        # trigrams
        tg = top_ngrams_for_file(fp, n=3, top_k=15)
        if tg:
            labels = [t for t,_ in tg]
            vals = [c for _,c in tg]
            plt.figure(figsize=(8,5))
            sns.barplot(x=vals, y=labels)
            plt.title(f'{author_label} - {base} top trigrams')
            plt.tight_layout(); plt.savefig(f'{out_prefix}_{author_label}_file{idx}_trigrams.png'); plt.close()
        idx += 1


def plot_combined_ngrams(paths_a: List[str], paths_b: List[str], sample_n: int, out_prefix: str):
    # select files
    def gather(paths):
        files = []
        for p in paths:
            if os.path.isdir(p):
                for root, _, fns in os.walk(p):
                    for fn in fns:
                        if fn.lower().endswith('.txt'):
                            files.append(os.path.join(root, fn))
            elif os.path.isfile(p):
                files.append(p)
        return sorted(files)[:sample_n][:5]
    fa = gather(paths_a)
    fb = gather(paths_b)
    # pad to length 5
    while len(fa) < 5:
        fa.append(None)
    while len(fb) < 5:
        fb.append(None)
    # create figure with 5 rows, 2 cols per author group? We'll do 5 rows x 2 cols (left=bigram right=trigram) for A, then separate for B in separate figure
    plt.figure(figsize=(16, 20))
    row = 0
    for i in range(5):
        for col, n in enumerate([2,3]):
            plt.subplot(5,2,row*2 + col + 1)
            fp = fa[i]
            if not fp:
                plt.axis('off')
                continue
            ng = top_ngrams_for_file(fp, n=n, top_k=12)
            labels = [t for t,_ in ng]
            vals = [c for _,c in ng]
            sns.barplot(x=vals, y=labels, palette='tab20')
            plt.title(f'A {os.path.basename(fp)} - {n}-grams')
        row += 1
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_combined_A_5files_ngrams.png')
    plt.close()


def plot_single_figure_both_authors(paths_a: List[str], paths_b: List[str], sample_n: int, out_prefix: str):
    """Create a single figure with 5 rows and 4 columns: for each of up to 5 files,
    show A-bigrams, A-trigrams, B-bigrams, B-trigrams side-by-side. Include file base name and word count in titles."""
    def gather(paths):
        files = []
        for p in paths:
            if os.path.isdir(p):
                for root, _, fns in os.walk(p):
                    for fn in fns:
                        if fn.lower().endswith('.txt'):
                            files.append(os.path.join(root, fn))
            elif os.path.isfile(p):
                files.append(p)
        return sorted(files)[:sample_n][:5]

    fa = gather(paths_a)
    fb = gather(paths_b)
    # pad
    while len(fa) < 5:
        fa.append(None)
    while len(fb) < 5:
        fb.append(None)

    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(20, 22))
    for i in range(5):
        for j, (author_files, label) in enumerate(((fa, 'A'), (fb, 'B'))):
            fp = author_files[i]
            if not fp:
                # turn off both subplots for missing file
                axes[i, j*2].axis('off')
                axes[i, j*2+1].axis('off')
                continue
            base = os.path.splitext(os.path.basename(fp))[0]
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    txt = f.read()
            except Exception:
                txt = ''
            tokens = tokenize(clean_text(txt))
            wc = len(tokens)
            # bigrams
            bg = top_ngrams_for_file(fp, n=2, top_k=10)
            if bg:
                labels = [t for t,_ in bg]
                vals = [c for _,c in bg]
                sns.barplot(x=vals, y=labels, ax=axes[i, j*2], palette='tab10')
                axes[i, j*2].set_title(f'{label} {base} (words:{wc})\nBigrams')
            else:
                axes[i, j*2].axis('off')
            # trigrams
            tg = top_ngrams_for_file(fp, n=3, top_k=10)
            if tg:
                labels = [t for t,_ in tg]
                vals = [c for _,c in tg]
                sns.barplot(x=vals, y=labels, ax=axes[i, j*2+1], palette='tab20')
                axes[i, j*2+1].set_title(f'{label} {base} (words:{wc})\nTrigrams')
            else:
                axes[i, j*2+1].axis('off')

    plt.tight_layout()
    out_dir = os.path.dirname(out_prefix)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    outp = f'{out_prefix}_single_comparison_5x10.png'
    plt.savefig(outp, bbox_inches='tight')
    plt.close()
    print('Saved combined single figure:', outp)

    plt.figure(figsize=(16,20))
    row = 0
    for i in range(5):
        for col, n in enumerate([2,3]):
            plt.subplot(5,2,row*2 + col + 1)
            fp = fb[i]
            if not fp:
                plt.axis('off')
                continue
            ng = top_ngrams_for_file(fp, n=n, top_k=12)
            labels = [t for t,_ in ng]
            vals = [c for _,c in ng]
            sns.barplot(x=vals, y=labels, palette='tab20')
            plt.title(f'B {os.path.basename(fp)} - {n}-grams')
        row += 1
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_combined_B_5files_ngrams.png')
    plt.close()


    # TF-IDF summary
    if tfidf:
        rows = [{'metric':'cosine','value':tfidf['cosine']}]
        export_csv(f"{args.out_prefix}_tfidf.csv", rows, fieldnames=['metric','value'])

    # per-file TF-IDF matrix and heatmap
    try:
        files_a, mat_a = compute_pairwise_tfidf_matrix(paths_a)
        files_b, mat_b = compute_pairwise_tfidf_matrix(paths_b)
        # save CSVs
        import csv
        if mat_a is not None:
            with open(f'{args.out_prefix}_tfidf_matrix_a.csv', 'w', encoding='utf-8', newline='') as f:
                w = csv.writer(f)
                w.writerow(['file'] + files_a)
                for i, row in enumerate(mat_a):
                    w.writerow([files_a[i]] + list(row))
        if mat_b is not None:
            with open(f'{args.out_prefix}_tfidf_matrix_b.csv', 'w', encoding='utf-8', newline='') as f:
                w = csv.writer(f)
                w.writerow(['file'] + files_b)
                for i, row in enumerate(mat_b):
                    w.writerow([files_b[i]] + list(row))
        # combined heatmap for cross-author (if both present)
        if mat_a is not None and mat_b is not None:
            # compute cross similarity between A-files and B-files via TF-IDF over all docs
            import numpy as np
            docs_all = []
            files_all = []
            for p in paths_a:
                if os.path.isdir(p):
                    for root, _, fns in os.walk(p):
                        for fn in fns:
                            if fn.lower().endswith('.txt'):
                                fp = os.path.join(root, fn)
                                with open(fp, 'r', encoding='utf-8') as f:
                                    docs_all.append(f.read())
                                files_all.append(os.path.relpath(fp))
            for p in paths_b:
                if os.path.isdir(p):
                    for root, _, fns in os.walk(p):
                        for fn in fns:
                            if fn.lower().endswith('.txt'):
                                fp = os.path.join(root, fn)
                                with open(fp, 'r', encoding='utf-8') as f:
                                    docs_all.append(f.read())
                                files_all.append(os.path.relpath(fp))
            if docs_all and TfidfVectorizer:
                vec = TfidfVectorizer(stop_words='english')
                X = vec.fit_transform(docs_all)
                na = len(files_a)
                nb = len(files_b)
                cross = cosine_similarity(X[:na], X[na:na+nb])
                plt.figure(figsize=(10,8))
                sns.heatmap(cross, xticklabels=files_b, yticklabels=files_a, cmap='viridis')
                plt.title('TF-IDF cross-similarity A vs B')
                plt.tight_layout(); plt.savefig(f'{args.out_prefix}_tfidf_cross_heatmap.png'); plt.close()
    except Exception:
        pass

    # generate multiple n-gram plots per-file (up to sample-files)
    try:
        plot_multiple_ngrams_for_author(paths_a, args.sample_files, args.out_prefix, 'A')
        plot_multiple_ngrams_for_author(paths_b, args.sample_files, args.out_prefix, 'B')
        plot_combined_ngrams(paths_a, paths_b, args.sample_files, args.out_prefix)
        plot_single_figure_both_authors(paths_a, paths_b, args.sample_files, args.out_prefix)
    except Exception:
        pass

    # per-file embeddings and heatmaps
    try:
        files_a_e, embs_a = compute_per_file_embeddings(paths_a)
        files_b_e, embs_b = compute_per_file_embeddings(paths_b)
        import numpy as np
        if embs_a is not None:
            # save embeddings
            np.savetxt(f'{args.out_prefix}_embeddings_a.csv', embs_a, delimiter=',')
        if embs_b is not None:
            np.savetxt(f'{args.out_prefix}_embeddings_b.csv', embs_b, delimiter=',')
        if embs_a is not None and embs_b is not None:
            cross = cosine_similarity(embs_a, embs_b)
            # save cross matrix
            import pandas as pd
            df = pd.DataFrame(cross, index=files_a_e, columns=files_b_e)
            df.to_csv(f'{args.out_prefix}_emb_cross.csv')
            plt.figure(figsize=(10,8))
            sns.heatmap(df, cmap='magma')
            plt.title('Embedding cross-similarity A vs B')
            plt.tight_layout(); plt.savefig(f'{args.out_prefix}_emb_cross_heatmap.png'); plt.close()
    except Exception:
        pass

    # print extended summary
    print('\nTop 10 trigrams A:\n', ngrams_a[:10])
    print('\nTop 10 trigrams B:\n', ngrams_b[:10])
    print('\nChorus lines A (count):\n', chorus_a[:10])
    print('\nChorus lines B (count):\n', chorus_b[:10])
    if tfidf:
        print('\nTF-IDF cosine similarity:', tfidf['cosine'])
    print('\nSentiment A:', sent_a)
    print('Sentiment B:', sent_b)
    # generate additional plots
    try:
        plot_additional(stats_a, stats_b, texts_a, texts_b, per_a, per_b, tfidf, ngrams_a, ngrams_b, args.out_prefix)
    except Exception:
        pass

    # topic modelling
    try:
        tm_a = build_topic_model(texts_a, num_topics=6)
        tm_b = build_topic_model(texts_b, num_topics=6)
        import json
        with open(f'{args.out_prefix}_topics_a.json', 'w', encoding='utf-8') as f:
            json.dump({'topics': tm_a['topics'] if tm_a else []}, f, ensure_ascii=False, indent=2)
        with open(f'{args.out_prefix}_topics_b.json', 'w', encoding='utf-8') as f:
            json.dump({'topics': tm_b['topics'] if tm_b else []}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # classifier
    try:
        clf_res = train_classifier(paths_a, paths_b, args.out_prefix)
    except Exception:
        clf_res = None

    # fingerprint and radar
    try:
        fp_a = build_fingerprint(stats_a)
        fp_b = build_fingerprint(stats_b)
        labels = list(fp_a.keys())
        plot_radar(fp_a, fp_b, labels, f'{args.out_prefix}_fingerprint_radar.png')
        import json
        with open(f'{args.out_prefix}_fingerprint_a.json','w',encoding='utf-8') as f:
            json.dump(fp_a,f,ensure_ascii=False,indent=2)
        with open(f'{args.out_prefix}_fingerprint_b.json','w',encoding='utf-8') as f:
            json.dump(fp_b,f,ensure_ascii=False,indent=2)
    except Exception:
        pass

    # print summary
    print('Author A stats:', stats_a['total_words'], 'words,', stats_a['unique_words'], 'unique')
    print('Author B stats:', stats_b['total_words'], 'words,', stats_b['unique_words'], 'unique')

    plot_comparison(stats_a, stats_b, args.author_a, args.author_b, out_prefix=args.out_prefix)

    # wordclouds
    try:
        generate_wordcloud(stats_a['raw_text'], f'{args.out_prefix}_wc_a.png')
        generate_wordcloud(stats_b['raw_text'], f'{args.out_prefix}_wc_b.png')
    except Exception:
        pass


def analyze_artists_folder(lang_dir: str, out_prefix: str):
    """Run analysis across all artist folders under lang_dir and produce summary CSV + TTR barplot."""
    artists = [d for d in os.listdir(lang_dir) if os.path.isdir(os.path.join(lang_dir, d))]
    rows = []
    for artist in artists:
        path = os.path.join(lang_dir, artist)
        texts = read_author_files([path])
        if not texts:
            continue
        stats = aggregate_stats(texts)
        unique = stats['unique_words']
        total = stats['total_words']
        ttr = unique / total if total else 0
        pos = stats.get('pos_counts', {})
        verbs = pos.get('VERB', 0) + pos.get('AUX', 0)
        adjectives = pos.get('ADJ', 0)
        nouns = pos.get('NOUN', 0)
        rows.append({'artist': artist, 'words': total, 'unique': unique, 'ttr': ttr, 'verbs_pct': (verbs/total*100) if total else 0, 'adj_pct': (adjectives/total*100) if total else 0, 'noun_pct': (nouns/total*100) if total else 0})
        # generate wordcloud per artist
        try:
            generate_wordcloud(stats['raw_text'], os.path.join(os.path.dirname(out_prefix), f'{artist}_wordcloud.png'))
        except Exception:
            pass
    # export CSV
    csv_path = f"{out_prefix}_artists_summary.csv"
    export_csv(csv_path, rows, fieldnames=['artist','words','unique','ttr','verbs_pct','adj_pct','noun_pct'])
    # barplot TTR
    try:
        import pandas as pd
        df = pd.DataFrame(rows).sort_values('ttr', ascending=False)
        plt.figure(figsize=(10, max(4, len(df)/4)))
        sns.barplot(x='ttr', y='artist', data=df, palette='magma')
        plt.title('Artist vocabulary richness (TTR)')
        outp = f"{out_prefix}_artists_ttr.png"
        out_dir = os.path.dirname(outp)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(outp, bbox_inches='tight')
        plt.close()
    except Exception:
        pass


if __name__ == '__main__':
    main()
