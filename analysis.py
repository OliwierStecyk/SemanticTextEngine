import os
import glob
import logging
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import nltk
from nltk.util import ngrams
from wordcloud import WordCloud

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

#Pobieranie zasobów NLTK
nltk.download('punkt', quiet=True)

### CZYSZCZENIE
#Polski model spaCy
try:
    nlp = spacy.load("pl_core_news_sm")
except OSError:
    logging.error("Nie znaleziono modelu pl_core_news_sm. Uruchom: python -m spacy download pl_core_news_sm")
    exit(1)


def load_all_lyrics_for_artist(artist_path):
    """Wczytuje wszystkie pliki .txt z folderu artysty i łączy je w jeden tekst."""
    txt_files = glob.glob(os.path.join(artist_path, "*.txt"))
    if not txt_files:
        return None

    all_text = []
    for fpath in txt_files:
        with open(fpath, "r", encoding="utf-8") as f:
            all_text.append(f.read())
    return "\n\n".join(all_text)


def analyze_text_with_spacy(text):
    """
    Przetwarza tekst za pomocą spaCy.
    Zwraca formy podstawowe (lemmaty) do analizy słownictwa oraz statystyki części mowy.
    """
    doc = nlp(text.lower())

    lemmas = []
    pos_counts = Counter()
    total_tokens = 0

    for token in doc:
        # Pomijamy spacje, interpunkcję i liczby
        if token.is_punct or token.is_space or token.like_num:
            continue

        total_tokens += 1
        pos_counts[token.pos_] += 1

        # Zapisujemy lemat (formę podstawową), jeśli słowo nie jest stop-wordem
        if not token.is_stop:
            lemmas.append(token.lemma_)

    return lemmas, pos_counts, total_tokens


def generate_artist_wordcloud(lemmas, artist_name, output_dir):
    """Generuje i zapisuje chmurę słów dla artysty."""
    if not lemmas:
        return
    text_for_wc = " ".join(lemmas)
    wc = WordCloud(width=800, height=400, background_color="white", max_words=100).generate(text_for_wc)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Chmura najczęstszych słów: {artist_name.upper()}", fontsize=14, pad=10)

    out_path = os.path.join(output_dir, f"{artist_name}_wordcloud.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def get_top_bigrams(lemmas, n=5):
    """Wyciąga najpopularniejsze pary słów (bigramy) przy użyciu NLTK."""
    if len(lemmas) < 2:
        return []
    bi_grams = ngrams(lemmas, 2)
    freq_dist = Counter(bi_grams)
    # Formatujemy jako czytelny tekst 'słowo1 + słowo2'
    return [f"{bg[0]} {bg[1]}" for bg, count in freq_dist.most_common(n)]


def main():
    # Ścieżka do pobranych danych (zakładamy, że langdetect zapisał je do folderu PL lub PL_CODE)
    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    # Przeszukaj podfoldery w poszukiwaniu języka polskiego (obsługa 'PL' oraz 'pl')
    pl_dir = os.path.join(data_dir, 'PL')
    if not os.path.exists(pl_dir):
        pl_dir = os.path.join(data_dir, 'pl')

    if not os.path.exists(pl_dir):
        logging.error(f"Nie znaleziono folderu z polskimi tekstami ({pl_dir}). Uruchom najpierw main.py!")
        return

    # Folder na wyniki analizy
    output_dir = os.path.join(os.path.dirname(__file__), 'analysis_results')
    os.makedirs(output_dir, exist_ok=True)

    artists_folders = [f for f in glob.glob(os.path.join(pl_dir, "*")) if os.path.isdir(f)]

    if not artists_folders:
        logging.warning("Folder językowy istnieje, ale jest pusty. Pobierz utwory!")
        return

    logging.info(f"Rozpoczynam analizę dla {len(artists_folders)} artystów...")

    results = []

    for folder in artists_folders:
        artist_name = os.path.basename(folder)
        logging.info(f"Analizuję artystę: {artist_name}")

        raw_text = load_all_lyrics_for_artist(folder)
        if not raw_text:
            continue

        # ANALIZA NLP
        lemmas, pos_counts, total_tokens = analyze_text_with_spacy(raw_text)

        if total_tokens == 0:
            continue

        # 1. Bogactwo językowe (Type-Token Ratio na lematach) = Liczba unikalnych słów/Całkowita liczba słów
        unique_lemmas = len(set(lemmas))
        ttr_richness = unique_lemmas / total_tokens if total_tokens > 0 else 0

        # 2. Ekstrakcja bigramów (NLTK)
        top_bigrams = get_top_bigrams(lemmas, n=3)

        # 3. Proporcje części mowy (Dynamika tekstu: Czasowniki vs Przymiotniki)
        verbs = pos_counts.get("VERB", 0) + pos_counts.get("AUX", 0)
        adjectives = pos_counts.get("ADJ", 0)
        nouns = pos_counts.get("NOUN", 0)

        # Generowanie chmury słów
        generate_artist_wordcloud(lemmas, artist_name, output_dir)

        # Zapis wyników do słownika
        results.append({
            "Artysta": artist_name,
            "Liczba słów (razem)": total_tokens,
            "Unikalne słowa (oczyszczone)": unique_lemmas,
            "Bogactwo językowe (TTR)": round(ttr_richness, 4),
            "Rzeczowniki %": round((nouns / total_tokens) * 100, 2) if total_tokens > 0 else 0,
            "Czasowniki %": round((verbs / total_tokens) * 100, 2) if total_tokens > 0 else 0,
            "Przymiotniki %": round((adjectives / total_tokens) * 100, 2) if total_tokens > 0 else 0,
            "Najczęstsze frazy": ", ".join(top_bigrams)
        })

    # Tworzenie DataFrame z wynikami
    df = pd.DataFrame(results)

    # Sortujemy według bogactwa językowego
    df = df.sort_values(by="Bogactwo językowe (TTR)", ascending=False)

    # Zapis raportu do CSV
    report_path = os.path.join(output_dir, "raport_koncowy.csv")
    df.to_csv(report_path, index=False, encoding="utf-8-sig")
    logging.info(f"Zapisano raport tekstowy w: {report_path}")

    # WIZUALIZACJA  (Matplotlib / Seaborn) ---
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    # Wykres bogactwa językowego
    ax = sns.barplot(x="Bogactwo językowe (TTR)", y="Artysta", data=df, palette="viridis")
    plt.title("Porównanie bogactwa językowego wykonawców (Type-Token Ratio)", fontsize=14, pad=15)
    plt.xlabel("Wskaźnik TTR (Wyższy = bogatsze słownictwo)")
    plt.ylabel("Artysta")

    # Dodanie wartości liczbowych na słupkach
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 0.002, p.get_y() + p.get_height() / 2 + 0.1, f'{width:.3f}', ha="left", va="center",
                fontsize=10)

    chart_path = os.path.join(output_dir, "porownanie_bogactwa_jezykowego.png")
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close()
    logging.info(f"Zapisano wykres porównawczy w: {chart_path}")

    print("\n=== PODSUMOWANIE ANALIZY W KONSOLI ===")
    print(df[["Artysta", "Bogactwo językowe (TTR)", "Czasowniki %", "Przymiotniki %"]].to_string(index=False))


if __name__ == "__main__":
    main()