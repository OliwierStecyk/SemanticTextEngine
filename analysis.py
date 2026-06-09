import os
import glob
import logging
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import spacy
import nltk
from nltk.util import ngrams
import Scraping
from Scraping import find_artist_song_urls, scrape_lyrics, save_lyrics

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
nltk.download('punkt', quiet=True)

try:
    nlp = spacy.load("pl_core_news_sm")
except OSError:
    logging.error("Uruchom w konsoli: python -m spacy download pl_core_news_sm")
    exit(1)


def fetch_data_for_battle(artist1_slug: str, artist2_slug: str, limit: int = 15):
    """Automatycznie sprawdza i pobiera piosenki dla dwóch artystów (po 15 utworów)."""
    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    for slug in [artist1_slug, artist2_slug]:
        logging.info(f"Sprawdzam dostępność utworów dla: {slug}")
        urls = find_artist_song_urls(slug, limit=limit)

        if not urls:
            logging.warning(f"Nie znaleziono utworów dla {slug} na Tekstowo!")
            continue

        for u in urls:
            parts = u.rstrip('/').split('/')
            title = parts[-1].replace('.html', '')
            artist_name = parts[-2] if len(parts) >= 2 else slug

            try:
                # Scraper z pliku Scraping.py zwraca lyrics oraz year
                lyrics, year = scrape_lyrics(u)
                # Zapisujemy w strukturze folderów
                save_lyrics(lyrics, "PL", data_dir, artist_name, title, year)
            except Exception as e:
                continue
    logging.info("Pobieranie i aktualizacja bazy piosenek zakończona.")


#NLP
def clean_and_analyze_lyrics(artist_slug: str):
    """Przeszukuje foldery i analizuje pobrane teksty wybranego artysty."""
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'PL')
    search_path = os.path.join(data_dir, "**", artist_slug, "*.txt")
    files = glob.glob(search_path, recursive=True)

    if not files:
        search_path_alt = os.path.join(data_dir, artist_slug, "*.txt")
        files = glob.glob(search_path_alt)

    if not files:
        logging.error(f"Brak pobranych plików tekstowych dla {artist_slug} w data/PL/")
        return None

    all_songs_data = []

    web_noise = {"tekst", "piosenka", "utwór", "zwrotka", "refren", "wykonawca", "teledysk", "tłumaczenie", "wersuj.pl",
                 "li", "la", "laj"}

    for fpath in files:
        path_parts = fpath.split(os.sep)
        # Przykładowo: .../data/PL/2021/sanah/utwor.txt -> rok to 3. element od końca
        year_str = path_parts[-3]
        year = int(year_str) if year_str.isdigit() else None

        with open(fpath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        if not raw_text.strip():
            continue

        # Analiza struktury wiersza przed oczyszczeniem słów
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        avg_line_length = sum(len(line.split()) for line in lines) / len(lines) if lines else 0

        # Przetwarzanie spaCy (tylko tagowanie części mowy i lematyzacja)
        doc = nlp(raw_text.lower())

        lemmas = []
        pos_counts = Counter()
        word_lengths = []
        total_tokens = 0

        for token in doc:
            if token.is_punct or token.is_space or token.like_num:
                continue

            total_tokens += 1
            pos_counts[token.pos_] += 1
            word_lengths.append(len(token.text))

            # Nasze reguły lingwistyczne (czyszczenie i poprawianie form słów bez ciężkiego ML)
            if not token.is_stop and token.lemma_ not in web_noise and len(token.lemma_) > 1:
                lemma = token.lemma_

                # Korekta czasowników
                if token.pos_ in ["VERB", "AUX"] and not (lemma.endswith("ć") or lemma.endswith("c")):
                    if lemma.endswith("i") or lemma.endswith("y"):
                        lemma += "ć"
                    elif lemma.endswith("a"):
                        lemma = lemma[:-1] + "ać"

                # Korekta przymiotników
                elif token.pos_ == "ADJ" and (lemma.endswith("a") or lemma.endswith("e")):
                    if lemma[:-1].endswith("k") or lemma[:-1].endswith("g"):
                        lemma = lemma[:-1] + "i"
                    else:
                        lemma = lemma[:-1] + "y"

                lemmas.append(lemma)

        if total_tokens == 0:
            continue

        avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0

        all_songs_data.append({
            "filename": os.path.basename(fpath),
            "year": year,
            "total_words": total_tokens,
            "lemmas": lemmas,
            "pos_counts": pos_counts,
            "avg_word_len": avg_word_length,
            "avg_line_len": avg_line_length
        })

    return all_songs_data


# WYKRESY (OSOBNE OKNA I PLIKI)
def plot_artist_battle(data1, data2, name1, name2):
    sns.set_theme(style="whitegrid")
    current_dir = os.path.dirname(os.path.abspath(__file__))

    battle_folder_name = f"{name1}_vs_{name2}"
    output_dir = os.path.join(current_dir, 'results', battle_folder_name)

    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Zapisuję wykresy w katalogu wynikowym: {output_dir}")
    # Przygotowanie danych zbiorczych do wykresów pudełkowych
    rows = []
    for d in data1: rows.append(
        {"Artysta": name1, "Słowa": d["total_words"], "DłSłowa": d["avg_word_len"], "DłWersu": d["avg_line_len"],
         "TTR": len(set(d["lemmas"])) / d["total_words"] if d["total_words"] else 0})
    for d in data2: rows.append(
        {"Artysta": name2, "Słowa": d["total_words"], "DłSłowa": d["avg_word_len"], "DłWersu": d["avg_line_len"],
         "TTR": len(set(d["lemmas"])) / d["total_words"] if d["total_words"] else 0})
    df_box = pd.DataFrame(rows)

    # --- WYKRES 1: Liczba słów ---
    plt.figure(figsize=(8, 5))
    # Dodane hue="Artysta" oraz legend=False
    sns.boxplot(x="Artysta", y="Słowa", data=df_box, hue="Artysta", palette="Set2", legend=False)
    plt.title(f"Rozkład liczby słów w piosenkach: {name1.upper()} vs {name2.upper()}", fontsize=12, weight='bold')
    plt.savefig(os.path.join(output_dir, "wykres_1_liczba_slow.png"), bbox_inches="tight", dpi=150)

    # --- WYKRES 2: Bogactwo językowe (TTR) ---
    plt.figure(figsize=(8, 5))
    # Dodane hue="Artysta" oraz legend=False
    sns.boxplot(x="Artysta", y="TTR", data=df_box, hue="Artysta", palette="Set2", legend=False)
    plt.title("Bogactwo językowe (Type-Token Ratio)", fontsize=12, weight='bold')
    plt.savefig(os.path.join(output_dir, "wykres_2_ttr.png"), bbox_inches="tight", dpi=150)

    # --- WYKRES 3: Średnia długość słowa ---
    plt.figure(figsize=(8, 5))
    # Dodane hue="Artysta" oraz legend=False
    sns.boxplot(x="Artysta", y="DłSłowa", data=df_box, hue="Artysta", palette="Set2", legend=False)
    plt.title("Średnia długość słowa (liczba liter)", fontsize=12, weight='bold')
    plt.savefig(os.path.join(output_dir, "wykres_3_dlugosc_slowa.png"), bbox_inches="tight", dpi=150)

    # --- WYKRES 4: Średnia długość wersu ---
    plt.figure(figsize=(8, 5))
    # Dodane hue="Artysta" oraz legend=False
    sns.boxplot(x="Artysta", y="DłWersu", data=df_box, hue="Artysta", palette="Set2", legend=False)
    plt.title("Średnia długość wersu (liczba słów)", fontsize=12, weight='bold')
    plt.savefig(os.path.join(output_dir, "wykres_4_dlugosc_wersu.png"), bbox_inches="tight", dpi=150)

    # --- WYKRES 5: Profil części mowy (Gramatyka) ---
    pos_rows = []
    for d, name in [(data1, name1), (data2, name2)]:
        for item in d:
            t = item["total_words"]
            v = item["pos_counts"].get("VERB", 0) + item["pos_counts"].get("AUX", 0)
            a = item["pos_counts"].get("ADJ", 0)
            n = item["pos_counts"].get("NOUN", 0)
            pos_rows.append({"Artysta": name, "Część mowy": "Rzeczowniki %", "Wartość": (n / t) * 100})
            pos_rows.append({"Artysta": name, "Część mowy": "Czasowniki %", "Wartość": (v / t) * 100})
            pos_rows.append({"Artysta": name, "Część mowy": "Przymiotniki %", "Wartość": (a / t) * 100})
    df_pos = pd.DataFrame(pos_rows)

    plt.figure(figsize=(9, 5))
    sns.barplot(x="Część mowy", y="Wartość", hue="Artysta", data=df_pos, palette="Set2")
    plt.title("Porównanie profilu części mowy (%)", fontsize=12, weight='bold')
    plt.ylabel("Udział procentowy w tekście")
    plt.savefig(os.path.join(output_dir, "wykres_5_czesci_mowy.png"), bbox_inches="tight", dpi=150)

    # --- WYKRES 6: Zmiana objętości tekstów w czasie ---
    years_data = []
    for d, name in [(data1, name1), (data2, name2)]:
        for item in d:
            if item["year"]:
                years_data.append({"Artysta": name, "Rok": item["year"], "Słowa": item["total_words"]})
    if years_data:
        plt.figure(figsize=(10, 5))
        df_years = pd.DataFrame(years_data).groupby(["Artysta", "Rok"]).mean().reset_index()
        sns.lineplot(x="Rok", y="Słowa", hue="Artysta", marker="o", data=df_years, palette="Set2")
        plt.title("Średnia liczba słów na przestrzeni lat", fontsize=12, weight='bold')
        plt.savefig(os.path.join(output_dir, "wykres_6_trendy_czas.png"), bbox_inches="tight", dpi=150)

    # --- WYKRES 7: Najczęstsze pojedyncze słowa ---
    all_lemmas1 = [l for d in data1 for l in d["lemmas"]]
    all_lemmas2 = [l for d in data2 for l in d["lemmas"]]
    words_c1 = Counter(all_lemmas1).most_common(3)
    words_c2 = Counter(all_lemmas2).most_common(3)
    df_words = pd.DataFrame(
        [{"Artysta": name1, "Słowo": w, "Liczba": c} for w, c in words_c1] +
        [{"Artysta": name2, "Słowo": w, "Liczba": c} for w, c in words_c2]
    )
    plt.figure(figsize=(9, 5))
    sns.barplot(x="Liczba", y="Słowo", hue="Artysta", data=df_words, palette="Set2")
    plt.title("Najpopularniejsze słowa-klucze (Top 3)", fontsize=12, weight='bold')
    plt.savefig(os.path.join(output_dir, "wykres_7_slowa_klucze.png"), bbox_inches="tight", dpi=150)

    # --- WYKRES 8: Najczęstsze frazy 3-wyrazowe (Trigramy) ---
    tg1 = [" ".join(gram) for gram in ngrams(all_lemmas1, 3)] if len(all_lemmas1) >= 3 else []
    tg2 = [" ".join(gram) for gram in ngrams(all_lemmas2, 3)] if len(all_lemmas2) >= 3 else []
    tg_c1 = Counter(tg1).most_common(3)
    tg_c2 = Counter(tg2).most_common(3)
    df_tg = pd.DataFrame(
        [{"Artysta": name1, "Fraza (3 słowa)": tg, "Liczba": c} for tg, c in tg_c1] +
        [{"Artysta": name2, "Fraza (3 słowa)": tg, "Liczba": c} for tg, c in tg_c2]
    )
    plt.figure(figsize=(11, 5))
    sns.barplot(x="Liczba", y="Fraza (3 słowa)", hue="Artysta", data=df_tg, palette="Set2")
    plt.title("Najczęstsze frazy 3-wyrazowe (Trigramy)", fontsize=12, weight='bold')
    plt.savefig(os.path.join(output_dir, "wykres_8_trigramy.png"), bbox_inches="tight", dpi=150)

    logging.info(f"Wszystkie osobne wykresy zostały zapisane w folderze skryptu.")
    plt.show()


#MAIN
if __name__ == "__main__":
    ARTIST_A = "maryla-rodowicz"
    ARTIST_B = "mazowsze"

    fetch_data_for_battle(ARTIST_A, ARTIST_B, limit=15)

    logging.info("Rozpoczynam analizę lingwistyczną...")
    data_A = clean_and_analyze_lyrics(ARTIST_A)
    data_B = clean_and_analyze_lyrics(ARTIST_B)

    if data_A and data_B:
        plot_artist_battle(data_A, data_B, ARTIST_A, ARTIST_B)
    else:
        logging.error("Nie udało się zebrać wystarczającej ilości danych do wygenerowania wykresu.")