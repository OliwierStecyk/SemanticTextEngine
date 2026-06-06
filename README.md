# SemanticTextEngine

Szczegółowy przewodnik: jak przygotować środowisko, które pakiety zainstalować, oraz jak uruchomić skrypt `explore_authors.py` z przykładami i opisem funkcji.

## Spis treści
- Wymagania
- Tworzenie i aktywacja środowiska (venv)
- Instalacja zależności (minimalna i pełna)
- Przygotowania opcjonalne (NLTK, spaCy, modele, SBERT)
- Przykłady uruchomień
- Opis najważniejszych funkcji w `explore_authors.py`
- Wyjścia / pliki wynikowe
- Rozwiązywanie problemów


## Wymagania
- Python 3.8+ (zalecane 3.10/3.11)
- Dostęp do internetu przy instalacji dodatkowych modeli (opcjonalne)


## 1) Tworzenie i aktywacja środowiska (Windows PowerShell)
Otwórz PowerShell w katalogu projektu i wykonaj:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
W systemach UNIX/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```


## 2) Instalacja zależności
Plik `requirements.txt` może nie zawierać wszystkich pakietów używanych opcjonalnie. Poniżej dwie propozycje instalacji.

- Minimalne (szybsze, obsługuje podstawowe statystyki i wykresy):
```powershell
pip install matplotlib seaborn pandas scikit-learn
```

- Pełne (wszystkie funkcje: NLP, embeddingi, wordcloud, topic modelling):
```powershell
pip install nltk spacy textstat gensim scikit-learn sentence-transformers scipy numpy pyphen syllapy vaderSentiment wordcloud pandas seaborn matplotlib
```

Uwaga: `sentence-transformers` pobiera modele (ok. kilkadziesiąt–kilkaset MB). Gdy instalujesz dużo pakietów, przygotuj odpowiednią przepustowość i czas.


## 3) Przygotowania opcjonalne
- NLTK (do tokenizacji i wyznaczania zdań):
```powershell
python -c "import nltk; nltk.download('punkt')"
```
- spaCy (polski model, daje POS):
```powershell
pip install spacy
python -m spacy download pl_core_news_sm
```
- Jeśli chcesz embeddingi (sentence-transformers), nie trzeba niczego dodatkowego poza instalacją pakietu — model `all-MiniLM-L6-v2` zostanie pobrany automatycznie przy pierwszym użyciu.


## 4) Jak uruchomić `explore_authors.py` — przykłady
Skrypt porównuje dwóch autorów (podając katalog lub nazwę autora — skrypt spróbuje wyszukać folder w `--data-root`/`--language`). Wyniki to wykresy i pliki CSV/JSON zapisane z prefiksem `--out-prefix`.

- Porównanie dwóch folderów:
```powershell
python explore_authors.py "data/PL/2024/komety" "data/PL/2025/sanah" --out-prefix results/komety_vs_sanah --sample-files 5
```

- Porównanie wg nazwy autora (skrypt wyszuka katalog w `--data-root` i `--language`):
```powershell
python explore_authors.py "sanah" "maryla-rodowicz" --data-root data --language PL --out-prefix results/sanah_vs_maryla
```

- Szybki test na małej próbce (minimalne zależności wystarczą):
```powershell
python explore_authors.py "data/EN/hey" "data/EN/2021/sanah" --out-prefix tmp/test --sample-files 2
```


## 5) Co robi skrypt — opis funkcji (skrótowo)
- `read_author_files(paths)`: czyta wszystkie pliki `.txt` z podanych ścieżek.
- `clean_text(text)`: czyści tekst (usuwa znaki specjalne z uwzględnieniem polskich liter) i normalizuje spacje.
- `tokenize(text)`: tokenizacja (NLTK jeśli dostępny, inaczej `split`).
- `aggregate_stats(texts)`: zbiera statystyki: liczba słów, unikalne słowa, średnia długość słowa, średnia długość zdania, liczba sylab (heurystyka / pyphen / syllapy), top słów, wykrywanie POS (jeśli spaCy), wskaźniki czytelności (jeśli textstat).
- `find_ngrams(texts, n, top_k)`: zwraca najczęstsze n-gramy.
- `detect_chorus(texts)`: wykrywa powtarzające się dokładne linie (np. refreny).
- `compute_tfidf_similarity(texts_a, texts_b)`: TF-IDF na dokumentach A i B + kosinusowa miara podobieństwa oraz top cechy TF-IDF.
- `compute_pairwise_tfidf_matrix(paths)`: TF-IDF per-file i macierz podobieństw plików.
- `compute_per_file_embeddings(paths)`: embeddingi per file (wymaga `sentence-transformers`).
- `build_topic_model(texts)`: prosty LDA (gensim).
- `train_classifier(paths_a, paths_b)`: buduje prosty klasyfikator (LogisticRegression) na TF-IDF per-file i zapisuje raport.
- Funkcje wizualizacyjne: `plot_comparison`, `plot_additional`, `plot_radar`, `plot_zipf`, `plot_single_figure_both_authors`, itp. — tworzą pliki PNG z analizami.


## 6) Wyjścia (pliki wynikowe)
- Pliki mają prefiks `--out-prefix`. Przykłady plików generowanych:
	- `_metrics.csv`, `_ngrams_a.csv`, `_ngrams_b.csv`
	- `_per_file_a.csv`, `_per_file_b.csv`
	- `_tfidf_matrix_a.csv`, `_tfidf_matrix_b.csv`, `_tfidf_cross_heatmap.png`
	- `_embeddings_a.csv`, `_embeddings_b.csv`, `_emb_cross.csv`, `_emb_cross_heatmap.png` (gdy embeddingi dostępne)
	- `_topics_a.json`, `_topics_b.json` (topic modelling)
	- `_classifier_report.json` (jeśli klasyfikator został wytrenowany)
	- `_fingerprint_a.json`, `_fingerprint_b.json`, `_fingerprint_radar.png`
	- różne wykresy PNG: `_metrics.png`, `_top_words.png`, `_trigrams.png`, `_syl_kde.png`, `_wordlen_hist.png` itd.


## 7) Rozwiązywanie problemów
- Brak wyników / pusty folder wynikowy: upewnij się, że podane ścieżki zawierają pliki `.txt`.
- Błędy importu: jeśli skrypt pomija funkcjonalność (np. brak spaCy), zobacz komunikaty w konsoli i zainstaluj brakujące pakiety z instrukcji powyżej.
- Problemy z pamięcią przy `sentence-transformers`: jeśli model nie chce się załadować, spróbuj uruchomić bez tej biblioteki lub zainstalować mniejszy model.
- Jeśli potrzebujesz pełnego logowania, mogę dodać flagę `--debug` do skryptu (włączając `logging` i wypisując dostępność bibliotek).


## 8) Propozycje dalszych zmian (mogę zrobić za Ciebie)
- Dodać `requirements.txt` (minimalny / pełny).
- Dodać `--debug` flagę pokazującą, które biblioteki są dostępne i dlaczego pewne bloki zostały pominięte.
- Dodać prosty wrapper `run_example.ps1` z gotowymi poleceniami do testów.


python explore_authors.py "maryla-rodowicz" "perfect" --language PL --data-root data --out-prefix results/maryla_vs_perfect                          
 
- uruchomienie explore_authors.py

---
Jeśli chcesz, wygeneruję teraz `requirements.txt` (minimalny lub pełny) i/lub dopiszę `README.md` z przykładowym `run_example.ps1`.
