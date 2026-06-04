# SemanticTextEngine — Author Comparison

This adds `explore_authors.py` to compare two authors' corpora (from `data/`).

Usage:

```bash
python explore_authors.py "authorA" "authorB" --language PL --data-root data --out-prefix results/compare
```

Features implemented:
- Basic lexical stats (word counts, unique, avg word length)
- Sentence stats (count, avg length)
- Type-token ratio, hapax
- Syllable estimation (pyphen / syllapy fallback)
- N-grams (trigrams) and chorus (repeated lines)
- TF-IDF + cosine similarity (if `scikit-learn` installed)
- Sentiment (VADER if installed)
- Rhyme-scheme heuristic based on line endings
- Exports CSVs and PNG plots in `--out-prefix`

Dependencies (recommended):
- numpy, pandas, matplotlib, seaborn
- scikit-learn (optional, TF-IDF)
- pyphen or syllapy (optional, syllable counts)
- vaderSentiment (optional, sentiment)
- nltk, spacy, textstat, wordcloud (optional enhancements)

Notes:
- The script is standalone; you don't need to delete `analyzer.py`.
- Optional libs improve results but are not strictly required.

If you want, I can run a test on `maryla-rodowicz` vs `perfect` and attach resulting CSVs and plots.