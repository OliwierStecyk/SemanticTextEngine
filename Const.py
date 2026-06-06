pl_set = [
    "marek-grechuta",
    "ewa-demarczyk",
    "dawid-podsiadlo",
    # "quebonafide",
    # "taco-hemingway",
    # "kult",
    # "mela-koteluk",
    "maryla-rodowicz",
    #"dziarma",
    "sanah",
    # "oh-myk",
    # "nosowska",
    # "komety",
    # "daria-zawialow",
    # "brodka",
    # "flirtini",
    # "znane-zespoly",
    # "anna-wyszkoni",
    "perfect",
    # "hey",
    "mazowsze"
]

#eng_set = []

# Dictionary grouping available artist sets
ARTIST_SETS = {
    'pl_set': pl_set,
   # 'eng_set': eng_set,
}

# Choose which set to use: 'pl' or 'eng'
ENV = 'pl'

# If ARTIST_SLUG is non-empty, main.py will process only that artist
ARTIST_SLUG = ""

AUTOFETCH_LIMIT = 50  # Limit pobierania utworów na artystę