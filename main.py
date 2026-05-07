import os
import re
import time
import logging
from Const import ARTIST_SETS, AUTOFETCH_LIMIT, ARTIST_SLUG, ENV
import Scraping
from Scraping import find_artist_song_urls, scrape_lyrics, save_song_list_html, save_lyrics
from langdetect import detect, DetectorFactory, LangDetectException
import argparse


DetectorFactory.seed = 0

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def process_artist(artist_slug: str):
    try:
        urls = find_artist_song_urls(artist_slug, limit=AUTOFETCH_LIMIT)
    except Exception as e:
        logging.error('Błąd pobierania listy utworów dla %s: %s', artist_slug, e)
        return

    if not urls:
        logging.info('Nie znaleziono utworów na stronie artysty %s.', artist_slug)
        return

    list_file = save_song_list_html(artist_slug, urls, os.path.join(os.path.dirname(__file__), 'data'))
    logging.info('Saved song list HTML to %s', list_file)

    for u in urls:
        parts = u.rstrip('/') .split('/')
        title = parts[-1].replace('.html', '')
        artist = parts[-2] if len(parts) >= 2 else artist_slug
        try:
            lyrics = scrape_lyrics(u)
        except Exception as e:
            logging.error('Error fetching %s - %s: %s', artist, title, e)
            continue
        try:
            lang = detect(lyrics)
        except LangDetectException:
            lang = 'unknown'
        except Exception:
            lang = 'unknown'
        lang_code = lang.upper() if lang and isinstance(lang, str) and len(lang) == 2 else lang
        out_file = save_lyrics(
            lyrics,
            lang_code,
            os.path.join(os.path.dirname(__file__), 'data'),
            artist,
            title
        )
        logging.info('Saved lyrics to %s (lang=%s)', out_file, lang_code)
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', help="Environment key: 'pl' or 'eng'", default=None)
    args = parser.parse_args()

    env = args.env if args.env else ENV

    if ARTIST_SLUG:
        logging.info('Using ARTIST_SLUG from Const: %s', ARTIST_SLUG)
        process_artist(ARTIST_SLUG)
    else:
        key = f"{env}_set"
        artists = ARTIST_SETS.get(key, [])
        logging.info('No ARTIST_SLUG set — processing %s (%d artists)', key, len(artists))
        for a in artists:
            process_artist(a)


if __name__ == '__main__':
    main()
