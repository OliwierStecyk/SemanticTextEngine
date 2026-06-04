import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote, urljoin
import re
import time
from Const import ARTIST_SLUG, AUTOFETCH_LIMIT




def find_lyrics_url(artist: str, title: str) -> str:
    query = f"{artist} {title} tekstowo"
    search_url = f"https://www.google.com/search?q={quote(query)}"

    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(search_url, headers=headers, timeout=5)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    for a in soup.select("a"):
        href = a.get("href", "")
        if "tekstowo.pl" in href:
            # try to extract actual URL or return joined url
            if href.startswith('/url?q='):
                href = href.split('/url?q=')[-1].split('&')[0]
            return href
    
    raise Exception("Nie znaleziono URL")

def find_artist_song_urls(artist_slug: str, limit: int = 10, strona: str = "https://www.tekstowo.pl" ) -> list:
    """
    Visit tekstowo artist page and return up to `limit` song URLs.
    """
    base = f"{strona}/{artist_slug}"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(base, headers=headers, timeout=5)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    links = []
    for a in soup.select('a'):
        href = a.get('href', '')
        try:
            if href.startswith('/'):
                full = urljoin(strona, href)
            else:
                full = href
        except Exception:
            full = href

        if not full:
            continue

        if 'tekstowo.pl' in full and f"/{artist_slug.split('/')[-1]}/" in full:
            links.append(full)
        if len(links) >= limit:
            break
    seen = set()
    out = []
    for u in links:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out[:limit]

def _extract_year_from_text(text: str) -> str:
    m = re.search(r'\b(19\d{2}|20\d{2})\b', text)
    if m:
        return m.group(1)
    return 'unknown'


def scrape_lyrics(url: str) -> tuple:
    """Return (lyrics, year) where year is heuristically extracted from the page."""
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')

    # try common lyric blocks
    selectors = ["div.song-text", "div.lyrics", "div.text", "div#song-text"]
    for sel in selectors:
        node = soup.select_one(sel)
        if node and node.get_text(strip=True):
            lyrics = node.get_text(separator='\n').strip()
            break
    else:
        parts = [p.get_text(separator='\n').strip() for p in soup.find_all('p')]
        lyrics = '\n\n'.join([p for p in parts if p]) if parts else soup.get_text(separator='\n').strip()

    # Heuristic: look for year in metadata blocks or near title/artist
    year = 'unknown'
    # Check meta tags
    for meta in soup.find_all('meta'):
        if meta.get('name', '').lower() in ('date', 'release', 'dcterms.date') or 'pub' in meta.get('property', '').lower():
            content = meta.get('content', '')
            year = _extract_year_from_text(content)
            if year != 'unknown':
                break

    if year == 'unknown':
        # Search visible page text near possible info boxes
        info_texts = []
        for sel in ['div.info', 'div.song-info', 'div#song-info', 'p.song-meta', 'div.release']:
            node = soup.select_one(sel)
            if node:
                info_texts.append(node.get_text(' ', strip=True))
        # also use title and headings
        title_h = soup.find(['h1', 'h2', 'h3'])
        if title_h:
            info_texts.append(title_h.get_text(' ', strip=True))
        combined = ' '.join(info_texts)
        if combined:
            year = _extract_year_from_text(combined)

    if year == 'unknown':
        # fallback: scan entire page text for first plausible year
        year = _extract_year_from_text(soup.get_text(' ', strip=True))

    return lyrics, year

def save_lyrics(lyrics: str, lang: str, out_dir: str, artist: str, title: str, year: str = None):
    # sanitize artist folder and filename
    def sanitize_part(name: str) -> str:
        n = (name or '').lower()
        n = n.replace(' ', '-')
        n = re.sub(r'[^a-z0-9\-_.]', '', n)
        return n[:200]

    artist_safe = sanitize_part(artist)
    year_safe = 'unknown' if not year else re.sub(r'[^0-9]', '', str(year))[:4]
    path = os.path.join(out_dir, lang, year_safe, artist_safe)
    os.makedirs(path, exist_ok=True)
    filename = sanitize_part(title) + '.txt'
    file_path = os.path.join(path, filename)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(lyrics)

    return file_path


def save_song_list_html(artist_slug: str, urls: list, out_dir: str):
    path = os.path.join(out_dir, 'lists')
    os.makedirs(path, exist_ok=True)
    fname = f"{artist_slug}-songs.html"
    fname = re.sub(r'[^a-z0-9\-_.]', '', fname.lower())
    file_path = os.path.join(path, fname)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('<!doctype html>\n<html><head><meta charset="utf-8"><title>Songs</title></head><body>')
        f.write(f'<h1>Songs for {artist_slug}</h1>\n<ul>')
        for u in urls:
            f.write(f'<li><a href="{u}">{u}</a></li>\n')
        f.write('</ul></body></html>')
    return file_path

