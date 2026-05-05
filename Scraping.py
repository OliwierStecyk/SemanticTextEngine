import os
import requests
from bs4 import BeautifulSoup
from langdetect import detect, DetectorFactory
from urllib.parse import quote
import re


DetectorFactory.seed = 0

ARTIST_SLUG = "sylwia-grzeszczak" 
AUTOFETCH_LIMIT = 5

def find_lyrics_url(artist: str, title: str) -> str:
    query = f"{artist} {title} tekstowo"
    search_url = f"https://www.google.com/search?q={quote(query)}"

    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(resp.text, "html.parser")

    for a in soup.select("a"):
        href = a.get("href", "")
        if "tekstowo.pl" in href:
            return href.split("q=")[-1].split("&")[0]
    
    raise Exception("Nie znaleziono URL")

def find_artist_song_urls(artist_slug: str, limit: int = 10, strona: str = "https://www.tekstowo.pl" ) -> list:
    """
    Visit tekstowo artist page and return up to `limit` song URLs.
    """
    base = f"{strona}/{artist_slug}"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(base, headers=headers)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    links = []
    for a in soup.select('a'):
        href = a.get('href', '')
        if href and '/'+artist_slug.split('/')[-1]+'/' in href and 'tekstowo.pl' in href:
            # full url
            links.append(href)
        elif href.startswith('/') and artist_slug.split('/')[-1] in href:
            links.append('https://www.tekstowo.pl' + href)
        if len(links) >= limit:
            break
    seen = set()
    out = []
    for u in links:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out[:limit]

def scrape_lyrics(url: str) -> str:
	resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
	resp.raise_for_status()
	soup = BeautifulSoup(resp.text, 'html.parser')
	selectors = ["div.song-text", "div.lyrics", "div.text", "div#song-text"]
	for sel in selectors:
		node = soup.select_one(sel)
		if node and node.get_text(strip=True):
			return node.get_text(separator='\n').strip()
	parts = [p.get_text(separator='\n').strip() for p in soup.find_all('p')]
	content = '\n\n'.join([p for p in parts if p])
	if content:
		return content
	return soup.get_text(separator='\n').strip()

def save_lyrics(lyrics: str, lang: str, out_dir: str, artist: str, title: str):
    path = os.path.join(out_dir, lang)
    os.makedirs(path, exist_ok=True)

    def sanitize_filename(name: str) -> str:
        name = name.lower()
        name = name.replace(' ', '-')
        name = re.sub(r'[^a-z0-9\-_.]', '', name)
        return name[:200]

    filename = sanitize_filename(f"{artist}-{title}") + '.txt'
    file_path = os.path.join(path, filename)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(lyrics)

    return file_path

def main():
    
    artist_slug = ARTIST_SLUG or input("Podaj slug artysty z tekstowo (np. sylwia-grzeszczak): ").strip()
    if not artist_slug:
        print('No artist slug provided; exiting.')
        return

    try:
        urls = find_artist_song_urls(artist_slug, limit=AUTOFETCH_LIMIT)
    except Exception as e:
        print('Błąd pobierania listy utworów:', e)
        return

    if not urls:
        print('Nie znaleziono utworów na stronie artysty.')
        return

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

    list_file = save_song_list_html(artist_slug, urls, os.path.join(os.path.dirname(__file__), 'data'))
    print(f'Saved song list HTML to {list_file}')

    for u in urls:
        parts = u.rstrip('/').split('/')
        title = parts[-1].replace('.html', '')
        artist = parts[-2] if len(parts) >= 2 else artist_slug
        try:
            lyrics = scrape_lyrics(u)
        except Exception as e:
            print(f'Error fetching {artist} - {title}:', e)
            continue
        try:
            lang = detect(lyrics)
        except Exception:
            lang = 'unknown'
        lang_code = lang.upper() if lang and len(lang) == 2 else lang
        out_file = save_lyrics(
            lyrics,
            lang_code,
            os.path.join(os.path.dirname(__file__), 'data'),
            artist,
            title
        )
        print(f'Saved lyrics to {out_file} (lang={lang_code})')
if __name__ == '__main__':
	main()
