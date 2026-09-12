"""
Scrape Big Bang Theory transcripts from bigbangtrans.wordpress.com.

Output: data/raw_episodes.json
  [{"title":..., "url":..., "lines":[{"speaker":..., "text":..., "scene":int}, ...]}, ...]

NOTE: the scraped transcripts are copyrighted show dialogue. Keep data/ out of
version control (.gitignore) and ship this script instead.
"""

import json, re, time, argparse, pathlib, sys
import requests
from bs4 import BeautifulSoup

BASE = "https://bigbangtrans.wordpress.com/"
SEED = BASE + "series-1-episode-1-pilot-episode/"
HEADERS = {"User-Agent": "class-assignment-scraper/1.0"}

# "Sheldon: ..."  or  "Sheldon (mouths): ..."   (italic markers already stripped)
SPEAKER_RE = re.compile(r"^([A-Z][A-Za-z.'\- ]{0,28}?)\s*(?:\([^)]{0,40}\))?\s*:\s*(.+)$")

# lines that are pure stage direction / scene headers
SCENE_RE = re.compile(r"^\(?\s*Scene\s*:", re.I)


def normalize(s: str) -> str:
    for a, b in [("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'),
                 ("\u201d", '"'), ("\u2026", "..."), ("\u2013", "-"),
                 ("\u2014", "-"), ("\xa0", " ")]:
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


def get_episode_links(session):
    """Sidebar 'Pages' list appears on every page and holds every episode."""
    soup = BeautifulSoup(session.get(SEED, headers=HEADERS, timeout=30).text, "html.parser")
    seen, out = set(), []
    for a in soup.find_all("a", href=True):
        href = a["href"].split("#")[0].rstrip("/") + "/"
        title = normalize(a.get_text())
        if not re.match(r"^Series\s+\d+\s+Episode", title, re.I):
            continue
        if href in seen:
            continue
        seen.add(href)
        out.append({"title": title, "url": href})
    return out


def parse_episode(html):
    """Return list of {speaker, text, scene}. Scene index increments at scene headers."""
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find("div", class_=re.compile(r"entry|post-content|entrytext"))
    if body is None:
        body = soup

    lines, scene = [], 0
    for p in body.find_all("p"):
        # drop italic spans = stage directions, BEFORE reading text
        clone = BeautifulSoup(str(p), "html.parser")
        for tag in clone.find_all(["em", "i"]):
            tag.decompose()
        raw = normalize(clone.get_text())

        full = normalize(p.get_text())
        if SCENE_RE.match(full):
            scene += 1
            continue
        if not raw:
            continue
        if re.match(r"^Written by", full, re.I):
            continue

        m = SPEAKER_RE.match(raw)
        if not m:
            continue
        speaker, text = normalize(m.group(1)), normalize(m.group(2))
        # leftover parenthetical-only or empty
        text = re.sub(r"\(\s*\)", "", text).strip()
        if not text or not speaker:
            continue
        lines.append({"speaker": speaker, "text": text, "scene": scene})
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw_episodes.json")
    ap.add_argument("--limit", type=int, default=0, help="scrape only N episodes (testing)")
    ap.add_argument("--delay", type=float, default=1.0)
    args = ap.parse_args()

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    s = requests.Session()

    eps = get_episode_links(s)
    if args.limit:
        eps = eps[:args.limit]
    print(f"found {len(eps)} episode pages", file=sys.stderr)

    out = []
    for i, ep in enumerate(eps, 1):
        try:
            html = s.get(ep["url"], headers=HEADERS, timeout=30).text
            lines = parse_episode(html)
        except Exception as e:
            print(f"  !! {ep['title']}: {e}", file=sys.stderr)
            continue
        out.append({**ep, "lines": lines})
        print(f"[{i}/{len(eps)}] {ep['title']}: {len(lines)} lines", file=sys.stderr)
        time.sleep(args.delay)

    json.dump(out, open(args.out, "w"), indent=1)
    tot = sum(len(e["lines"]) for e in out)
    shel = sum(1 for e in out for l in e["lines"] if l["speaker"].lower() == "sheldon")
    print(f"\n{len(out)} episodes, {tot} lines, {shel} Sheldon lines -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
