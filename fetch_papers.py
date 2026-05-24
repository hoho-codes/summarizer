import os
import json
import feedparser
import requests
import time
from datetime import datetime, timezone
from pathlib import Path
from requests.exceptions import RequestException

# =========================
# Configuration
# =========================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

TOPICS = [t.strip().lower() for t in os.environ.get(
    "TOPICS", "quantum computing").split(",") if t.strip()]

MAX_PAPERS = 10
REQUEST_DELAY = 5          # arXiv courtesy delay
MAX_ABSTRACT_CHARS = 1500

# ---- Groq safety ----
GROQ_MODEL = "llama-3.3-70b-versatile"
MIN_GROQ_INTERVAL = 10     # seconds (≈6 RPM)
LAST_GROQ_CALL = 0

if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY")

# =========================
# arXiv category mappings
# =========================

ARXIV_CATEGORIES = {
    "quantum algorithms": ["quant-ph"],
    "quantum information": ["quant-ph"],
    "quantum computing": ["quant-ph"],
    "statistical physics": ["cond-mat.stat-mech"],
    "turbulence": ["physics.flu-dyn"],
    "geophysical turbulence": ["physics.ao-ph"],
    "fluid dynamics": ["physics.flu-dyn"],
    "atmospheric physics": ["physics.ao-ph"],
    "nonlinear sciences": ["nlin.CD", "nlin.PS", "nlin.SI"],
    "chaos": ["nlin.CD"],
    "pattern formation": ["nlin.PS"],
    "integrable systems": ["nlin.SI"],
}

# =========================
# arXiv Fetching
# =========================

def fetch_arxiv_papers(category, max_results):
    rss_url = f"http://export.arxiv.org/rss/{category}"
    feed = feedparser.parse(rss_url)

    if not feed.entries:
        return []

    papers = []
    for e in feed.entries[:max_results]:
        papers.append({
            "title": e.title.strip(),
            "authors": e.get("author", "Unknown"),
            "abstract": e.summary.strip(),
            "url": e.link,
            "published": e.get("published", "Unknown"),
        })
    return papers

# =========================
# Groq helpers
# =========================

def groq_call(payload):
    global LAST_GROQ_CALL

    now = time.time()
    wait = MIN_GROQ_INTERVAL - (now - LAST_GROQ_CALL)
    if wait > 0:
        time.sleep(wait)

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(3):
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )

            if r.status_code == 429:
                print("    Groq 429 — sleeping 20s")
                time.sleep(20)
                continue

            r.raise_for_status()
            LAST_GROQ_CALL = time.time()
            return r.json()["choices"][0]["message"]["content"].strip()

        except RequestException as e:
            print(f"    Groq error: {e}")
            time.sleep(10)

    return "- Summary unavailable."

def normalize_bullets(text, max_lines=3):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    bullets = []
    for l in lines:
        bullets.append(l if l.startswith("-") else f"- {l}")
        if len(bullets) >= max_lines:
            break
    return "\n".join(bullets)

def summarize_paper(title, abstract):
    abstract = abstract[:MAX_ABSTRACT_CHARS].replace("\n", " ").strip()

    prompt = (
        "Summarize this paper as 3 concise bullet points "
        "(max 12 words each):\n\n"
        f"Title: {title}\n"
        f"Abstract: {abstract}"
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 120,
    }

    return normalize_bullets(groq_call(payload))

def summarize_category(category, papers):
    unique_titles = list({p["title"] for p in papers})[:8]
    titles = "\n".join(f"- {t}" for t in unique_titles)

    prompt = (
        f"Summarize the following {category} papers "
        "in 2–3 short bullet points:\n\n"
        f"{titles}"
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 120,
    }

    return normalize_bullets(groq_call(payload), max_lines=3)

# =========================
# Main
# =========================

def main():
    print("Starting daily paper fetch...")

    seen_titles = set()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = Path("summaries")
    out_dir.mkdir(exist_ok=True)

    category_map = {}
    for topic in TOPICS:
        for cat in ARXIV_CATEGORIES.get(topic, ["quant-ph"]):
            category_map.setdefault(cat, []).append(topic)

    markdown = [
        f"# Research Paper Summaries — {today}\n\n",
        f"**Topics**: {', '.join(TOPICS)}\n\n"
    ]

    all_papers = []

    for category, topics in category_map.items():
        print(f"\nFetching {category} ({', '.join(topics)})")

        papers = fetch_arxiv_papers(category, MAX_PAPERS)
        time.sleep(REQUEST_DELAY)

        papers = [p for p in papers if p["title"] not in seen_titles]

        if not papers:
            print("  No new papers found in this category.")
            continue

        for p in papers:
            seen_titles.add(p["title"])
            print(f"  Summarizing: {p['title'][:60]}...")
            p["summary"] = summarize_paper(p["title"], p["abstract"])
            p["category"] = category
            all_papers.append(p)

        print("  Creating category summary...")
        cat_summary = summarize_category(category, papers)

        markdown.append(f"## {category}\n\n")
        markdown.append(f"{cat_summary}\n\n")

        for p in papers:
            markdown.append(f"### {p['title']}\n\n")
            markdown.append(f"**Authors:** {p['authors']}\n\n")
            markdown.append(f"{p['summary']}\n\n")
            markdown.append(f"[Read paper]({p['url']})\n\n---\n\n")

    with open(out_dir / f"{today}.json", "w", encoding="utf-8") as f:
        json.dump(all_papers, f, indent=2, ensure_ascii=False)

    with open(out_dir / f"{today}.md", "w", encoding="utf-8") as f:
        f.write("".join(markdown))

    print(f"\n✅ Saved {len(all_papers)} papers")

if __name__ == "__main__":
    main()
