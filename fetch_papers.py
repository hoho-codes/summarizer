import os
import json
import feedparser
import requests
import time
import re
from datetime import datetime, timezone
from pathlib import Path
from requests.exceptions import RequestException
from collections import defaultdict

# =========================
# Configuration
# =========================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GROQ_API_KEY and not GEMINI_API_KEY:
    raise RuntimeError("At least one API key must be set.")

TOPICS = [
    t.strip().lower()
    for t in os.environ.get("TOPICS", "quantum computing").split(",")
    if t.strip()
]

MAX_PAPERS = 10
REQUEST_DELAY = 5
MAX_ABSTRACT_CHARS = 1500
CATEGORY_TOP_N = 8

PAPER_MODEL_GROQ = "llama-3.1-8b-instant"
CATEGORY_MODEL_GROQ = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-1.5-flash"

MIN_GROQ_INTERVAL = 12
MIN_GEMINI_INTERVAL = 4
MAX_GROQ_CALLS = 25

LAST_GROQ_CALL = 0.0
LAST_GEMINI_CALL = 0.0
GROQ_CALL_COUNT = 0

session = requests.Session()
CACHE_PATH = Path("summaries/cache.json")

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
# Cache Helpers
# =========================

def load_cache():
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("[!] Cache corrupted, resetting.")
    return {}

def save_cache(cache):
    CACHE_PATH.parent.mkdir(exist_ok=True)
    tmp_path = CACHE_PATH.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(CACHE_PATH)

def arxiv_id_from_url(url: str):
    if not url:
        return None
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([^?#]+)", url)
    if not m:
        return None
    pid = m.group(1).replace(".pdf", "").strip()
    return re.sub(r"v\d+$", "", pid)

# =========================
# Fetch & LLM Engine
# =========================

def fetch_arxiv_papers(category, max_results):
    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": f"cat:{category}",
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    headers = {
        "User-Agent": "ResearchPaperSummarizer/1.0 (contact: github-bot)"
    }

    try:
        r = session.get(url, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        feed = feedparser.parse(r.content)
    except RequestException as e:
        print(f"[!] arXiv request failed for {category}: {e}")
        return []
    except Exception as e:
        print(f"[!] feed parsing failed for {category}: {e}")
        return []

    papers = []
    for e in feed.entries:
        title = re.sub(r"\s+", " ", getattr(e, "title", "").replace("\n", "").strip())
        abstract = re.sub(r"\s+", " ", getattr(e, "summary", "").replace("\n", "").strip())
        authors = ", ".join(getattr(a, "name", "Unknown") for a in getattr(e, "authors", []))

        papers.append({
            "title": title,
            "authors": authors or "Unknown",
            "abstract": abstract,
            "url": getattr(e, "link", ""),
            "published": getattr(e, "published", "Unknown"),
        })

    return papers

# =========================
# LLM Engine
# =========================

def get_llm_summary(prompt, model):
    global LAST_GROQ_CALL, LAST_GEMINI_CALL, GROQ_CALL_COUNT

    # ---------------- Groq ----------------
    if GROQ_API_KEY and GROQ_CALL_COUNT < MAX_GROQ_CALLS:
        elapsed = time.time() - LAST_GROQ_CALL
        if elapsed < MIN_GROQ_INTERVAL:
            time.sleep(MIN_GROQ_INTERVAL - elapsed)

        try:
            r = session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 150,
                },
                timeout=30,
            )

            if r.status_code == 200:
                LAST_GROQ_CALL = time.time()
                GROQ_CALL_COUNT += 1
                return r.json()["choices"][0]["message"]["content"].strip()

            print(f"[!] Groq error {r.status_code}: {r.text[:120]}")

        except Exception as e:
            print(f"[!] Groq request failed: {e}")

    # ---------------- Gemini fallback ----------------
    if GEMINI_API_KEY:
        elapsed = time.time() - LAST_GEMINI_CALL
        if elapsed < MIN_GEMINI_INTERVAL:
            time.sleep(MIN_GEMINI_INTERVAL - elapsed)

        try:
            r = session.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.2,
                        "maxOutputTokens": 150,
                    },
                },
                timeout=30,
            )

            LAST_GEMINI_CALL = time.time()

            data = r.json()
            return (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                .strip()
            )

        except Exception as e:
            print(f"[!] Gemini request failed: {e}")

    return None

# =========================
# Processing Logic
# =========================

def format_bullets(text, max_lines=3, min_lines=3):
    if not text:
        return None

    def clean(l):
        return re.sub(r'^[\*\-\•]|\d+[\.\)]', '', l).strip()

    lines = []
    for l in text.splitlines():
        c = clean(l)
        if c:
            lines.append(f"- {c}")

    if len(lines) < min_lines:
        return None

    return "\n".join(lines[:max_lines])


def summarize_paper(paper, cache, seen):
    pid = arxiv_id_from_url(paper.get("url"))
    if not pid or pid in seen:
        return None

    if pid in cache and isinstance(cache[pid], dict):
        seen.add(pid)
        return cache[pid].get("summary")

    # UPDATED PROMPT:
    # Explicitly asking for NO introductory text and exactly 3 points.
    prompt = (
        "Provide exactly 3 bullet points summarizing the paper below. "
        "Do not include any introductory or concluding sentences. "
        "Each bullet point must be a maximum of 12 words.\n\n"
        f"Title: {paper['title']}\n"
        f"Abstract: {paper['abstract'][:MAX_ABSTRACT_CHARS]}"
    )

    summary = format_bullets(get_llm_summary(prompt, PAPER_MODEL_GROQ))

    if summary:
        cache[pid] = {
            "title": paper["title"],
            "summary": summary,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }
        seen.add(pid)

    return summary

def summarize_category(category, papers):
    valid = [p for p in papers if p.get("summary")]
    if not valid:
        return ""

    titles = "\n".join(f"- {p['title']}" for p in valid[:CATEGORY_TOP_N])
    prompt = f"Summarize themes in these {category} papers:\n\n{titles}"

    return format_bullets(get_llm_summary(prompt, CATEGORY_MODEL_GROQ)) or ""

# =========================
# Main
# =========================

def main():
    Path("summaries").mkdir(exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    cache = load_cache()
    seen = set()
    all_papers = []

    markdown = [
        f"# Research Paper Summaries — {today}\n\n",
        f"Topics: {', '.join(TOPICS)}\n\n",
    ]

    cat_map = defaultdict(list)
    for t in TOPICS:
        if t in ARXIV_CATEGORIES:
            for c in ARXIV_CATEGORIES[t]:
                cat_map[c].append(t)

    for cat in cat_map:
        time.sleep(REQUEST_DELAY)

        papers = fetch_arxiv_papers(cat, MAX_PAPERS)
        cat_papers = []

        for p in papers:
            summary = summarize_paper(p, cache, seen)
            if summary:
                p["summary"] = summary
                cat_papers.append(p)
                all_papers.append(p)

        if cat_papers:
            markdown.append(f"## {cat}\n\n")

            cat_summary = summarize_category(cat, cat_papers)
            if cat_summary:
                markdown.append(cat_summary + "\n\n")

            for p in cat_papers:
                markdown.append(
                    f"### {p['title']}\n\n"
                    f"**Authors:** {p['authors']}\n\n"
                    f"{p['summary']}\n\n"
                    f"[Read paper]({p['url']})\n\n---\n\n"
                )

    save_cache(cache)

    with open(f"summaries/{today}.md", "w", encoding="utf-8") as f:
        f.write("".join(markdown))

    with open(f"summaries/{today}.json", "w", encoding="utf-8") as f:
        json.dump(all_papers, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Processed {len(all_papers)} papers.")


if __name__ == "__main__":
    main()
