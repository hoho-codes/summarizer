import os
import json
import feedparser
import requests
import time
import re
import html  # Added for robust API text escaping
from datetime import datetime, timezone, timedelta
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
CACHE_RETENTION_DAYS = 90

PAPER_MODEL_GROQ = "llama-3.1-8b-instant"
CATEGORY_MODEL_GROQ = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-1.5-flash"

session = requests.Session()
session.headers.update({
    "User-Agent": "ResearchPaperSummarizer/1.0 (contact: github-bot)"
})

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
    if not CACHE_PATH.exists():
        return {}

    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)

        cutoff = datetime.now(timezone.utc) - timedelta(days=CACHE_RETENTION_DAYS)
        pruned = {}

        for k, v in cache.items():
            try:
                ts = datetime.strptime(v["timestamp"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if ts >= cutoff:
                    pruned[k] = v
            except Exception:
                pass

        return pruned

    except json.JSONDecodeError:
        print("[!] Cache corrupted, resetting.")
        return {}

def save_cache(cache):
    CACHE_PATH.parent.mkdir(exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(CACHE_PATH)
    except Exception as e:
        print(f"[!] Critical: Failed to save cache safely: {e}")

def arxiv_id_from_url(url):
    if not url:
        return None
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([^?#]+)", url)
    if not m:
        return None
    return re.sub(r"v\d+$", "", m.group(1).replace(".pdf", "").strip())

# =========================
# Fetch Engine
# =========================

def clean_feed_text(text):
    if not text:
        return ""
    # Unescape HTML entities (e.g., &lt; to <) and normalize spacing
    text = html.unescape(text)
    return re.sub(r"[ \t\r\f\v]+", " ", text.replace("\n", " ")).strip()

def normalize_date(time_struct=None):
    if time_struct:
        try:
            return datetime(*time_struct[:6], tzinfo=timezone.utc).strftime("%Y-%m-%d")
        except Exception:
            pass
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def fetch_arxiv_papers(category, max_results):
    rss_url = f"https://rss.arxiv.org/rss/{category}"

    for attempt in range(3):
        try:
            r = session.get(rss_url, timeout=15)
            r.raise_for_status()
            feed = feedparser.parse(r.content)

            if feed.entries:
                return [{
                    "title": clean_feed_text(e.get("title")),
                    "authors": clean_feed_text(e.get("author", "Unknown")),
                    "abstract": clean_feed_text(e.get("summary")),
                    "url": e.get("link"),
                    "published": normalize_date(getattr(e, "published_parsed", None)),
                } for e in feed.entries[:max_results]]
        except Exception:
            time.sleep(2 ** (attempt + 1))

    time.sleep(REQUEST_DELAY)

    api_url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": f"cat:{category}",
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    for attempt in range(3):
        try:
            r = session.get(api_url, params=params, timeout=20)
            r.raise_for_status()
            feed = feedparser.parse(r.content)

            return [{
                "title": clean_feed_text(e.get("title")),
                "authors": ", ".join(a.name for a in getattr(e, "authors", [])) or "Unknown",
                "abstract": clean_feed_text(e.get("summary")),
                "url": e.get("link"),
                "published": normalize_date(getattr(e, "published_parsed", None)),
            } for e in feed.entries]

        except RequestException:
            time.sleep(2 ** (attempt + 1))

    return []

# =========================
# LLM Helpers
# =========================

def format_bullets(text, max_lines=3, min_lines=1):
    if not text:
        return None

    lines = []
    for line in text.splitlines():
        line = re.sub(r'^[\*\-\•\s]+|\d+[\.\)]', '', line).strip()
        if line:
            lines.append(f"- {line}")

    if len(lines) < min_lines:
        return None

    return "\n".join(lines[:max_lines])

def get_llm_summary(prompt, model, is_category_wide=False):
    if GROQ_API_KEY:
        # Throttling delay matches safety boundaries perfectly
        time.sleep(3.0 if is_category_wide else 2.2)
        for attempt in range(3):
            try:
                r = session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2,
                        "max_tokens": 300,
                    },
                    timeout=20,
                )
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"].strip()
                elif r.status_code == 429:
                    time.sleep(6 * (attempt + 1))
            except Exception:
                time.sleep(2)

    if GEMINI_API_KEY:
        for _ in range(3):
            try:
                r = session.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}",
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                    timeout=20,
                )
                if r.status_code == 200:
                    data = r.json()
                    if data.get("candidates") and "content" in data["candidates"][0]:
                        parts = data["candidates"][0]["content"].get("parts", [])
                        if parts and "text" in parts[0]:
                            return parts[0]["text"].strip()

                    print("  [!] Warning: Gemini payload tripped safety constraints. Falling back safely.")
                    return "Summary unavailable due to automated API safety flags."
            except Exception:
                time.sleep(2)

    return None

# =========================
# Processing
# =========================

def summarize_paper(paper, cache, seen):
    pid = arxiv_id_from_url(paper["url"])
    if not pid or pid in seen:
        return None
    seen.add(pid)

    if pid in cache:
        return cache[pid]["summary"]

    prompt = (
        "Provide up to 3 short bullet points summarizing the paper below. "
        "Keep each bullet highly concise.\n\n"
        f"Title: {paper['title']}\n"
        f"Abstract: {paper['abstract'][:MAX_ABSTRACT_CHARS]}"
    )

    raw = get_llm_summary(prompt, PAPER_MODEL_GROQ, is_category_wide=False)
    summary = format_bullets(raw)

    # Optimization Fix: Fallback gracefully to raw un-bulleted output if formatting regex fails
    if not summary:
        if raw:
            summary = f"- {raw}"
        else:
            return None

    cache[pid] = {
        "title": paper["title"],
        "summary": summary,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }

    return summary

def summarize_category(category, papers):
    titles = "\n".join(f"- {p['title']}" for p in papers[:CATEGORY_TOP_N])
    prompt = f"Summarize major themes or core concepts shared among these {category} papers:\n\n{titles}"
    
    raw = get_llm_summary(prompt, CATEGORY_MODEL_GROQ, is_category_wide=True)
    summary = format_bullets(raw)
    
    if not summary and raw:
        return f"- {raw}"
    return summary or ""

# =========================
# Main
# =========================

def main():
    Path("summaries").mkdir(exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    cache = load_cache()
    seen = set()
    all_papers = []
    newly_summarized_count = 0

    markdown = [
        f"# Research Paper Summaries — {today}\n\n",
        f"**Topics:** {', '.join(TOPICS)}\n\n---\n\n"
    ]

    cat_map = defaultdict(list)
    for t in TOPICS:
        for c in ARXIV_CATEGORIES.get(t, []):
            cat_map[c].append(t)

    total_categories = len(cat_map)

    for idx, cat in enumerate(cat_map, 1):
        print(f"[{idx}/{total_categories}] Fetching papers for category: {cat}...")
        papers = fetch_arxiv_papers(cat, MAX_PAPERS)
        print(f"  -> Found {len(papers)} papers in feed.")

        cat_papers = []

        for p in papers:
            pid = arxiv_id_from_url(p["url"])
            is_cached = pid and pid in cache

            summary = summarize_paper(p, cache, seen)
            if summary:
                p["summary"] = summary
                cat_papers.append(p)
                all_papers.append(p)

                if not is_cached:
                    newly_summarized_count += 1
                    if newly_summarized_count % 5 == 0:
                        save_cache(cache)

        if not cat_papers:
            continue

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

    # Definitive execution barrier cache save
    save_cache(cache)

    with open(f"summaries/{today}.md", "w", encoding="utf-8") as f:
        f.write("".join(markdown))

    with open(f"summaries/{today}.json", "w", encoding="utf-8") as f:
        json.dump(all_papers, f, indent=2, ensure_ascii=False)

    print(f"\n[+] Complete! Successfully compiled {len(all_papers)} papers across {total_categories} categories.")

if __name__ == "__main__":
    main()
