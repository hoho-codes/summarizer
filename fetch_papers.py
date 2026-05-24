import os
import json
import feedparser
import requests
import time
import re
from datetime import datetime, timezone
from pathlib import Path
from requests.exceptions import RequestException

# =========================
# Configuration
# =========================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GROQ_API_KEY and not GEMINI_API_KEY:
    raise RuntimeError(
        "At least one API key (GROQ_API_KEY or GEMINI_API_KEY) must be set."
    )

TOPICS = [
    t.strip().lower()
    for t in os.environ.get("TOPICS", "quantum computing").split(",")
    if t.strip()
]

MAX_PAPERS = 10
REQUEST_DELAY = 5
MAX_ABSTRACT_CHARS = 1500

# ---- Models ----
PAPER_MODEL_GROQ = "llama-3.1-8b-instant"
CATEGORY_MODEL_GROQ = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-1.5-flash"

# ---- LLM safety ----
MIN_GROQ_INTERVAL = 12
MAX_GROQ_CALLS = 25
GROQ_CALL_COUNT = 0

MIN_GEMINI_INTERVAL = 4

LAST_GROQ_CALL = time.time() - MIN_GROQ_INTERVAL
LAST_GEMINI_CALL = time.time() - MIN_GEMINI_INTERVAL

# =========================
# arXiv categories
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
# Cache
# =========================

CACHE_PATH = Path("summaries/cache.json")

def load_cache():
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("  [!] Cache corrupted — starting fresh")
    return {}

def save_cache(cache):
    CACHE_PATH.parent.mkdir(exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def arxiv_id_from_url(url):
    m = re.search(r"arxiv\.org/(abs|pdf)/([^/]+)", url)
    if m:
        return m.group(2).replace(".pdf", "")
    return url.rstrip("/").split("/")[-1]

# =========================
# arXiv Fetch
# =========================

def fetch_arxiv_papers(category, max_results):
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"cat:{category}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        feed = feedparser.parse(r.content)
    except Exception as e:
        print(f"  [!] arXiv API failed: {e}")
        return []

    papers = []
    for e in feed.entries:
        papers.append({
            "title": getattr(e, "title", "").replace("\n", " ").strip(),
            "authors": ", ".join(a.name for a in getattr(e, "authors", [])) or "Unknown",
            "abstract": getattr(e, "summary", "").replace("\n", " ").strip(),
            "url": getattr(e, "link", ""),
            "published": getattr(e, "published", "Unknown"),
        })
    return papers

# =========================
# LLM Calls
# =========================

def groq_call(prompt, model, max_tokens=150):
    global LAST_GROQ_CALL, GROQ_CALL_COUNT

    if not GROQ_API_KEY or GROQ_CALL_COUNT >= MAX_GROQ_CALLS:
        return None

    wait = MIN_GROQ_INTERVAL - (time.time() - LAST_GROQ_CALL)
    if wait > 0:
        time.sleep(wait)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        LAST_GROQ_CALL = time.time()
        GROQ_CALL_COUNT += 1
        return r.json()["choices"][0]["message"]["content"].strip()
    except RequestException as e:
        print(f"  [!] Groq API call failed: {e}")
        return None

def gemini_call(prompt, max_tokens=150):
    global LAST_GEMINI_CALL

    if not GEMINI_API_KEY:
        return None

    wait = MIN_GEMINI_INTERVAL - (time.time() - LAST_GEMINI_CALL)
    if wait > 0:
        time.sleep(wait)

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens},
    }

    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        LAST_GEMINI_CALL = time.time()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"  [!] Gemini API call failed: {e}")
        return None

def get_llm_summary(prompt, model, force_groq=False):
    if force_groq:
        return groq_call(prompt, model) or "- Summary unavailable."
    return groq_call(prompt, model) or gemini_call(prompt) or "- Summary unavailable."

# =========================
# Summarization
# =========================

def normalize_bullets(text, max_lines):
    lines = []
    for l in text.splitlines():
        l = re.sub(r"^(\*|-|\d+\.)\s*", "", l.strip())
        if l and not re.match(r"^(here|sure|summary)", l.lower()):
            lines.append(f"- {l}")
        if len(lines) >= max_lines:
            break
    return "\n".join(lines) or "- Summary unavailable."

def summarize_paper(paper, cache):
    if not paper["abstract"]:
        return "- Abstract missing; summary unavailable."

    pid = arxiv_id_from_url(paper["url"])

    if pid in cache:
        print("    (Loaded from cache)")
        return cache[pid]["summary"]

    prompt = (
        "Summarize this paper in exactly 3 concise bullet points "
        "(max 12 words each). Output bullets only.\n\n"
        f"Title: {paper['title']}\n"
        f"Abstract: {paper['abstract'][:MAX_ABSTRACT_CHARS]}"
    )

    summary = normalize_bullets(
        get_llm_summary(prompt, PAPER_MODEL_GROQ), 3
    )

    if summary != "- Summary unavailable.":
        cache[pid] = {
            "title": paper["title"],
            "summary": summary,
            "model": PAPER_MODEL_GROQ,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }
        save_cache(cache)

    return summary

def summarize_category(category, papers):
    valid = [p for p in papers if p["summary"] != "- Summary unavailable."]
    if len(valid) < 2:
        return ""

    titles = "\n".join(f"- {p['title']}" for p in valid[:8])
    prompt = (
        f"Summarize the main themes of these {category} papers "
        "in 2–3 bullet points. Output bullets only.\n\n" + titles
    )

    return normalize_bullets(
        get_llm_summary(prompt, CATEGORY_MODEL_GROQ, force_groq=True), 3
    )

# =========================
# Main
# =========================

def main():
    print("Starting daily paper fetch...")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = Path("summaries")
    out_dir.mkdir(exist_ok=True)

    cache = load_cache()
    all_papers = []
    seen = set()

    markdown = [
        f"# Research Paper Summaries — {today}\n\n",
        f"**Topics:** {', '.join(TOPICS)}\n\n",
    ]

    category_map = {}
    for t in TOPICS:
        if t in ARXIV_CATEGORIES:
            for c in ARXIV_CATEGORIES[t]:
                category_map.setdefault(c, []).append(t)

    for cat in category_map:
        papers = fetch_arxiv_papers(cat, MAX_PAPERS)
        time.sleep(REQUEST_DELAY)

        cat_papers_today = []

        for p in papers:
            pid = arxiv_id_from_url(p["url"])
            if pid in seen:
                continue
            seen.add(pid)

            print(f"  Processing: {p['title'][:60]}...")
            p["summary"] = summarize_paper(p, cache)
            p["category"] = cat

            cat_papers_today.append(p)
            all_papers.append(p)

        if not cat_papers_today:
            continue

        markdown.append(f"## {cat}\n\n")
        cat_summary = summarize_category(cat, cat_papers_today)
        if cat_summary:
            markdown.append(cat_summary + "\n\n")

        for p in cat_papers_today:
            markdown.extend([
                f"### {p['title']}\n\n",
                f"**Authors:** {p['authors']}\n\n",
                f"{p['summary']}\n\n",
                f"[Read paper]({p['url']})\n\n---\n\n",
            ])

    save_cache(cache)

    with open(out_dir / f"{today}.md", "w", encoding="utf-8") as f:
        f.write("".join(markdown))

    with open(out_dir / f"{today}.json", "w", encoding="utf-8") as f:
        json.dump(all_papers, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done. Processed {len(all_papers)} papers.")

if __name__ == "__main__":
    main()
