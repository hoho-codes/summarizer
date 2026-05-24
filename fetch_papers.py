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

TOPICS = [t.strip().lower() for t in os.environ.get(
    "TOPICS", "quantum computing").split(",") if t.strip()]

MAX_PAPERS = 10
REQUEST_DELAY = 5
MAX_ABSTRACT_CHARS = 1500

# ---- LLM safety ----
GROQ_MODEL = "llama-3.3-70b-versatile"
MIN_GROQ_INTERVAL = 10
LAST_GROQ_CALL = 0

GEMINI_MODEL = "gemini-1.5-flash"
MIN_GEMINI_INTERVAL = 4
LAST_GEMINI_CALL = 0

if not GROQ_API_KEY and not GEMINI_API_KEY:
    raise RuntimeError("At least one of GROQ_API_KEY or GEMINI_API_KEY must be set.")

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
    feed = feedparser.parse(f"http://export.arxiv.org/rss/{category}")
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
# LLM calls
# =========================

def groq_call(prompt, max_tokens, temperature):
    global LAST_GROQ_CALL
    if not GROQ_API_KEY:
        return None

    wait = MIN_GROQ_INTERVAL - (time.time() - LAST_GROQ_CALL)
    if wait > 0:
        time.sleep(wait)

    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for _ in range(3):
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

            if r.status_code == 429:
                time.sleep(20)
                continue

            r.raise_for_status()
            LAST_GROQ_CALL = time.time()
            text = r.json()["choices"][0]["message"]["content"].strip()
            return text or None

        except RequestException:
            time.sleep(10)

    return None

def gemini_call(prompt, max_tokens, temperature):
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
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }

    for _ in range(2):
        try:
            r = requests.post(url, json=payload, timeout=30)
            if r.status_code == 429:
                time.sleep(10)
                continue

            r.raise_for_status()
            LAST_GEMINI_CALL = time.time()
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            return text or None

        except Exception:
            time.sleep(5)

    return None

def get_llm_summary(prompt, max_tokens=120, temperature=0.2):
    return (
        groq_call(prompt, max_tokens, temperature)
        or gemini_call(prompt, max_tokens, temperature)
        or "- Summary unavailable."
    )

# =========================
# Normalization
# =========================

def normalize_bullets(text, max_lines=3):
    lines = []
    for l in text.splitlines():
        l = l.strip()
        if not l:
            continue
        if re.match(r"^(here|sure|summary|these)\b", l.lower()):
            continue
        l = re.sub(r"^(\*|-|\d+\.)\s*", "", l)
        lines.append(f"- {l}")
        if len(lines) >= max_lines:
            break
    return "\n".join(lines) or "- Summary unavailable."

def summarize_paper(title, abstract):
    abstract = abstract[:MAX_ABSTRACT_CHARS].replace("\n", " ").strip()
    prompt = (
        "Summarize this paper as 3 concise bullet points "
        "(max 12 words each). Output bullets only:\n\n"
        f"Title: {title}\nAbstract: {abstract}"
    )
    return normalize_bullets(get_llm_summary(prompt))

def summarize_category(category, papers):
    if len(papers) < 2:
        return ""  # skip reducer for single-paper categories

    titles = "\n".join(f"- {p['title']}" for p in papers[:8])
    prompt = (
        f"Summarize these {category} papers in 2–3 bullet points. "
        "Output bullets only:\n\n" + titles
    )
    return normalize_bullets(get_llm_summary(prompt), max_lines=3)

# =========================
# Main
# =========================

def main():
    print("Starting daily paper fetch...")
    print(f"Groq: {'ON' if GROQ_API_KEY else 'OFF'} | Gemini: {'ON' if GEMINI_API_KEY else 'OFF'}")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = Path("summaries")
    out_dir.mkdir(exist_ok=True)

    seen = set()
    markdown = [
        f"# Research Paper Summaries — {today}\n\n",
        f"**Topics:** {', '.join(TOPICS)}\n\n"
    ]
    all_papers = []

    category_map = {}
    for topic in TOPICS:
        for cat in ARXIV_CATEGORIES.get(topic, ["quant-ph"]):
            category_map.setdefault(cat, []).append(topic)

    for cat, topics in category_map.items():
        print(f"\nFetching {cat} ({', '.join(topics)})...")
        papers = fetch_arxiv_papers(cat, MAX_PAPERS)
        time.sleep(REQUEST_DELAY)

        papers = [p for p in papers if p["title"] not in seen]

        if not papers:
            print("  No new papers found in this category.")
            continue

        for p in papers:
            seen.add(p["title"])
            print(f"  Summarizing: {p['title'][:60]}...")
            p["summary"] = summarize_paper(p["title"], p["abstract"])
            p["category"] = cat
            all_papers.append(p)

        cat_summary = summarize_category(cat, papers)
        markdown.append(f"## {cat}\n\n")
        if cat_summary:
            markdown.append(cat_summary + "\n\n")

        for p in papers:
            markdown.extend([
                f"### {p['title']}\n\n",
                f"**Authors:** {p['authors']}\n\n",
                f"{p['summary']}\n\n",
                f"[Read paper]({p['url']})\n\n---\n\n",
            ])

    with open(out_dir / f"{today}.json", "w", encoding="utf-8") as f:
        json.dump(all_papers, f, indent=2, ensure_ascii=False)

    with open(out_dir / f"{today}.md", "w", encoding="utf-8") as f:
        f.write("".join(markdown))

    print(f"\n✅ Saved {len(all_papers)} papers")

if __name__ == "__main__":
    main()
