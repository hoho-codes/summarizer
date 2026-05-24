import os
import time
import json
import feedparser
import requests
from datetime import datetime
from requests.exceptions import ReadTimeout, RequestException

# =========================
# Configuration
# =========================

MAX_PAPERS = 10
REQUEST_DELAY = 5
GROQ_DELAY = 8
GROQ_MODEL = "llama-3.3-70b-versatile"

OUTPUT_DIR = "summaries"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOPICS = [t.strip().lower() for t in os.getenv("TOPICS", "").split(",") if t.strip()]

ARXIV_CATEGORIES = {
    "quantum algorithms": "quant-ph",
    "quantum information": "quant-ph",
    "statistical physics": "cond-mat.stat-mech",
    "turbulence": "physics.flu-dyn",
    "geophysical turbulence": "physics.ao-ph",
    "nonlinear sciences": ["nlin.CD", "nlin.PS", "nlin.SI"],
}

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY")

# =========================
# arXiv Fetching
# =========================

def fetch_arxiv_api(category, max_results, retries=3):
    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query=cat:{category}"
        f"&start=0&max_results={max_results}"
        "&sortBy=submittedDate&sortOrder=descending"
    )

    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            feed = feedparser.parse(r.content)

            papers = []
            for e in feed.entries:
                papers.append({
                    "title": e.title.strip(),
                    "authors": ", ".join(a.name for a in getattr(e, "authors", [])) or "Unknown",
                    "abstract": e.summary.strip(),
                    "url": e.link,
                    "published": getattr(e, "published", "Unknown"),
                })
            return papers

        except (ReadTimeout, RequestException):
            print(f"  API timeout {attempt}/{retries} for {category}")
            time.sleep(5 * attempt)

    print(f"  ❌ API failed for {category}")
    return []


def fetch_arxiv(category, max_results):
    rss_url = f"http://export.arxiv.org/rss/{category}"
    feed = feedparser.parse(rss_url)

    if not feed.entries:
        print("  RSS empty, using API")
        time.sleep(REQUEST_DELAY)
        return fetch_arxiv_api(category, max_results)

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
# Groq Summarization
# =========================

def summarize_with_groq(text, short=False):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    prompt = (
        "Summarize the following set of research abstracts in one coherent paragraph:\n\n"
        if short else
        "Summarize this research abstract in 3 concise sentences:\n\n"
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt + text[:2000]}],
        "temperature": 0.3,
    }

    for _ in range(3):
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )

        if r.status_code == 429:
            print("  Groq rate limit — sleeping")
            time.sleep(15)
            continue

        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    return "Summary unavailable due to rate limits."

# =========================
# Main
# =========================

def main():
    print("Starting daily paper fetch...")

    today = datetime.utcnow().strftime("%Y-%m-%d")
    md_path = os.path.join(OUTPUT_DIR, f"{today}.md")
    json_path = os.path.join(OUTPUT_DIR, f"{today}.json")

    all_results = []
    seen_titles = set()

    category_map = {}
    for topic in TOPICS:
        cats = ARXIV_CATEGORIES.get(topic, [])
        if isinstance(cats, str):
            cats = [cats]
        for c in cats:
            category_map.setdefault(c, []).append(topic)

    with open(md_path, "w") as md:
        md.write(f"# Daily Research Papers — {today}\n\n")

        for category, topics in category_map.items():
            print(f"Fetching {category} ({', '.join(topics)})")

            try:
                papers = fetch_arxiv(category, MAX_PAPERS)
            except Exception as e:
                print(f"  ❌ Skipping {category}: {e}")
                continue

            if not papers:
                continue

            # ---- CATEGORY HEADER ----
            md.write(f"## {category}\n\n")

            # ---- CATEGORY SUMMARY (TOP) ----
            abstracts = []
            for p in papers:
                if p["title"] not in seen_titles:
                    abstracts.append(p["abstract"])

            if abstracts:
                combined = "\n\n".join(abstracts)
                cat_summary = summarize_with_groq(combined, short=True)
                time.sleep(GROQ_DELAY)
                md.write(f"**Category Summary:** {cat_summary}\n\n")

            # ---- INDIVIDUAL PAPERS ----
            for p in papers:
                if p["title"] in seen_titles:
                    continue
                seen_titles.add(p["title"])

                summary = summarize_with_groq(p["abstract"])
                time.sleep(GROQ_DELAY)

                md.write(f"### {p['title']}\n")
                md.write(f"- **Authors:** {p['authors']}\n")
                md.write(f"- **Link:** {p['url']}\n\n")
                md.write(f"{summary}\n\n")

                p["summary"] = summary
                p["category"] = category
                all_results.append(p)

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"✅ Saved summaries for {len(all_results)} papers")

if __name__ == "__main__":
    main()
