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
# Groq Batch Summarization
# =========================

def summarize_category_batch(papers):
    """
    One Groq call per category.
    Returns:
      - category_summary (bullets)
      - paper_summaries {title: bullets}
    """

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    content = ""
    for i, p in enumerate(papers, 1):
        content += f"\nPaper {i}: {p['title']}\nAbstract: {p['abstract']}\n"

    prompt = """
You are preparing a DAILY RESEARCH DIGEST.

TASK:
1. Produce a VERY SHORT category summary (2–3 bullet points).
2. For EACH paper, produce 2 concise bullet points.

FORMAT EXACTLY AS:

CATEGORY SUMMARY:
- bullet
- bullet

PAPER SUMMARIES:
Title: <paper title>
- bullet
- bullet
"""

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "user", "content": prompt + content[:12000]}
        ],
        "temperature": 0.2,
    }

    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    r.raise_for_status()

    text = r.json()["choices"][0]["message"]["content"]

    category_summary = ""
    paper_summaries = {}

    section = None
    current_title = None

    for line in text.splitlines():
        line = line.strip()

        if line.startswith("CATEGORY SUMMARY"):
            section = "category"
            continue
        if line.startswith("PAPER SUMMARIES"):
            section = "papers"
            continue

        if section == "category" and line.startswith("-"):
            category_summary += line + "\n"

        if section == "papers":
            if line.startswith("Title:"):
                current_title = line.replace("Title:", "").strip()
                paper_summaries[current_title] = ""
            elif line.startswith("-") and current_title:
                paper_summaries[current_title] += line + "\n"

    return category_summary.strip(), paper_summaries

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

            papers = fetch_arxiv(category, MAX_PAPERS)
            papers = [p for p in papers if p["title"] not in seen_titles]

            if not papers:
                continue

            for p in papers:
                seen_titles.add(p["title"])

            cat_summary, paper_summaries = summarize_category_batch(papers)

            md.write(f"## {category}\n\n")
            md.write("**Category Summary:**\n")
            md.write(f"{cat_summary}\n\n")

            for p in papers:
                md.write(f"### {p['title']}\n")
                md.write(f"- **Authors:** {p['authors']}\n")
                md.write(f"- **Link:** {p['url']}\n\n")
                md.write(paper_summaries.get(p["title"], "- Summary unavailable\n"))
                md.write("\n")

                p["summary"] = paper_summaries.get(p["title"], "")
                p["category"] = category
                all_results.append(p)

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"✅ Saved summaries for {len(all_results)} papers")
    print(f"📄 Summary file: summaries/{today}.md")

if __name__ == "__main__":
    main()
