import os
import json
import feedparser
import requests
import time
from datetime import datetime
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

TOPICS = os.environ.get(
    "TOPICS",
    "quantum algorithms,quantum information,statistical physics,"
    "turbulence,geophysical turbulence,nonlinear sciences"
).split(",")

MAX_PAPERS = 10
REQUEST_DELAY = 5          # arXiv politeness delay
GROQ_DELAY = 3.0           # Groq hard throttle
BATCH_SIZE = 3             # papers per Groq call

LAST_GROQ_CALL = 0.0

ARXIV_CATEGORIES = {
    "quantum algorithms": "quant-ph",
    "quantum information": "quant-ph",
    "quantum computing": "quant-ph",
    "statistical physics": "cond-mat.stat-mech",
    "turbulence": "physics.flu-dyn",
    "geophysical turbulence": "physics.ao-ph",
    "fluid dynamics": "physics.flu-dyn",
    "atmospheric physics": "physics.ao-ph",
    "nonlinear sciences": ["nlin.CD", "nlin.PS", "nlin.SI"],
    "chaos": "nlin.CD",
    "pattern formation": "nlin.PS",
}

# ============================================================
# Rate limiting
# ============================================================

def groq_rate_limit():
    global LAST_GROQ_CALL
    elapsed = time.time() - LAST_GROQ_CALL
    if elapsed < GROQ_DELAY:
        time.sleep(GROQ_DELAY - elapsed)
    LAST_GROQ_CALL = time.time()

# ============================================================
# arXiv fetching (RSS → API fallback)
# ============================================================

def fetch_arxiv_papers_api(category, max_results):
    base_url = "http://export.arxiv.org/api/query?"
    params = {
        "search_query": f"cat:{category}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    url = base_url + "&".join(f"{k}={v}" for k, v in params.items())
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    feed = feedparser.parse(response.content)
    papers = []

    for entry in feed.entries:
        papers.append({
            "title": entry.title.strip(),
            "authors": ", ".join(a.name for a in getattr(entry, "authors", [])) or "Unknown",
            "abstract": entry.summary.strip(),
            "url": entry.link,
            "published": getattr(entry, "published", "Unknown"),
        })

    return papers

def fetch_arxiv_papers(category, max_results):
    rss_url = f"http://export.arxiv.org/rss/{category}"
    feed = feedparser.parse(rss_url)

    if not feed.entries:
        print("  RSS empty, falling back to arXiv API")
        time.sleep(REQUEST_DELAY)
        return fetch_arxiv_papers_api(category, max_results)

    papers = []
    for entry in feed.entries[:max_results]:
        papers.append({
            "title": entry.title.strip(),
            "authors": entry.get("author", "Unknown"),
            "abstract": entry.summary.strip(),
            "url": entry.link,
            "published": entry.get("published", "Unknown"),
        })

    return papers

# ============================================================
# Groq summarization
# ============================================================

def summarize_papers_with_groq(papers):
    if not GROQ_API_KEY or not papers:
        return ["Summary unavailable."] * len(papers)

    groq_rate_limit()

    blocks = []
    for i, p in enumerate(papers, 1):
        abstract = p["abstract"][:2000].replace("\n", " ")
        blocks.append(f"{i}. Title: {p['title']}\nAbstract: {abstract}")

    prompt = (
        "Summarize each research paper in 2–3 sentences.\n"
        "Return a numbered list matching the paper numbers.\n\n"
        + "\n\n".join(blocks)
    )

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 600,
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    response.raise_for_status()

    text = response.json()["choices"][0]["message"]["content"]

    summaries = []
    for line in text.split("\n"):
        if line.strip() and line.lstrip()[0].isdigit():
            summaries.append(line.split(".", 1)[-1].strip())

    while len(summaries) < len(papers):
        summaries.append("Summary unavailable.")

    return summaries[:len(papers)]

def summarize_category(category, papers):
    if not papers:
        return "No papers available for this category today."

    groq_rate_limit()

    bullets = [f"- {p['title']}: {p['summary']}" for p in papers]

    prompt = (
        "Write one concise paragraph summarizing the following research papers "
        "as a group. Focus on shared themes, methods, and scientific direction.\n\n"
        + "\n".join(bullets)
    )

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 200,
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# ============================================================
# Main
# ============================================================

def main():
    print("Starting daily paper fetch...")

    seen_titles = set()
    all_papers = []
    category_summaries = {}

    unique_categories = {}
    for topic in TOPICS:
        topic = topic.strip().lower()
        cats = ARXIV_CATEGORIES.get(topic, ["cs.AI"])
        if isinstance(cats, str):
            cats = [cats]
        for cat in cats:
            unique_categories.setdefault(cat, []).append(topic)

    for category, topics in unique_categories.items():
        print(f"Fetching {category} ({', '.join(topics)})")

        papers = fetch_arxiv_papers(category, MAX_PAPERS)
        print(f"  Found {len(papers)} papers")
        time.sleep(REQUEST_DELAY)

        category_papers = []
        batch = []

        for paper in papers:
            if paper["title"] in seen_titles:
                continue
            seen_titles.add(paper["title"])

            batch.append(paper)

            if len(batch) == BATCH_SIZE:
                summaries = summarize_papers_with_groq(batch)
                for p, s in zip(batch, summaries):
                    p["summary"] = s
                    p["topic"] = topics[0]
                    category_papers.append(p)
                    all_papers.append(p)
                batch = []

        if batch:
            summaries = summarize_papers_with_groq(batch)
            for p, s in zip(batch, summaries):
                p["summary"] = s
                p["topic"] = topics[0]
                category_papers.append(p)
                all_papers.append(p)

        category_summaries[category] = summarize_category(category, category_papers)

    today = datetime.utcnow().strftime("%Y-%m-%d")
    output_dir = Path("summaries")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / f"{today}.json", "w", encoding="utf-8") as f:
        json.dump(
            {"date": today, "categories": category_summaries, "papers": all_papers},
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(output_dir / f"{today}.md", "w", encoding="utf-8") as f:
        f.write(f"# Research Paper Summaries — {today}\n\n")
        for category, summary in category_summaries.items():
            f.write(f"## {category}\n\n{summary}\n\n")
            for p in [x for x in all_papers if x["topic"] == unique_categories[category][0]]:
                f.write(f"### {p['title']}\n\n")
                f.write(f"**Authors:** {p['authors']}\n\n")
                f.write(f"**Summary:** {p['summary']}\n\n")
                f.write(f"[Link]({p['url']})\n\n")
            f.write("---\n\n")

    print(f"\n✅ Saved summaries for {len(all_papers)} papers")

if __name__ == "__main__":
    main()
