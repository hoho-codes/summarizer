# fetch_papers.py
# One-paper-per-call summarization + category reducer
# Groq-safe: bounded tokens, no category-sized prompts

import os
import time
import requests
import datetime
from pathlib import Path

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Hard safety margins
PER_PAPER_SLEEP = 2.2          # seconds between Groq calls
CATEGORY_REDUCER_SLEEP = 2.2
MAX_ABSTRACT_CHARS = 700        # hard truncate

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json",
}


def groq_chat(prompt, max_tokens=200):
    """Single bounded Groq request."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    r = requests.post(GROQ_URL, headers=HEADERS, json=payload, timeout=60)
    if r.status_code == 429:
        raise RuntimeError("Groq TPM exceeded — prompt too large")
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


# ---------------- Summarization ----------------

def summarize_single_paper(paper):
    abstract = (paper.get("abstract") or "")[:MAX_ABSTRACT_CHARS]

    prompt = f"""
Summarize the following paper in 2–3 very short bullet points.
Be technical but concise.

Title: {paper['title']}

Abstract:
{abstract}
"""
    summary = groq_chat(prompt, max_tokens=120)
    time.sleep(PER_PAPER_SLEEP)
    return summary


def summarize_category_reducer(paper_summaries, category):
    joined = "\n".join(f"- {s}" for s in paper_summaries)

    prompt = f"""
Given the following bullet summaries of recent papers in {category},
write ONE short paragraph (3–4 lines) summarizing overall themes.

{joined}
"""
    summary = groq_chat(prompt, max_tokens=180)
    time.sleep(CATEGORY_REDUCER_SLEEP)
    return summary


# ---------------- Main Pipeline ----------------

def main():
    today = datetime.date.today().isoformat()
    outdir = Path("summaries")
    outdir.mkdir(exist_ok=True)
    outfile = outdir / f"{today}.md"

    # Example category map (unchanged fetching logic assumed upstream)
    categories = {
        "quant-ph": "quantum algorithms, quantum information",
        "cond-mat.stat-mech": "statistical physics",
        "physics.flu-dyn": "turbulence",
        "physics.ao-ph": "geophysical turbulence",
        "nlin.CD": "nonlinear sciences",
        "nlin.PS": "nonlinear sciences",
        "nlin.SI": "nonlinear sciences",
    }

    total = 0

    with open(outfile, "w") as md:
        md.write(f"# Daily arXiv Paper Summaries ({today})\n\n")

        for cat, desc in categories.items():
            print(f"Fetching {cat} ({desc})")

            papers = fetch_papers_for_category(cat)  # EXISTING FUNCTION
            if not papers:
                continue

            md.write(f"## {cat} — {desc}\n\n")

            paper_summaries = []
            for p in papers:
                try:
                    s = summarize_single_paper(p)
                except Exception as e:
                    print(f"  Skipping paper due to error: {e}")
                    continue

                paper_summaries.append(s)
                md.write(f"**{p['title']}**\n{s}\n\n")
                total += 1

            if paper_summaries:
                cat_summary = summarize_category_reducer(paper_summaries, cat)
                md.write(f"**Category summary:** {cat_summary}\n\n")

    print(f"✅ Saved summaries for {total} papers")
    print(f"📄 Summary file: {outfile}")


# ---------------- Entry ----------------

if __name__ == "__main__":
    main()
