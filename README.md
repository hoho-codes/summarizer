# 📚 Automated Daily Research Paper Summarizer

A zero-overhead, production-grade automated pipeline that tracks, fetches, and summarizes academic literature from arXiv using state-of-the-art Large Language Models (**Groq / Llama 3** & **Google Gemini**). 

Running entirely on **GitHub Actions**, this script updates daily, handles API rate-limits defensively, skips already-processed papers via local disk caching, and outputs elegant, human-readable Markdown digests.

---

## 🚀 Quick Start (3-Minute Setup)

Get your automated daily newsletter running in three simple steps:

### 1. Add API Keys to GitHub Secrets
The script requires at least one API key to function. 
1. Go to your GitHub Repository **Settings > Secrets and variables > Actions**.
2. Click **New repository secret** and add at least one of these:
   * `GROQ_API_KEY` (Recommended primary engine for blistering speed)
   * `GEMINI_API_KEY` (Excellent failover backup layer)

### 2. Configure Your Target Topics
Open `.github/workflows/daily_summary.yml` and modify the `TOPICS` string to match your research interests. Make sure they match the supported keys inside `fetch_papers.py`:

```yaml
env:
  GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
  GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
  TOPICS: "quantum algorithms, quantum computing"
