# 📚 Automated Daily Research Paper Summarizer

This is a completely vibe coded project developed in my spare time - using the free tiers of Claude, ChatGPT and Gemini. The only human generated text are these two lines 😁.

---

## 📋 Overview

An automated pipeline that tracks, fetches, and summarizes academic literature from arXiv using state-of-the-art Large Language Models (**Groq / Llama 3** & **Google Gemini**). 

Running entirely on **GitHub Actions**, this script updates daily, handles API rate-limits defensively, skips already-processed papers via local disk caching, and outputs human-readable Markdown digests.

---

## 🚀 Quick Start (3-Minute Setup)

Get your automated daily newsletter running in four simple steps:

### 1. Fork this Repository
1. Scroll to the very top of this GitHub page.
2. Click the **Fork** button in the upper-right corner.
3. Select your personal GitHub account as the owner and click **Create fork**.
4. Once the redirection finishes, you are officially working out of your own personal copy!

### 2. Add API Keys to GitHub Secrets
The script requires at least one API key to function. 

1. Go to your GitHub Repository **Settings > Secrets and variables > Actions**.
2. Click **New repository secret** and add at least one of these:
   * `GROQ_API_KEY` (Recommended primary engine for blistering speed)
   * `GEMINI_API_KEY` (Excellent failover backup layer)

### 3. Configure Your Target Topics
Open `.github/workflows/daily_summary.yml` and modify the `TOPICS` string to match your research interests. Make sure they match the supported keys inside `fetch_papers.py`:

```yaml
  GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
  GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
  TOPICS: "quantum algorithms, quantum computing"
```

⚠️ Public Repo Protection: By default, the schedule block inside the workflow file is commented out with # so it does not run automatically on forks or public copies. To enable the daily automatic execution at 00:30 UTC, simply uncomment those lines:

```yaml
  on:
    schedule:
      - cron: '30 0 * * *'
    workflow_dispatch:
```

### 4. Run Your First Test Manually
Whether the automatic schedule is commented out or not, you can always kick off a run instantly via the GitHub UI:
1. Navigate to the Actions tab at the top of your GitHub repository page.
2. Select Daily Research Paper Summary from the left-hand sidebar.
3. Click the Run workflow dropdown menu on the right and hit the green button.

```plaintext
[Actions] ➔ [Daily Research Paper Summary] ➔ [Run workflow ▼] ➔ [Run workflow]
```
Once completed, a fresh automated summary markdown file and raw JSON catalog will appear inside your repository's `/summaries` directory!

## 🛠️ Repository Architecture
To ensure the automated workflow functions correctly, structure your repository exactly like this:

```Plaintext
├── .github/
│   └── workflows/
│       └── daily_summary.yml      # Your GitHub Actions Workflow file
├── summaries/
│   ├── cache.json                 # Auto-generated/updated rolling paper cache
│   ├── YYYY-MM-DD.md              # Daily human-readable Markdown report
│   └── YYYY-MM-DD.json            # Structured raw metadata backup
├── fetch_papers.py                # Main Python execution engine
└── README.md                      # Documentation
```
