import os
import json
import feedparser
import requests
from datetime import datetime
from pathlib import Path

# Configuration
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
TOPICS = os.environ.get('TOPICS', 'machine learning').split(',')
MAX_PAPERS = 10  # Adjust based on how many papers you want daily

# arXiv category mappings (add more as needed)
ARXIV_CATEGORIES = {
    'quantum algorithms': 'quant-ph',
    'quantum information': 'quant-ph',
    'quantum computing': 'quant-ph',
    'statistical physics': 'cond-mat.stat-mech',
    'turbulence': 'physics.flu-dyn',
    'geophysical turbulence': 'physics.ao-ph',
    'fluid dynamics': 'physics.flu-dyn',
    'atmospheric physics': 'physics.ao-ph',
    'machine learning': 'cs.LG',
    'artificial intelligence': 'cs.AI',
    'computer vision': 'cs.CV',
    'natural language processing': 'cs.CL',
    'robotics': 'cs.RO',
    'deep learning': 'cs.LG',
    'neural networks': 'cs.NE',
}

def fetch_arxiv_papers(category, max_results=10):
    """Fetch recent papers from arXiv RSS feed"""
    url = f'http://export.arxiv.org/rss/{category}'
    feed = feedparser.parse(url)
    
    papers = []
    for entry in feed.entries[:max_results]:
        papers.append({
            'title': entry.title,
            'authors': entry.get('author', 'Unknown'),
            'abstract': entry.summary,
            'url': entry.link,
            'published': entry.get('published', 'Unknown date')
        })
    return papers

def summarize_with_groq(text, title):
    """Summarize paper abstract using Groq API"""
    if not GROQ_API_KEY:
        return "Error: GROQ_API_KEY not set"
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""Summarize this research paper in 2-3 sentences. Focus on: 
1) What problem it solves
2) The key approach/method
3) Main results or contributions

Title: {title}
Abstract: {text}"""
    
    payload = {
        "model": "llama-3.1-70b-versatile",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 200
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error summarizing: {str(e)}"

def main():
    print("Starting daily paper fetch and summarization...")
    
    all_papers = []
    
    # Fetch papers for each topic
    for topic in TOPICS:
        topic = topic.strip().lower()
        category = ARXIV_CATEGORIES.get(topic, 'cs.AI')
        print(f"\nFetching papers for: {topic} (category: {category})")
        
        papers = fetch_arxiv_papers(category, MAX_PAPERS)
        
        for paper in papers:
            print(f"  - {paper['title'][:60]}...")
            summary = summarize_with_groq(paper['abstract'], paper['title'])
            paper['summary'] = summary
            paper['topic'] = topic
            all_papers.append(paper)
    
    # Save results
    today = datetime.now().strftime('%Y-%m-%d')
    output_dir = Path('summaries')
    output_dir.mkdir(exist_ok=True)
    
    # Save as JSON
    json_file = output_dir / f'{today}.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_papers, f, indent=2, ensure_ascii=False)
    
    # Save as readable markdown
    md_file = output_dir / f'{today}.md'
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(f"# Research Paper Summaries - {today}\n\n")
        f.write(f"**Topics**: {', '.join(TOPICS)}\n")
        f.write(f"**Total Papers**: {len(all_papers)}\n\n")
        f.write("---\n\n")
        
        for i, paper in enumerate(all_papers, 1):
            f.write(f"## {i}. {paper['title']}\n\n")
            f.write(f"**Authors**: {paper['authors']}\n\n")
            f.write(f"**Published**: {paper['published']}\n\n")
            f.write(f"**Topic**: {paper['topic']}\n\n")
            f.write(f"**Summary**: {paper['summary']}\n\n")
            f.write(f"**Link**: [{paper['url']}]({paper['url']})\n\n")
            f.write("---\n\n")
    
    print(f"\n✅ Saved {len(all_papers)} paper summaries to {output_dir}")
    print(f"   - JSON: {json_file}")
    print(f"   - Markdown: {md_file}")

if __name__ == '__main__':
    main()
