import os
import json
import feedparser
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path

# Configuration
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
TOPICS = os.environ.get('TOPICS', 'machine learning').split(',')
MAX_PAPERS = 10  # Adjust based on how many papers you want daily
REQUEST_DELAY = 3  # Seconds between arXiv requests to avoid rate limiting

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

def fetch_arxiv_papers_api(category, max_results=10):
    """Fetch recent papers using arXiv API (more reliable than RSS)"""
    base_url = 'http://export.arxiv.org/api/query?'
    
    # Search for papers from the last 7 days
    query = f'cat:{category}'
    params = {
        'search_query': query,
        'start': 0,
        'max_results': max_results,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending'
    }
    
    try:
        url = base_url + '&'.join([f'{k}={v}' for k, v in params.items()])
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        feed = feedparser.parse(response.content)
        
        papers = []
        for entry in feed.entries:
            # Extract authors
            authors = ', '.join([author.name for author in entry.authors]) if hasattr(entry, 'authors') else 'Unknown'
            
            papers.append({
                'title': entry.title,
                'authors': authors,
                'abstract': entry.summary,
                'url': entry.link,
                'published': entry.published if hasattr(entry, 'published') else 'Unknown date',
                'arxiv_id': entry.id.split('/abs/')[-1] if hasattr(entry, 'id') else None
            })
        
        return papers
    except Exception as e:
        print(f"  Error fetching from arXiv API: {e}")
        return []

def fetch_arxiv_papers(category, max_results=10):
    """Fetch recent papers from arXiv RSS feed"""
    url = f'http://export.arxiv.org/rss/{category}'
    
    try:
        feed = feedparser.parse(url)
        
        if not feed.entries:
            print(f"  RSS feed empty, trying API instead...")
            return fetch_arxiv_papers_api(category, max_results)
        
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
    except Exception as e:
        print(f"  Error fetching RSS: {e}, trying API...")
        return fetch_arxiv_papers_api(category, max_results)

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
    print(f"Using {REQUEST_DELAY}s delay between requests to respect arXiv rate limits\n")
    
    all_papers = []
    seen_titles = set()  # Track duplicates across categories
    
    # Deduplicate topics that share categories
    unique_categories = {}
    for topic in TOPICS:
        topic = topic.strip().lower()
        category = ARXIV_CATEGORIES.get(topic, 'cs.AI')
        if category not in unique_categories:
            unique_categories[category] = []
        unique_categories[category].append(topic)
    
    print(f"Fetching from {len(unique_categories)} unique arXiv categories for {len(TOPICS)} topics\n")
    
    # Fetch papers for each unique category
    for category, topics in unique_categories.items():
        topic_names = ', '.join(topics)
        print(f"Fetching papers for: {topic_names} (category: {category})")
        
        papers = fetch_arxiv_papers(category, MAX_PAPERS)
        print(f"  Found {len(papers)} papers")
        
        for paper in papers:
            # Skip duplicates (same paper in multiple categories)
            if paper['title'] in seen_titles:
                print(f"  Skipping duplicate: {paper['title'][:60]}...")
                continue
            
            seen_titles.add(paper['title'])
            print(f"  Processing: {paper['title'][:60]}...")
            
            summary = summarize_with_groq(paper['abstract'], paper['title'])
            paper['summary'] = summary
            paper['topic'] = topics[0]  # Assign first topic from this category
            all_papers.append(paper)
    
    # Save results even if no papers found
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
        
        if not all_papers:
            f.write("*No new papers found today. This can happen on weekends or when arXiv feeds are delayed.*\n\n")
            f.write("The workflow will run again tomorrow and fetch any new papers.\n")
        else:
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
    
    if not all_papers:
        print("\n⚠️  No papers found today. This is normal on weekends or during arXiv update cycles.")
        print("   The workflow will continue to run daily and fetch papers when available.")

if __name__ == '__main__':
    main()
