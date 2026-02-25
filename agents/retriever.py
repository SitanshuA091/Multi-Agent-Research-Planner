import os
import time
import requests
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class RetrieverAgent:
    def __init__(self):
        """Initialize retriever with API endpoints"""
        self.wiki_search_url = "https://en.wikipedia.org/w/rest.php/v1/search/page"
        self.wiki_summary_url = "https://en.wikipedia.org/api/rest_v1/page/summary"
        self.arxiv_api_url = "http://export.arxiv.org/api/query"
        
        self.headers = {
            "User-Agent": "ResearchPlannerApp/1.0 (Educational Project)"
        }
    
    def _fetch_wikipedia(self, keyword: str) -> Dict[str, str]:
        try:
            search_params = {"q": keyword, "limit": 1}
            
            search_response = requests.get(
                self.wiki_search_url, 
                params=search_params,
                headers=self.headers,
                timeout=10
            )
            
            if not search_response.ok:
                return {"source": "wikipedia", "title": "", "url": "", "content": ""}
            
            search_data = search_response.json()
            pages = search_data.get('pages', [])
            
            if not pages:
                return {"source": "wikipedia", "title": "", "url": "", "content": ""}
            
            article_title = pages[0]['title']
            summary_url = f"{self.wiki_summary_url}/{article_title}"
            
            summary_response = requests.get(summary_url, headers=self.headers, timeout=10)
            
            if summary_response.ok:
                summary_data = summary_response.json()
                content = summary_data.get('extract', '')
                url = summary_data.get('content_urls', {}).get('desktop', {}).get('page', '')
                
                return {
                    "source": "wikipedia",
                    "title": article_title,
                    "url": url,
                    "content": content
                }
            
            return {"source": "wikipedia", "title": "", "url": "", "content": ""}
            
        except Exception as e:
            return {"source": "wikipedia", "title": "", "url": "", "content": ""}
    
    def _fetch_arxiv(self, keyword: str, max_results: int = 1) -> List[Dict]:
        try:
            params = {
                "search_query": f"all:{keyword}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            
            response = requests.get(self.arxiv_api_url, params=params, timeout=15)
            
            if not response.ok:
                return []
            
            papers = self._parse_arxiv_response(response.text)
            return papers
            
        except Exception as e:
            return []
    
    def _parse_arxiv_response(self, xml_text: str) -> List[Dict]:
        try:
            papers = []
            entries = xml_text.split('<entry>')
            
            for entry in entries[1:]:
                title_start = entry.find('<title>') + 7
                title_end = entry.find('</title>')
                title = entry[title_start:title_end].strip() if title_start > 6 else ""
                
                summary_start = entry.find('<summary>') + 9
                summary_end = entry.find('</summary>')
                summary = entry[summary_start:summary_end].strip() if summary_start > 8 else ""
                
                id_start = entry.find('<id>') + 4
                id_end = entry.find('</id>')
                paper_url = entry[id_start:id_end].strip() if id_start > 3 else ""
                
                published_start = entry.find('<published>') + 11
                published_end = entry.find('</published>')
                published = entry[published_start:published_end].strip()[:10] if published_start > 10 else ""
                
                if title and summary:
                    papers.append({
                        "source": "arxiv",
                        "title": title,
                        "url": paper_url,
                        "published": published,
                        "abstract": summary
                    })
            
            return papers
            
        except Exception as e:
            return []
    
    def retrieve(self, keywords: List[str]) -> List[Dict]:
        print("\n" + "="*80)
        print("RETRIEVER AGENT")
        print("="*80)
        
        results = []
        total_keywords = len(keywords)
        
        print(f"\nFetching sources for {total_keywords} keywords...")
        
        for i, keyword in enumerate(keywords, 1):
            print(f"  [{i}/{total_keywords}] {keyword}...")
            wiki_result = self._fetch_wikipedia(keyword)
            time.sleep(1)
            
            arxiv_results = self._fetch_arxiv(keyword)
            results.append({
                "keyword": keyword,
                "wikipedia": wiki_result,
                "arxiv_papers": arxiv_results
            })
            if i < total_keywords:
                time.sleep(1)
        
        print(f"\nFetching complete\n")
        print("="*80)
        print("RETRIEVED SOURCES")
        print("="*80)
        print("\n WIKIPEDIA SOURCES:\n")
        wiki_count = 0
        for result in results:
            wiki = result['wikipedia']
            if wiki['title']:
                wiki_count += 1
                print(f"  • {wiki['title']}")
                print(f"    {wiki['url']}")
                print()
        
        if wiki_count == 0:
            print("  (No Wikipedia sources found)\n")
        print("ARXIV PAPERS:\n")
        arxiv_count = 0
        for result in results:
            for paper in result['arxiv_papers']:
                arxiv_count += 1
                print(f"  • {paper['title']}")
                print(f"    {paper['url']}")
                print(f"    Published: {paper['published']}")
                print()
        
        if arxiv_count == 0:
            print("  (No arXiv papers found)\n")
        
        print("="*80)
        print(f"Total: {wiki_count} Wikipedia articles, {arxiv_count} arXiv papers")
        print("="*80)
        
        return results