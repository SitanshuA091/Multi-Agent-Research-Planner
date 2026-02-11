"""
Summarizer Agent - Condenses sources into structured bullet points
Uses Gemini Flash 2.0 (free tier, fast, good at summarization)
"""
import os
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
load_dotenv()

class Summary(BaseModel):
    source_type: str = Field(description="'wikipedia' or 'arxiv'")
    title: str = Field(description="Title of the source")
    key_points: List[str] = Field(description="5-7 key points as bullet points")


class SummarizerAgent:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.3,  # Lower temperature for focused summarization
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, str]:
        prompts_dir = Path(__file__).parent.parent / "prompts"
        
        prompts = {}
        wiki_prompt_file = prompts_dir / "summarizer_wiki.txt"
        prompts['wikipedia'] = wiki_prompt_file.read_text()
        arxiv_prompt_file = prompts_dir / "summarizer_arxiv.txt"
        prompts['arxiv'] = arxiv_prompt_file.read_text()
        
        return prompts
    
    def _create_summary_prompt(self, source_type: str, title: str, content: str) -> str:
        prompt_template = self.prompts.get(source_type, self.prompts['wikipedia'])
        prompt = prompt_template.format(title=title, content=content)
        
        return prompt
    
    def _parse_bullet_points(self, response) -> List[str]:
        if isinstance(response, list):
            return response[:7]  # Return first 7 if already a list
        
        # Convert to string if not already
        if not isinstance(response, str):
            response = str(response)
        
        lines = response.strip().split('\n')
        bullets = []
        
        for line in lines:
            line = line.strip()
            # Removing leading dashes, asterisks, or numbers
            if line.startswith('- '):
                line = line[2:]
            elif line.startswith('* '):
                line = line[2:]
            elif line and line[0].isdigit() and '. ' in line:
                line = line.split('. ', 1)[1]
            
            if line:
                bullets.append(line)
        if len(bullets) < 5:
            return bullets  # Return what we have
        elif len(bullets) > 7:
            return bullets[:7]  # Take first 7
        else:
            return bullets
    
    def _summarize_source(self, source_type: str, title: str, content: str, url: str = "") -> Dict:
        if not content or not title:
            return {
                "source_type": source_type,
                "title": title or "No title",
                "url": url,
                "key_points": ["No content available to summarize"],
                "original_length": 0,
                "summary_length": 0
            }
        prompt = self._create_summary_prompt(source_type, title, content)
        try:
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            key_points = self._parse_bullet_points(response_text)
            return {
                "source_type": source_type,
                "title": title,
                "url": url,
                "key_points": key_points,
                "original_length": len(content),
                "summary_length": sum(len(point) for point in key_points)
            }
            
        except Exception as e:
            print(f"  ✗ Error summarizing {title}: {e}")
            return {
                "source_type": source_type,
                "title": title,
                "url": url,
                "key_points": [f"Error during summarization: {str(e)}"],
                "original_length": len(content),
                "summary_length": 0
            }
    
    def summarize(self, retrieved_docs: List[Dict]) -> List[Dict]:
        print("\n" + "="*80)
        print("SUMMARIZER AGENT")
        print("="*80)
        
        all_summaries = []
        total_sources = sum(
            1 + len(doc.get('arxiv_papers', []))  # 1 wiki + N arxiv papers
            for doc in retrieved_docs
        )
        
        current = 0
        
        for doc in retrieved_docs:
            keyword = doc['keyword']
            summaries_for_keyword = []
            
            print(f"\nProcessing keyword: '{keyword}'")
            
            # Summarizing Wikipedia sources
            wiki = doc.get('wikipedia', {})
            if wiki.get('title'):
                current += 1
                print(f"  [{current}/{total_sources}] Summarizing Wikipedia: {wiki['title'][:50]}...")
                
                wiki_summary = self._summarize_source(
                    source_type="wikipedia",
                    title=wiki['title'],
                    content=wiki.get('content', ''),
                    url=wiki.get('url', '')
                )
                summaries_for_keyword.append(wiki_summary)
            
            # Summarizing arXiv papers
            arxiv_papers = doc.get('arxiv_papers', [])
            for paper in arxiv_papers:
                current += 1
                print(f"  [{current}/{total_sources}] Summarizing arXiv: {paper['title'][:50]}...")
                
                arxiv_summary = self._summarize_source(
                    source_type="arxiv",
                    title=paper['title'],
                    content=paper.get('abstract', ''),
                    url=paper.get('url', '')
                )
                summaries_for_keyword.append(arxiv_summary)
            
            all_summaries.append({
                "keyword": keyword,
                "summaries": summaries_for_keyword
            })
        
        print(f"\n✓ Summarization complete")
        print(f"  Total sources summarized: {current}")
        print("="*80)
        
        return all_summaries