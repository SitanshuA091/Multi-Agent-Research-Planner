from agents.planner import PlannerAgent
from agents.retriever import RetrieverAgent
from agents.summarizer import SummarizerAgent
from pathlib import Path
import json


def main():
    print("="*80)
    print("RESEARCH PIPELINE TEST")
    print("="*80)
    print()
    
    # Get topic from user
    topic = input("Enter research topic: ").strip()
    print(f"\nYou entered: '{topic}'")
    print()
    
    if not topic.strip():
        print("Error: No topic entered!")
        return
    
    # STEP 1: Planner (with LangGraph retry and manual edit)
    planner = PlannerAgent()
    result = planner.plan(topic)
    keywords = result['keywords']
    
    # Save keywords to JSON
    print()
    save_keywords = input("Save keywords to JSON? (y/n): ").lower().strip()
    
    if save_keywords == 'y':
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        
        keywords_file = output_dir / "keywords.json"
        
        with open(keywords_file, 'w', encoding='utf-8') as f:
            json.dump({"topic": topic, "keywords": keywords}, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Keywords saved to: {keywords_file}")
    
    # STEP 2: Retriever (simple, no retry)
    print()
    input("Press Enter to start retrieval...")
    
    retriever = RetrieverAgent()
    retrieval_results = retriever.retrieve(keywords)
    
    # Save retrieval results option
    print()
    save_retrieval = input("\nDownload retrieval results as JSON? (y/n): ").lower().strip()
    
    if save_retrieval == 'y':
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / "retrieval_results.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(retrieval_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Retrieval results saved to: {output_file}")
    
    # STEP 3: Summarizer (automatic, no retry)
    print()
    proceed_summary = input("\nGenerate research summaries? (y/n): ").lower().strip()
    
    if proceed_summary != 'y':
        print("\nStopping pipeline. Goodbye!")
        return
    
    summarizer = SummarizerAgent()
    summaries = summarizer.summarize(retrieval_results)
    
    # Display summary preview
    print()
    print("="*80)
    print("SUMMARY PREVIEW")
    print("="*80)
    
    for item in summaries:
        print(f"\nKeyword: {item['keyword']}")
        print(f"  Summaries generated: {len(item['summaries'])}")
    
    print()
    print("-"*80)
    
    # Save summaries
    save_summaries = input("\nDownload summaries as JSON? (y/n): ").lower().strip()
    
    if save_summaries == 'y':
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / "summaries.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Summaries saved to: {output_file}")
    else:
        print("\nSummaries not saved.")
    
    print("\n" + "="*80)
    print("Pipeline complete!")
    print("="*80)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()