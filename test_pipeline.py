from agents.planner import PlannerAgent
from agents.retriever import RetrieverAgent
from agents.summarizer import SummarizerAgent
from agents.synthesizer import SynthesizerAgent
from pathlib import Path
import json

def main():
    print("="*80)
    print("RESEARCH PIPELINE TEST")
    print("="*80)
    print()
    topic = input("Enter research topic: ").strip()
    print(f"\nYou entered: '{topic}'")
    print()
    
    if not topic.strip():
        print("Error: No topic entered!")
        return

    planner = PlannerAgent()
    result = planner.plan(topic)
    keywords = result['keywords']

    print()
    save_keywords = input("Save keywords to JSON? (y/n): ").lower().strip()
    
    if save_keywords == 'y':
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        
        keywords_file = output_dir / "keywords.json"
        
        with open(keywords_file, 'w', encoding='utf-8') as f:
            json.dump({"topic": topic, "keywords": keywords}, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Keywords saved to: {keywords_file}")
    
    print()
    input("Press Enter to start retrieval...")
    
    retriever = RetrieverAgent()
    retrieval_results = retriever.retrieve(keywords)
    
    print()
    save_retrieval = input("\nDownload retrieval results as JSON? (y/n): ").lower().strip()
    
    if save_retrieval == 'y':
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / "retrieval_results.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(retrieval_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Retrieval results saved to: {output_file}")

    print()
    proceed_summary = input("\nGenerate research summaries? (y/n): ").lower().strip()
    
    if proceed_summary != 'y':
        print("\nStopping pipeline. Goodbye!")
        return
    
    summarizer = SummarizerAgent()
    summaries = summarizer.summarize(retrieval_results)

    print()
    print("="*80)
    print("SUMMARY PREVIEW")
    print("="*80)
    
    for item in summaries:
        print(f"\nKeyword: {item['keyword']}")
        print(f"  Summaries generated: {len(item['summaries'])}")
    
    print()
    print("-"*80)

    save_summaries = input("\nDownload summaries as JSON? (y/n): ").lower().strip()
    
    if save_summaries == 'y':
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / "summaries.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Summaries saved to: {output_file}")

    print()
    proceed_synthesis = input("\nSynthesize final research report? (y/n): ").lower().strip()
    
    if proceed_synthesis != 'y':
        print("\nStopping pipeline. Goodbye!")
        return
    
    synthesizer = SynthesizerAgent()
    synthesis = synthesizer.synthesize(summaries, topic)

    print()
    print("="*80)
    print("SYNTHESIS PREVIEW")
    print("="*80)
    print(f"\nReport length: {len(synthesis['report_text'])} characters")
    print(f"\nFirst 300 characters:")
    print(synthesis['report_text'][:300] + "...")
    print()
    print("-"*80)

    save_pdf = input("\nGenerate PDF report? (y/n): ").lower().strip()
    
    if save_pdf == 'y':
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)

        safe_filename = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_filename = safe_filename.replace(' ', '_')[:50]  # Limit length
        
        pdf_file = output_dir / f"{safe_filename}_report.pdf"
        
        synthesizer.generate_pdf(synthesis, str(pdf_file))
        
        print(f"\n✓ PDF report saved to: {pdf_file}")

    save_synthesis_json = input("\nAlso save synthesis as JSON? (y/n): ").lower().strip()
    
    if save_synthesis_json == 'y':
        output_dir = Path(__file__).parent / "outputs"
        synthesis_file = output_dir / "synthesis.json"
        
        with open(synthesis_file, 'w', encoding='utf-8') as f:
            json.dump(synthesis, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Synthesis saved to: {synthesis_file}")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    output_dir = Path(__file__).parent / "outputs"
    if output_dir.exists():
        for file in sorted(output_dir.iterdir()):
            if file.is_file():
                print(f"  • {file.name}")

if __name__ == "__main__":
    main()