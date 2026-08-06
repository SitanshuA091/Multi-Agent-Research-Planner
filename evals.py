import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv

from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Run, Example
from langchain_groq import ChatGroq

from agents.planner import PlannerAgent
from agents.retriever import RetrieverAgent
from agents.summarizer import SummarizerAgent
from agents.synthesizer import SynthesizerAgent

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "research-planner-agent"


class ResearchAgentEvaluator:
    
    def __init__(self):
        self.client = Client()
        self.judge_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        self.planner = PlannerAgent()
        self.retriever = RetrieverAgent()
        self.summarizer = SummarizerAgent()
        self.synthesizer = SynthesizerAgent()
    
    def create_dataset(self, dataset_name: str, test_topics: List[str]):
        try:
            dataset = self.client.create_dataset(
                dataset_name=dataset_name,
                description="Research topics for evaluating multi-agent pipeline"
            )
        except Exception:
            dataset = self.client.read_dataset(dataset_name=dataset_name)
        
        for topic in test_topics:
            self.client.create_example(
                inputs={"topic": topic},
                outputs={},
                dataset_id=dataset.id
            )
        
        return dataset
    
    def _score_with_llm(self, prompt: str) -> float:
        response = self.judge_llm.invoke(prompt)
        text = response.content if hasattr(response, 'content') else str(response)
        
        try:
            score = float(text.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5
        
    def keyword_relevance_evaluator(self, run: Run, example: Example) -> Dict:
        topic = example.inputs.get("topic", "")
        keywords = run.outputs.get("keywords", [])
        
        prompt = f"""Rate how relevant these keywords are to the topic on a scale of 0.0 to 1.0.
 
Topic: {topic}
Keywords: {', '.join(keywords)}
 
Consider relevance, specificity, and coverage of the topic.
Return ONLY a number between 0.0 and 1.0."""
        
        score = self._score_with_llm(prompt)
        
        return {
            "key": "keyword_relevance",
            "score": score,
            "comment": f"Keywords: {keywords}"
        }
    
    
    def keyword_specificity_evaluator(self, run: Run, example: Example) -> Dict:
        keywords = run.outputs.get("keywords", [])
        
        prompt = f"""Rate how specific (not generic) these keywords are on a scale of 0.0 to 1.0.

Keywords: {', '.join(keywords)}

Consider if they are technical/domain-specific rather than generic terms.
Return ONLY a number between 0.0 and 1.0."""
        
        score = self._score_with_llm(prompt)
        
        return {
            "key": "keyword_specificity",
            "score": score
        }
    
    def source_quality_evaluator(self, run: Run, example: Example) -> Dict:
        sources = run.outputs.get("sources", [])
        
        if not sources:
            return {"key": "source_quality", "score": 0.0}
        
        quality_score = 0.0
        for source in sources:
            if source.get('title'):
                quality_score += 0.3
            if source.get('content') and len(source.get('content', '')) > 100:
                quality_score += 0.4
            if source.get('url'):
                quality_score += 0.3
        
        avg_score = quality_score / len(sources)
        
        return {
            "key": "source_quality",
            "score": avg_score,
            "comment": f"Evaluated {len(sources)} sources"
        }
    
    def source_diversity_evaluator(self, run: Run, example: Example) -> Dict:
        sources = run.outputs.get("sources", [])
        
        if not sources:
            return {"key": "source_diversity", "score": 0.0}
        
        source_types = set(s.get('source', 'unknown') for s in sources)
        diversity_score = len(source_types) / 2.0
        
        return {
            "key": "source_diversity",
            "score": min(1.0, diversity_score)
        }
    
    def summary_completeness_evaluator(self, run: Run, example: Example) -> Dict:
        source_content = run.outputs.get("source_content", "")
        summary = run.outputs.get("summary", "")
        
        if not source_content or not summary:
            return {"key": "summary_completeness", "score": 0.0}
        
        prompt = f"""Rate how complete this summary is compared to the source on a scale of 0.0 to 1.0.

Source (first 500 chars): {source_content[:500]}
Summary: {summary}

Consider if main points and key details are preserved.
Return ONLY a number between 0.0 and 1.0."""
        
        score = self._score_with_llm(prompt)
        
        return {
            "key": "summary_completeness",
            "score": score
        }
    
    def synthesis_coherence_evaluator(self, run: Run, example: Example) -> Dict:
        report = run.outputs.get("report_text", "")
        
        if not report:
            return {"key": "synthesis_coherence", "score": 0.0}
        
        prompt = f"""Rate the coherence and flow of this research report on a scale of 0.0 to 1.0.

Report (first 1000 chars): {report[:1000]}

Consider logical flow, transitions, and consistent style.
Return ONLY a number between 0.0 and 1.0."""
        
        score = self._score_with_llm(prompt)
        
        return {
            "key": "synthesis_coherence",
            "score": score
        }
    
    def synthesis_relevance_evaluator(self, run: Run, example: Example) -> Dict:
        topic = example.inputs.get("topic", "")
        report = run.outputs.get("report_text", "")
        
        if not report:
            return {"key": "synthesis_relevance", "score": 0.0}
        
        prompt = f"""Rate how relevant this report is to the topic on a scale of 0.0 to 1.0.

Topic: {topic}
Report (first 1000 chars): {report[:1000]}

Consider if it directly addresses the topic without tangential content.
Return ONLY a number between 0.0 and 1.0."""
        
        score = self._score_with_llm(prompt)
        
        return {
            "key": "synthesis_relevance",
            "score": score
        }
    
    def synthesis_structure_evaluator(self, run: Run, example: Example) -> Dict:
        report = run.outputs.get("report_text", "")
        
        if not report:
            return {"key": "synthesis_structure", "score": 0.0}
        
        required_sections = ['Introduction', 'Main Findings', 'Applications', 'Challenges', 'Conclusion']
        found_sections = sum(1 for section in required_sections if section.lower() in report.lower())
        
        score = found_sections / len(required_sections)
        
        return {
            "key": "synthesis_structure",
            "score": score,
            "comment": f"Found {found_sections}/{len(required_sections)} sections"
        }
    
    def run_planner_pipeline(self, inputs: Dict) -> Dict:
        topic = inputs["topic"]
        result = self.planner.generate_keywords(topic, retry_count=0)
        return {"keywords": result["keywords"]}
    
    def run_retriever_pipeline(self, inputs: Dict) -> Dict:
        topic = inputs["topic"]
        keyword_result = self.planner.generate_keywords(topic, retry_count=0)
        keywords = keyword_result["keywords"]
        
        results = self.retriever.retrieve(keywords)
        
        all_sources = []
        for result in results:
            if result['wikipedia']['title']:
                all_sources.append(result['wikipedia'])
            all_sources.extend(result['arxiv_papers'])
        
        return {"sources": all_sources, "keywords": keywords}
    
    def run_summarizer_pipeline(self, inputs: Dict) -> Dict:
        topic = inputs["topic"]
        keyword_result = self.planner.generate_keywords(topic, retry_count=0)
        keywords = keyword_result["keywords"]
        
        retrieval_results = self.retriever.retrieve(keywords)
        summaries = self.summarizer.summarize(retrieval_results)
        
        first_summary = summaries[0]['summaries'][0] if summaries and summaries[0]['summaries'] else {}
        
        return {
            "summary": ' '.join(first_summary.get('key_points', [])),
            "source_content": first_summary.get('url', ''),
            "all_summaries": summaries
        }
    
    def run_full_pipeline(self, inputs: Dict) -> Dict:
        topic = inputs["topic"]
        
        keyword_result = self.planner.generate_keywords(topic, retry_count=0)
        keywords = keyword_result["keywords"]
        
        retrieval_results = self.retriever.retrieve(keywords)
        
        summaries = self.summarizer.summarize(retrieval_results)
        
        synthesis = self.synthesizer.synthesize(summaries, topic)
        
        return {
            "keywords": keywords,
            "report_text": synthesis["report_text"],
            "topic": topic
        }
    
    def evaluate_planner_agent(self, dataset_name: str):
        print(f"\n{'='*80}")
        print("EVALUATING PLANNER AGENT")
        print(f"{'='*80}\n")
        
        results = evaluate(
            self.run_planner_pipeline,
            data=dataset_name,
            evaluators=[
                self.keyword_relevance_evaluator,
                self.keyword_specificity_evaluator
            ],
            experiment_prefix="planner-eval",
            client=self.client
        )
        
        return results
    
    def evaluate_retriever_agent(self, dataset_name: str):
        print(f"\n{'='*80}")
        print("EVALUATING RETRIEVER AGENT")
        print(f"{'='*80}\n")
        
        results = evaluate(
            self.run_retriever_pipeline,
            data=dataset_name,
            evaluators=[
                self.source_quality_evaluator,
                self.source_diversity_evaluator
            ],
            experiment_prefix="retriever-eval",
            client=self.client
        )
        
        return results
    
    def evaluate_summarizer_agent(self, dataset_name: str):
        print(f"\n{'='*80}")
        print("EVALUATING SUMMARIZER AGENT")
        print(f"{'='*80}\n")
        
        results = evaluate(
            self.run_summarizer_pipeline,
            data=dataset_name,
            evaluators=[
                self.summary_completeness_evaluator
            ],
            experiment_prefix="summarizer-eval",
            client=self.client
        )
        
        return results
    
    def evaluate_full_pipeline(self, dataset_name: str):
        print(f"\n{'='*80}")
        print("EVALUATING FULL PIPELINE")
        print(f"{'='*80}\n")
        
        results = evaluate(
            self.run_full_pipeline,
            data=dataset_name,
            evaluators=[
                self.keyword_relevance_evaluator,
                self.keyword_specificity_evaluator,
                self.synthesis_coherence_evaluator,
                self.synthesis_relevance_evaluator,
                self.synthesis_structure_evaluator
            ],
            experiment_prefix="full-pipeline-eval",
            client=self.client
        )
        
        return results


# def main():
#     evaluator = ResearchAgentEvaluator()
#
#     test_topics = [
#         "Vision Transformers",
#         "Recurrence Memory Transformer",
#         "Reinforcement Learning with Human Feedback"
#     ]
#
#     dataset_name = "research-agent-eval-dataset"
#
#     print(f"Creating dataset: {dataset_name}")
#     evaluator.create_dataset(dataset_name, test_topics)
#
#     print("\nRunning full pipeline evaluation...")
#     results = evaluator.evaluate_full_pipeline(dataset_name)
#
#     print(f"\n{'='*80}")
#     print("EVALUATION COMPLETE")
#     print(f"{'='*80}")
#     print(f"View results at: https://smith.langchain.com/")
#     print(f"Project: research-planner-agent")
#     print(f"Dataset: {dataset_name}")
#     print(f"{'='*80}\n")
#
#
# if __name__ == "__main__":
#     main()
