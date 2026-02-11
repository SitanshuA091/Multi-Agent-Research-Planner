import os
import json
from pathlib import Path
from typing import List, TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
load_dotenv()

class ResearchPlan(BaseModel):
    keywords: List[str] = Field(description="List of research keywords/terms for literature search")

class PlannerState(TypedDict):
    topic: str
    keywords: List[str]
    user_decision: str  # 'accept', 'retry', 'manual'
    retry_count: int


class PlannerAgent:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.llm = ChatGroq(
            model=model_name,
            temperature=0.3,
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.parser = JsonOutputParser(pydantic_object=ResearchPlan)
        self.prompt_template = self._load_prompt()
        
        self.workflow = self._build_workflow()
        
    def _load_prompt(self) -> PromptTemplate:
        prompt_file = Path(__file__).parent.parent / "prompts" / "planner_prompt.txt"
        
        if not prompt_file.exists():
            raise FileNotFoundError(
                f"Prompt file not found at {prompt_file}\n"
                f"Please ensure 'planner_prompt.txt' exists in the prompts folder."
            )
        
        prompt_text = prompt_file.read_text()
        
        return PromptTemplate(
            template=prompt_text,
            input_variables=["topic", "retry_feedback"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
    
    def generate_keywords_node(self, state: PlannerState) -> PlannerState:
        topic = state['topic']
        retry_count = state['retry_count']
        
        if retry_count > 0:
            retry_feedback = "\n\nIMPORTANT: The previous keywords were too generic. Generate MORE SPECIFIC, TECHNICAL terms related to the exact topic. Avoid broad, general terms."
        else:
            retry_feedback = ""
        
        chain = self.prompt_template | self.llm | self.parser
        
        result = chain.invoke({"topic": topic, "retry_feedback": retry_feedback})
        
        state['keywords'] = result['keywords']
        
        return state
    
    def display_keywords_node(self, state: PlannerState) -> PlannerState:
        keywords = state['keywords']
        retry_count = state['retry_count']
        
        print("\n" + "="*80)
        print("GENERATED KEYWORDS")
        print("="*80)
        
        for i, kw in enumerate(keywords, 1):
            print(f"  {i}. {kw}")
        
        print("="*80)
        
        if retry_count > 0:
            print(f"(Retry attempt {retry_count}/1)")
        
        return state
    
    def user_decision_node(self, state: PlannerState) -> PlannerState:
        retry_count = state['retry_count']
        
        print()
        if retry_count == 0:
            print("Options:")
            print("  [A] Accept these keywords")
            print("  [R] Retry - regenerate all keywords (1 retry available)")
            print("  [M] Manual - replace a specific keyword")
        else:
            print("Options:")
            print("  [A] Accept these keywords")
            print("  [M] Manual - replace a specific keyword")
            print("  (No more retries available)")
        
        choice = input("\nYour choice: ").strip().upper()
        
        if choice == 'R' and retry_count == 0:
            state['user_decision'] = 'retry'
        elif choice == 'M':
            state['user_decision'] = 'manual'
        else:
            state['user_decision'] = 'accept'
        
        return state
    
    def manual_edit_node(self, state: PlannerState) -> PlannerState:
        keywords = state['keywords']
        
        print("\nCurrent keywords:")
        for i, kw in enumerate(keywords, 1):
            print(f"  {i}. {kw}")
        
        try:
            choice = int(input(f"\nWhich keyword to replace? (1-{len(keywords)}): ").strip())
            
            if 1 <= choice <= len(keywords):
                new_keyword = input("Enter new keyword: ").strip()
                
                if new_keyword:
                    old_keyword = keywords[choice - 1]
                    keywords[choice - 1] = new_keyword
                    state['keywords'] = keywords
                    
                    print(f"\n Replaced '{old_keyword}' with '{new_keyword}'")
                else:
                    print("\n No keyword entered, keeping original")
            else:
                print("\n Invalid choice, keeping original keywords")
        
        except ValueError:
            print("\n Invalid input, keeping original keywords")
        
        return state
    
    def retry_node(self, state: PlannerState) -> PlannerState:
        state['retry_count'] += 1
        print("\n→ Regenerating keywords with more specific instructions...")
        return state
    
    
    def route_decision(self, state: PlannerState) -> Literal["accept", "retry", "manual"]:
        decision = state['user_decision']
        
        if decision == 'retry':
            return "retry"
        elif decision == 'manual':
            return "manual"
        else:
            return "accept"
    
    def route_after_manual(self, state: PlannerState) -> Literal["display", "accept"]:
        print()
        ask_again = input("Review updated keywords? (y/n): ").strip().lower()
        
        if ask_again == 'y':
            return "display"
        else:
            return "accept"
    
    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(PlannerState)
        workflow.add_node("generate", self.generate_keywords_node)
        workflow.add_node("display", self.display_keywords_node)
        workflow.add_node("user_decision", self.user_decision_node)
        workflow.add_node("manual_edit", self.manual_edit_node)
        workflow.add_node("retry", self.retry_node)
        
        workflow.set_entry_point("generate")
        
        workflow.add_edge("generate", "display")
        workflow.add_edge("display", "user_decision")
        
        workflow.add_conditional_edges(
            "user_decision",
            self.route_decision,
            {
                "accept": END,
                "retry": "retry",
                "manual": "manual_edit"
            }
        )
        
        workflow.add_edge("retry", "generate")
        
        workflow.add_conditional_edges(
            "manual_edit",
            self.route_after_manual,
            {
                "display": "display",
                "accept": END
            }
        )
        
        return workflow.compile()
    
    def plan(self, topic: str) -> dict:
        print("\n" + "="*80)
        print("PLANNER AGENT")
        print("="*80)
        print(f"\nTopic: {topic}")
        
        # Initialize state
        initial_state: PlannerState = {
            'topic': topic,
            'keywords': [],
            'user_decision': '',
            'retry_count': 0
        }
        final_state = self.workflow.invoke(initial_state)
        
        print(f"\n✓ Keywords finalized")
        print("="*80)
        
        return {'keywords': final_state['keywords']}