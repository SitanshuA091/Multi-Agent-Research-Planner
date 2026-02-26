import os
from pathlib import Path
from typing import List, TypedDict, Literal, Optional, Callable
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
    user_decision: str
    retry_count: int
    awaiting_user_input: bool
    manual_edit_index: Optional[int]
    manual_edit_value: Optional[str]


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
        state['awaiting_user_input'] = False
        
        return state
    
    def display_keywords_node(self, state: PlannerState) -> PlannerState:
        state['awaiting_user_input'] = True
        return state
    
    def user_decision_node(self, state: PlannerState) -> PlannerState:
        return state
    
    def manual_edit_node(self, state: PlannerState) -> PlannerState:
        keywords = state['keywords']
        manual_index = state.get('manual_edit_index')
        manual_value = state.get('manual_edit_value')
        
        if manual_index is not None and manual_value and 0 <= manual_index < len(keywords):
            keywords[manual_index] = manual_value
            state['keywords'] = keywords
        
        state['manual_edit_index'] = None
        state['manual_edit_value'] = None
        
        return state
    
    def retry_node(self, state: PlannerState) -> PlannerState:
        state['retry_count'] += 1
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
        if state.get('user_decision') == 'review':
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
        initial_state: PlannerState = {
            'topic': topic,
            'keywords': [],
            'user_decision': '',
            'retry_count': 0,
            'awaiting_user_input': False,
            'manual_edit_index': None,
            'manual_edit_value': None
        }
        
        final_state = self.workflow.invoke(initial_state)
        
        return {'keywords': final_state['keywords']}
    
    def generate_keywords(self, topic: str, retry_count: int = 0) -> dict:
        if retry_count > 0:
            retry_feedback = "\n\nIMPORTANT: The previous keywords were too generic. Generate MORE SPECIFIC, TECHNICAL terms related to the exact topic. Avoid broad, general terms."
        else:
            retry_feedback = ""
        
        chain = self.prompt_template | self.llm | self.parser
        result = chain.invoke({"topic": topic, "retry_feedback": retry_feedback})
        
        return {'keywords': result['keywords']}
    
    def replace_keyword(self, keywords: List[str], index: int, new_keyword: str) -> List[str]:
        if 0 <= index < len(keywords):
            keywords[index] = new_keyword
        return keywords