# 📚 Multi-Agent Research Planner

A modular research automation system that uses LangGraph and multiple AI agents to generate comprehensive research reports from academic sources.

## Overview

This system employs four specialized agents working in sequence to transform a research topic into a professional PDF report. Each agent handles a specific part of the research pipeline, from keyword generation to final synthesis.

## Architecture

### Agent Pipeline

```
User Topic → Planner → Retriever → Summarizer → Synthesizer → PDF Report
```

### Agent Descriptions

**1. Planner Agent**
- **Purpose:** Generates 4-5 research keywords from a user-provided topic
- **Technology:** LangGraph state machine with conditional routing
- **Features:**
  - LLM-based keyword generation
  - User review and approval
  - Single retry with feedback for more specific terms
  - Manual keyword replacement option
- **Model:** Groq Llama 3.3 70B Versatile

**2. Retriever Agent**
- **Purpose:** Fetches relevant documents from Wikipedia and arXiv
- **Features:**
  - Wikipedia REST API integration (no authentication required)
  - arXiv API integration (no authentication required)
  - Retrieves top articles and research papers for each keyword
  - Groups and displays all sources by type
- **APIs Used:** Wikipedia REST API, arXiv Query API
- **No LLM Required:** Pure API-based retrieval

**3. Summarizer Agent**
- **Purpose:** Condenses each source into 5-7 key bullet points
- **Features:**
  - Separate prompts for Wikipedia articles vs research papers
  - Preserves technical terminology and key findings
  - Batch processing of all sources with strict summarization only
- **Model:** Google Gemini 2.5 Flash Experimental

**4. Synthesizer Agent**
- **Purpose:** Combines all summaries into a cohesive research report
- **Features:**
  - Integrates insights across all sources
  - Identifies themes, contradictions, and research gaps
  - Generates 800-1200 word academic report
  - Produces professional PDF with serif typography
- **Model:** Groq's GPT-OSS 120B
- **PDF Generation:** ReportLab with Times Roman font

## Technology Stack

### AI Models

| Agent | Model | Provider | Purpose |
|-------|-------|----------|---------|
| Planner | llama-3.3-70b-versatile | Groq | Keyword generation |
| Summarizer | gemini-3.0-pro=preview | Google | Source summarization |
| Synthesizer | GPT-OSS(120b) | Groq | Report synthesis |

<em>the model and provider choice is done so that I can avoid hitting the groq api more than the restricted RPM and space out those 2 API calls</em>

### Frameworks & Libraries

- **LangGraph:** State machine for agent workflows with conditional routing
- **LangChain:** LLM integration and prompt management
- **ReportLab:** Professional PDF generation
- **Requests:** API calls to Wikipedia and arXiv
- **Python 3.10+**

### External APIs

- **Wikipedia REST API:** Article retrieval (free, no key required)
- **arXiv API:** Research paper metadata and abstracts (free, no key required)


## Future Enhancements

- Additional source APIs (PubMed, Semantic Scholar)
- Citation management and bibliography generation
- Synthesizer feedback loop for completeness checks



