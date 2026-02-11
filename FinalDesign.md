# Multi-Agent Research Planner

## Overview
A system of collaborative agents that research technical topics using free APIs (Wikipedia, ArXiv). Agents decompose queries, retrieve information, summarize findings, and synthesize structured research briefs.

**Tech Stack:** LangGraph, Gemini/Groq APIs, FastAPI, Streamlit

---

## Agent Pipeline

```
User Query
   ↓
Planner Agent (task decomposition)
   ↓
Retriever Agent (fetch documents)
   ↓
Summarizer Agent (condense content)
   ↓
Synthesizer Agent (final brief)
   ↓
Structured Research Brief
```

---

## Agent Specifications

### 1. Planner Agent
**Responsibility:** Task decomposition

**Input:**
```
"Research topic: Graph Neural Networks"
```

**Output (JSON):**
```json
{
  "subtasks": [
    "What are Graph Neural Networks?",
    "Key architectures and variants",
    "Major applications",
    "Limitations and open research problems"
  ]
}
```

**Implementation:** LLM prompt → forced JSON output

---

### 2. Retriever Agent
**Responsibility:** Fetch documents from external APIs

**Process:**
- Iterate over subtasks
- Call Wikipedia REST API
- Call arXiv endpoint (no API key needed)
- Extract: title, summary, abstract, URL/Paper ID

**Output (JSON):**
```json
{
  "question": "What are Graph Neural Networks?",
  "sources": [
    {
      "source": "wikipedia",
      "content": "..."
    },
    {
      "source": "arxiv",
      "content": "..."
    }
  ]
}
```

---

### 3. Summarizer Agent
**Responsibility:** Condense raw text into structured bullets

**Process:**
- Take raw text
- Call LLM with constraints: max bullets, citation markers
- Batch LLM calls
- No invention, only summarization

**Output:**
- 5–7 bullet points per source
- Preserve citations (source tags)

---

### 4. Synthesizer Agent
**Responsibility:** Combine all summaries into final brief

**Process:**
- Single LLM call
- Combine all summaries

**Output (JSON):**
```json
{
  "title": "Graph Neural Networks: A Research Overview",
  "overview": "...",
  "sections": [
    {
      "heading": "Introduction",
      "content": "..."
    }
  ],
  "references": [...]
}
```

---

## Architecture

### Backend (FastAPI)
- Agent orchestration via LangGraph
- External API calls (Wikipedia, arXiv)
- Prompt engineering + LLM reasoning
- Return structured JSON responses

### Frontend (Streamlit MVP)
**UI Components:**
- Text input: research topic
- Button: "Run Research"
- Progress bar per agent
- Expandable sections:
  - Planner output
  - Retrieved sources
  - Final brief

**API Integration:**
```python
requests.post("http://localhost:8000/research", json={"topic": "..."})
```

---

## API Endpoint Design

### `POST /research`

**Request:**
```json
{
  "topic": "Graph Neural Networks"
}
```

**Flow:**
1. Planner → generate subtasks
2. Loop over subtasks:
   - Retriever → fetch documents
   - Summarizer → create summaries
3. Synthesizer → produce final brief
4. Return JSON response

---

## Design Principles

Each agent follows:
- **Single Responsibility:** One clear purpose
- **Clean Interface:** Well-defined inputs/outputs
- **Swappable:** LLM, API, or rules can be replaced
- **Structure:** (Prompt) + (Executor)

---

## Dependencies

```toml
[project]
name = "multi-agent-research"
version = "0.1.0"
dependencies = [
  "fastapi",
  "uvicorn",
  "pydantic",
  "requests",
  "wikipedia",
  "arxiv",
  "python-dotenv",
  "streamlit",
  "langchain-core",
  "langchain-groq",
  "langchain-google-genai",
  "langgraph",
  "tiktoken"
]
```

**LLM Models:**
- Groq: `llama-3.1-70b-versatile` or similar
- Gemini: `gemini-2.0-flash-exp`

---

## Constraints

- **Free APIs only:** Wikipedia, arXiv (no authentication needed)
- **No data invention:** Summarizer must not hallucinate
- **Structured outputs:** All agents return JSON
- **Single synthesis pass:** Synthesizer makes one LLM call
- **Batch processing:** Summarizer handles multiple sources efficiently