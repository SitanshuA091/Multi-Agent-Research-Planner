from dotenv import load_dotenv
from pathlib import Path

# .env lives inside backend/, next to this file.
# Run from project root with: uvicorn backend.app:app --reload --port 8000
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

import os
import io
import uuid
import threading
from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from langsmith.run_helpers import trace

from agents.planner import PlannerAgent
from agents.retriever import RetrieverAgent
from agents.summarizer import SummarizerAgent
from agents.synthesizer import SynthesizerAgent
from backend.evals import ResearchAgentEvaluator

app = FastAPI(title="Research Planner Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

planner = PlannerAgent()
retriever = RetrieverAgent()
summarizer = SummarizerAgent()
synthesizer = SynthesizerAgent()
evaluator = ResearchAgentEvaluator()

jobs_lock = threading.Lock()
jobs: Dict[str, dict] = {}


def log_feedback(run_id, feedback_dict):
    if run_id is None:
        return
    try:
        evaluator.client.create_feedback(
            run_id=run_id,
            key=feedback_dict["key"],
            score=feedback_dict["score"],
            comment=feedback_dict.get("comment")
        )
    except Exception:
        pass


class JobStage(str, Enum):
    KEYWORDS_GENERATED = "keywords_generated"
    RETRIEVING = "retrieving"
    RETRIEVED = "retrieved"
    SUMMARIZING = "summarizing"
    SUMMARIZED = "summarized"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"


class SubmitRequest(BaseModel):
    topic: str


class SubmitResponse(BaseModel):
    job_id: str
    topic: str
    keywords: List[str]
    retry_count: int
    stage: JobStage


class RetryResponse(BaseModel):
    job_id: str
    keywords: List[str]
    retry_count: int
    stage: JobStage


class ManualEditRequest(BaseModel):
    index: int
    new_keyword: str


class ManualEditResponse(BaseModel):
    job_id: str
    keywords: List[str]
    stage: JobStage


class AcceptResponse(BaseModel):
    job_id: str
    stage: JobStage
    message: str


class StatusResponse(BaseModel):
    job_id: str
    topic: str
    stage: JobStage
    keywords: List[str]
    retry_count: int
    error: Optional[str] = None


class ResultResponse(BaseModel):
    job_id: str
    topic: str
    stage: JobStage
    report_text: Optional[str] = None
    generated_at: Optional[str] = None


def get_job_or_404(job_id: str) -> dict:
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def run_pipeline_background(job_id: str):
    with jobs_lock:
        job = jobs[job_id]
        topic = job["topic"]
        keywords = job["keywords"]
        job["stage"] = JobStage.RETRIEVING

    try:
        with trace(name="retriever_stage", run_type="chain", inputs={"keywords": keywords}) as rt:
            retrieval_results = retriever.retrieve(keywords)

            all_sources = []
            for r in retrieval_results:
                if r['wikipedia']['title']:
                    all_sources.append(r['wikipedia'])
                all_sources.extend(r['arxiv_papers'])

            rt.end(outputs={"sources": all_sources})
            retriever_run_id = rt.id

        fake_example = SimpleNamespace(inputs={"keywords": keywords})
        fake_run = SimpleNamespace(outputs={"sources": all_sources})

        quality_feedback = evaluator.source_quality_evaluator(fake_run, fake_example)
        diversity_feedback = evaluator.source_diversity_evaluator(fake_run, fake_example)
        log_feedback(retriever_run_id, quality_feedback)
        log_feedback(retriever_run_id, diversity_feedback)

        with jobs_lock:
            job["retrieval_results"] = retrieval_results
            job["stage"] = JobStage.SUMMARIZING

        with trace(name="summarizer_stage", run_type="chain", inputs={"retrieval_results": "omitted_for_brevity"}) as rt:
            summaries = summarizer.summarize(retrieval_results)
            rt.end(outputs={"summaries": summaries})
            summarizer_run_id = rt.id

        first_summary_for_eval = None
        for item in summaries:
            if item["summaries"]:
                first = item["summaries"][0]
                first_summary_for_eval = {
                    "source_content": first.get("url", ""),
                    "summary": " ".join(first.get("key_points", []))
                }
                break

        if first_summary_for_eval:
            fake_run = SimpleNamespace(outputs=first_summary_for_eval)
            fake_example = SimpleNamespace(inputs={})
            completeness_feedback = evaluator.summary_completeness_evaluator(fake_run, fake_example)
            log_feedback(summarizer_run_id, completeness_feedback)

        with jobs_lock:
            job["summaries"] = summaries
            job["stage"] = JobStage.SYNTHESIZING

        with trace(name="synthesizer_stage", run_type="chain", inputs={"topic": topic}) as rt:
            synthesis = synthesizer.synthesize(summaries, topic)
            rt.end(outputs={"report_text": synthesis["report_text"]})
            synthesizer_run_id = rt.id

        fake_example = SimpleNamespace(inputs={"topic": topic})
        fake_run = SimpleNamespace(outputs={"report_text": synthesis["report_text"]})

        coherence_feedback = evaluator.synthesis_coherence_evaluator(fake_run, fake_example)
        relevance_feedback = evaluator.synthesis_relevance_evaluator(fake_run, fake_example)
        structure_feedback = evaluator.synthesis_structure_evaluator(fake_run, fake_example)

        log_feedback(synthesizer_run_id, coherence_feedback)
        log_feedback(synthesizer_run_id, relevance_feedback)
        log_feedback(synthesizer_run_id, structure_feedback)

        with jobs_lock:
            job["synthesis"] = synthesis
            job["stage"] = JobStage.COMPLETED

    except Exception as e:
        with jobs_lock:
            job["stage"] = JobStage.FAILED
            job["error"] = str(e)


@app.post("/submit", response_model=SubmitResponse)
def submit_topic(request: SubmitRequest):
    if not request.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty")

    job_id = str(uuid.uuid4())

    with trace(name="planner_stage", run_type="chain", inputs={"topic": request.topic}) as rt:
        result = planner.generate_keywords(request.topic, retry_count=0)
        rt.end(outputs={"keywords": result["keywords"]})
        planner_run_id = rt.id

    fake_example = SimpleNamespace(inputs={"topic": request.topic})
    fake_run = SimpleNamespace(outputs={"keywords": result["keywords"]})

    relevance_feedback = evaluator.keyword_relevance_evaluator(fake_run, fake_example)
    specificity_feedback = evaluator.keyword_specificity_evaluator(fake_run, fake_example)
    log_feedback(planner_run_id, relevance_feedback)
    log_feedback(planner_run_id, specificity_feedback)

    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "topic": request.topic,
            "keywords": result["keywords"],
            "retry_count": 0,
            "stage": JobStage.KEYWORDS_GENERATED,
            "retrieval_results": None,
            "summaries": None,
            "synthesis": None,
            "error": None,
            "created_at": datetime.now().isoformat()
        }

    return SubmitResponse(
        job_id=job_id,
        topic=request.topic,
        keywords=result["keywords"],
        retry_count=0,
        stage=JobStage.KEYWORDS_GENERATED
    )


@app.post("/jobs/{job_id}/retry", response_model=RetryResponse)
def retry_keywords(job_id: str):
    job = get_job_or_404(job_id)

    if job["stage"] != JobStage.KEYWORDS_GENERATED:
        raise HTTPException(status_code=400, detail="Retry only allowed before accepting keywords")

    if job["retry_count"] >= 1:
        raise HTTPException(status_code=400, detail="No more retries available")

    new_retry_count = job["retry_count"] + 1

    with trace(name="planner_stage_retry", run_type="chain", inputs={"topic": job["topic"]}) as rt:
        result = planner.generate_keywords(job["topic"], retry_count=new_retry_count)
        rt.end(outputs={"keywords": result["keywords"]})
        planner_run_id = rt.id

    fake_example = SimpleNamespace(inputs={"topic": job["topic"]})
    fake_run = SimpleNamespace(outputs={"keywords": result["keywords"]})

    relevance_feedback = evaluator.keyword_relevance_evaluator(fake_run, fake_example)
    specificity_feedback = evaluator.keyword_specificity_evaluator(fake_run, fake_example)
    log_feedback(planner_run_id, relevance_feedback)
    log_feedback(planner_run_id, specificity_feedback)

    with jobs_lock:
        job["keywords"] = result["keywords"]
        job["retry_count"] = new_retry_count

    return RetryResponse(
        job_id=job_id,
        keywords=job["keywords"],
        retry_count=job["retry_count"],
        stage=job["stage"]
    )


@app.post("/jobs/{job_id}/manual-edit", response_model=ManualEditResponse)
def manual_edit_keyword(job_id: str, request: ManualEditRequest):
    job = get_job_or_404(job_id)

    if job["stage"] != JobStage.KEYWORDS_GENERATED:
        raise HTTPException(status_code=400, detail="Manual edit only allowed before accepting keywords")

    if not (0 <= request.index < len(job["keywords"])):
        raise HTTPException(status_code=400, detail="Invalid keyword index")

    if not request.new_keyword.strip():
        raise HTTPException(status_code=400, detail="New keyword cannot be empty")

    updated_keywords = planner.replace_keyword(
        job["keywords"].copy(),
        request.index,
        request.new_keyword
    )

    with jobs_lock:
        job["keywords"] = updated_keywords

    return ManualEditResponse(
        job_id=job_id,
        keywords=job["keywords"],
        stage=job["stage"]
    )


@app.post("/jobs/{job_id}/accept", response_model=AcceptResponse)
def accept_keywords(job_id: str, background_tasks: BackgroundTasks):
    job = get_job_or_404(job_id)

    if job["stage"] != JobStage.KEYWORDS_GENERATED:
        raise HTTPException(status_code=400, detail="Keywords already accepted or job not in correct stage")

    background_tasks.add_task(run_pipeline_background, job_id)

    return AcceptResponse(
        job_id=job_id,
        stage=JobStage.RETRIEVING,
        message="Pipeline started. Poll /jobs/{job_id}/status for progress."
    )


@app.get("/jobs/{job_id}/status", response_model=StatusResponse)
def get_status(job_id: str):
    job = get_job_or_404(job_id)

    return StatusResponse(
        job_id=job_id,
        topic=job["topic"],
        stage=job["stage"],
        keywords=job["keywords"],
        retry_count=job["retry_count"],
        error=job.get("error")
    )


@app.get("/jobs/{job_id}/sources")
def get_sources(job_id: str):
    job = get_job_or_404(job_id)

    if job["retrieval_results"] is None:
        raise HTTPException(status_code=400, detail="Sources not available yet")

    return {"job_id": job_id, "retrieval_results": job["retrieval_results"]}


@app.get("/jobs/{job_id}/summaries")
def get_summaries(job_id: str):
    job = get_job_or_404(job_id)

    if job["summaries"] is None:
        raise HTTPException(status_code=400, detail="Summaries not available yet")

    return {"job_id": job_id, "summaries": job["summaries"]}


@app.get("/jobs/{job_id}/result", response_model=ResultResponse)
def get_result(job_id: str):
    job = get_job_or_404(job_id)

    if job["stage"] != JobStage.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Job not completed yet. Current stage: {job['stage']}")

    synthesis = job["synthesis"]

    return ResultResponse(
        job_id=job_id,
        topic=job["topic"],
        stage=job["stage"],
        report_text=synthesis["report_text"],
        generated_at=synthesis["generated_at"]
    )


@app.get("/jobs/{job_id}/pdf")
def get_pdf(job_id: str):
    job = get_job_or_404(job_id)

    if job["stage"] != JobStage.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Job not completed yet. Current stage: {job['stage']}")

    safe_filename = "".join(c for c in job["topic"] if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_filename = safe_filename.replace(' ', '_')[:50]

    buffer = io.BytesIO()
    synthesizer.generate_pdf(job["synthesis"], buffer)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={safe_filename}_report.pdf"}
    )


@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}