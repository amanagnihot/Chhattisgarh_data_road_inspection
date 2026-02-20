from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from config.settings import get_settings
from core import database as db
from core.job_runner import run_job, submit_job
from core.s3_manager import generate_presigned_url
from utils.logger import get_logger

logger   = get_logger(__name__)
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Road Assessment API")
    db.create_all_tables()
    logger.info("Database tables ready")
    yield
    logger.info("Road Assessment API shutting down")

app = FastAPI(
    title="Road Assessment API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SubmitJobRequest(BaseModel):
    video_s3_url: str
    srt_s3_url: str
    starting_chainage_m: float = 0.0

class SubmitJobResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    video_basename: Optional[str] = None
    total_detections: Optional[int] = None
    starting_chainage_m: Optional[float] = None
    ending_chainage_m: Optional[float] = None
    annotated_video_s3: Optional[str] = None
    report_json_s3: Optional[str] = None
    s3_prefix: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None

def _run_job_background(job_id: str):
    try:
        run_job(job_id)
    except Exception as exc:
        logger.error("Background job failed", extra={"job_id": job_id, "error": str(exc)})

@app.get("/health")
def health_check():
    return {"status": "ok", "version": app.version}

@app.post("/jobs", response_model=SubmitJobResponse, status_code=202)
def submit(request: SubmitJobRequest, background_tasks: BackgroundTasks):
    job_id = submit_job(
        video_s3_url=request.video_s3_url,
        srt_s3_url=request.srt_s3_url,
        starting_chainage_m=request.starting_chainage_m,
    )
    background_tasks.add_task(_run_job_background, job_id)
    return SubmitJobResponse(
        job_id=job_id,
        status="pending",
        message=f"Job accepted. Poll GET /jobs/{job_id} for status.",
    )

@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    record = db.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobStatusResponse(
        job_id=record["id"],
        status=record["status"],
        video_basename=record.get("video_basename"),
        total_detections=record.get("total_detections"),
        starting_chainage_m=record.get("starting_chainage_m"),
        ending_chainage_m=record.get("ending_chainage_m"),
        annotated_video_s3=record.get("annotated_video_s3"),
        report_json_s3=record.get("report_json_s3"),
        s3_prefix=record.get("output_s3_prefix"),
        created_at=record.get("created_at"),
        started_at=record.get("started_at"),
        completed_at=record.get("completed_at"),
        error_message=record.get("error_message"),
    )

@app.get("/jobs")
def list_jobs(limit: int = Query(default=20), offset: int = Query(default=0)):
    return {"jobs": db.list_jobs(limit=limit, offset=offset)}

@app.get("/jobs/{job_id}/detections")
def get_detections(job_id: str):
    record = db.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    detections = db.get_detections_for_job(job_id)
    return {"job_id": job_id, "total_detections": len(detections), "detections": detections}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.api_host, port=settings.api_port, reload=False)