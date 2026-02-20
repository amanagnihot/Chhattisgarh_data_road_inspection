# Road Assessment Pipeline — Production System

Drone-video road defect detection powered by **RFDETRSegMedium**, served via a
**FastAPI** REST API, with outputs stored in **AWS S3** and indexed in **MySQL**.

---

## Architecture

```
Client
  │
  │  POST /jobs  { video_s3_url, srt_s3_url }
  ▼
FastAPI (main.py)
  │
  ├─► Job record created in MySQL  (status = pending)
  │
  └─► Background task: job_runner.run_job(job_id)
            │
            ├─ 1. DOWNLOAD   S3 → temp/{job_id}/input/
            ├─ 2. PROCESS    pipeline.py (RF-DETR + tracker)
            │       → temp/{job_id}/output/
            ├─ 3. MOVE       temp/output/ → final/{job_id}/
            ├─ 4. UPLOAD     final/{job_id}/ → S3
            ├─ 5. DB WRITE   detections table (with S3 URLs)
            └─ 6. CLEANUP    rm -rf temp/{job_id}/

  GET /jobs/{job_id}  → status + S3 output URLs
  GET /jobs/{job_id}/detections  → all defects with S3 image links
```

### Folder Structure (runtime)

```
temp/
└── {job_id}/
    ├── input/
    │   ├── DJI_0001.MP4
    │   └── DJI_0001.SRT
    └── output/
        ├── annotated_video/
        ├── defects/
        │   ├── pothole/
        │   ├── cracking/
        │   └── ...
        └── frames/

final/
└── {job_id}/
    ├── output/         ← same structure as temp/output (after move)
    └── reports/
        └── DJI_0001_detections.json
```

### S3 Output Structure

```
s3://{bucket}/road-assessments/{job_id}/{video_basename}/
├── output/
│   ├── annotated_video/
│   │   └── DJI_0001_annotated.mp4
│   ├── defects/
│   │   ├── pothole/
│   │   │   ├── DJI_0001_pothole_crop_000001.jpg
│   │   │   └── ...
│   │   └── cracking/
│   └── frames/
│       └── DJI_0001_pothole_frame_000001.jpg
└── reports/
    └── DJI_0001_detections.json
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your AWS credentials, MySQL details, and model paths
```

### 3. Set up MySQL

```bash
mysql -u root -p < schema.sql
# or let the API auto-create tables on first startup
```

### 4. Start the API

```bash
python main.py
# or with uvicorn directly:
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## API Reference

### Submit a Job

```http
POST /jobs
Content-Type: application/json

{
  "video_s3_url": "s3://my-bucket/input/DJI_0001.MP4",
  "srt_s3_url":   "s3://my-bucket/input/DJI_0001.SRT",
  "starting_chainage_m": 0.0
}
```

**Response (202 Accepted)**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Job accepted. Poll GET /jobs/550e... for status."
}
```

---

### Poll Job Status

```http
GET /jobs/{job_id}
```

**Response**
```json
{
  "job_id": "550e8400-...",
  "status": "completed",
  "video_basename": "DJI_0001",
  "total_detections": 47,
  "starting_chainage_m": 0.0,
  "ending_chainage_m": 1523.4,
  "annotated_video_s3": "s3://bucket/road-assessments/.../DJI_0001_annotated.mp4",
  "report_json_s3":     "s3://bucket/road-assessments/.../DJI_0001_detections.json",
  "s3_prefix":          "road-assessments/550e.../DJI_0001",
  "created_at":  "2026-02-19T10:00:00",
  "started_at":  "2026-02-19T10:00:02",
  "completed_at":"2026-02-19T10:08:45"
}
```

Status values: `pending → downloading → processing → uploading → completed | failed`

---

### Get All Detections

```http
GET /jobs/{job_id}/detections
```

Returns every defect with chainage, GPS, and S3 image URLs.

---

### Generate Presigned URL

```http
POST /presign?s3_url=s3://bucket/...&expiry_seconds=3600
```

Converts an internal `s3://` URL into a shareable HTTPS link.

---

### List All Jobs

```http
GET /jobs?limit=20&offset=0
```

---

## Database Tables

| Table | Purpose |
|-------|---------|
| `processing_jobs` | One row per submitted job; tracks status, S3 URLs, chainage |
| `detections` | One row per defect; stores GPS, chainage, defect type, S3 image URLs |

---

## Multi-Video Chainage Continuity

To process multiple videos as a continuous road segment, pass the
`ending_chainage_m` from the previous job as `starting_chainage_m` in the next:

```python
# Job 1 — starts at 0 km
r1 = requests.post("/jobs", json={
    "video_s3_url": "s3://bucket/video1.MP4",
    "srt_s3_url":   "s3://bucket/video1.SRT",
    "starting_chainage_m": 0.0,
})

# Poll until completed, then:
status1 = requests.get(f"/jobs/{r1.json()['job_id']}").json()

# Job 2 — continues from where Job 1 ended
r2 = requests.post("/jobs", json={
    "video_s3_url": "s3://bucket/video2.MP4",
    "srt_s3_url":   "s3://bucket/video2.SRT",
    "starting_chainage_m": status1["ending_chainage_m"],
})
```