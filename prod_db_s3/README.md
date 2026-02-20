# Road Assessment API
RF-DETR based road defect detection pipeline with FastAPI, MySQL, and AWS S3.

---

## System Overview

```
Client → POST /jobs  (video S3 URL + SRT S3 URL)
       → Job queued → processed one at a time
       → Results uploaded to S3
       → All links saved to MySQL
       → Terminal notification on complete/fail
```

---

## S3 Folder Structure

```
novametrics-ai-data-processing/
└── Chattisgarh_Road_defect_detection/
    └── 2026-02-06_DJI_20260206174112_0754_D/     ← date_videoname (parent folder)
        ├── input/                                  ← original files (uploaded manually)
        │   ├── DJI_20260206174112_0754_D.MP4
        │   └── DJI_20260206174112_0754_D.SRT
        ├── temp/                                   ← safe to delete after review
        │   ├── defects/
        │   │   └── Patch/
        │   │       ├── DJI_..._Patch_crop_000001.jpg
        │   │       └── DJI_..._Patch_crop_000002.jpg
        │   └── frames/
        │       ├── DJI_..._Patch_frame_000001.jpg
        │       └── DJI_..._Patch_frame_000002.jpg
        └── output/                                 ← keep forever
            ├── annotated_video/
            │   └── DJI_20260206174112_0754_D_annotated.mp4
            └── reports/
                └── DJI_20260206174112_0754_D_detections.json
```

---

## Database Schema

### `processing_jobs` table
| Column | Description |
|--------|-------------|
| id | Job UUID |
| status | pending / downloading / processing / uploading / completed / failed |
| input_video_s3_url | S3 URL of input video |
| input_srt_s3_url | S3 URL of input SRT |
| input_s3_prefix | S3 input/ folder prefix |
| temp_s3_prefix | S3 temp/ folder prefix (crops + frames) |
| output_s3_prefix | S3 output/ folder prefix (video + report) |
| annotated_video_s3 | Full S3 URL of annotated video |
| report_json_s3 | Full S3 URL of JSON report |
| video_basename | Video filename without extension |
| starting_chainage_m | Starting chainage in meters |
| ending_chainage_m | Ending chainage in meters |
| total_detections | Total defects found |
| created_at / started_at / completed_at | Timestamps |
| error_message | Error details if failed |

### `detections` table
| Column | Description |
|--------|-------------|
| id | Auto increment |
| job_id | Foreign key to processing_jobs |
| defect_type | e.g. Patch, Cracking, Pothole |
| chainage_avg_m | Location on road in meters |
| gps_latitude / gps_longitude | GPS coordinates |
| crop_image_s3_url | S3 URL of cropped defect image (in temp/) |
| frame_image_s3_url | S3 URL of full frame image (in temp/) |
| frame_start / frame_end | Frame numbers |
| polygon | Segmentation polygon (JSON) |

---

## Project File Structure

```
prod_db_s3/
├── main.py                  ← FastAPI app, API endpoints
├── .env                     ← All configuration (never commit this)
├── requirements.txt
├── schema.sql
├── config/
│   ├── __init__.py
│   └── settings.py          ← Pydantic settings loaded from .env
├── core/
│   ├── __init__.py
│   ├── database.py          ← SQLAlchemy models + CRUD functions
│   ├── job_runner.py        ← Queue system + job orchestration
│   ├── pipeline.py          ← RF-DETR inference + tracking
│   ├── s3_manager.py        ← S3 upload/download + folder structure
│   ├── srt_parser.py        ← DJI SRT GPS parsing
│   └── tracker.py           ← Segmentation tracker
├── utils/
│   ├── __init__.py
│   ├── file_manager.py      ← Local temp/final folder management
│   └── logger.py            ← JSON structured logging
├── temp/                    ← Local working directory (auto cleaned)
├── final/                   ← Local staging before S3 upload (auto cleaned)
└── logs/                    ← Log files
```

---

## Setup

### 1. Install dependencies
```bash
pip install fastapi uvicorn pydantic pydantic-settings sqlalchemy pymysql \
            cryptography boto3 python-dotenv awscli
```

### 2. Configure .env
```env
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=ap-south-1
S3_BUCKET_NAME=novametrics-ai-data-processing
S3_OUTPUT_PREFIX=Chattisgarh_Road_defect_detection

DB_HOST=localhost
DB_PORT=3306
DB_NAME=road_assessment
DB_USER=roaduser
DB_PASSWORD=your_password

MODEL_CHECKPOINT_PATH=/path/to/checkpoint_best_total.pth
COCO_JSON_PATH=/path/to/_annotations.coco.json
IMAGE_SIZE=432
NUM_QUERIES=200

PROCESS_EVERY_N_FRAMES=1
BATCH_SIZE=1
OUT_WIDTH=1280
OUT_HEIGHT=720
MASK_OPACITY=0.20
GLOBAL_THRESHOLD=0.30
NMS_THRESHOLD=0.50
IOU_THRESHOLD=0.30
MAX_DISTANCE=50
MAX_LOST=30

TEMP_DIR=/path/to/prod_db_s3/temp
FINAL_DIR=/path/to/prod_db_s3/final
LOG_DIR=/path/to/prod_db_s3/logs
LOG_LEVEL=INFO

API_HOST=0.0.0.0
API_PORT=8001
API_WORKERS=1
```

### 3. Setup MySQL
```bash
sudo mysql
```
```sql
CREATE DATABASE road_assessment;
CREATE USER 'roaduser'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON road_assessment.* TO 'roaduser'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

### 4. Start the server
```bash
cd /path/to/prod_db_s3
python3 main.py
```

---

## API Usage

### Submit a job
```bash
curl -X POST "http://localhost:8001/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "video_s3_url": "s3://novametrics-ai-data-processing/Chattisgarh_Road_defect_detection/input/2026-02-06_17-41-12/DJI_20260206174112_0754_D.MP4",
    "srt_s3_url":   "s3://novametrics-ai-data-processing/Chattisgarh_Road_defect_detection/input/2026-02-06_17-41-12/DJI_20260206174112_0754_D.SRT",
    "starting_chainage_m": 0.0
  }'
```

Response:
```json
{"job_id": "920792ea-...", "status": "pending", "message": "Job accepted..."}
```

### Check job status
```bash
curl http://localhost:8001/jobs/920792ea-...
```

### List all jobs
```bash
curl http://localhost:8001/jobs
```

### Get detections for a job
```bash
curl http://localhost:8001/jobs/920792ea-.../detections
```

### Health check
```bash
curl http://localhost:8001/health
```

---

## Upload Input Video to S3

```bash
# Configure AWS CLI
aws configure

# Upload with correct folder structure
aws s3 cp "/path/to/DJI_20260206174112_0754_D.MP4" \
  "s3://novametrics-ai-data-processing/Chattisgarh_Road_defect_detection/input/2026-02-06_17-41-12/DJI_20260206174112_0754_D.MP4"

aws s3 cp "/path/to/DJI_20260206174112_0754_D.SRT" \
  "s3://novametrics-ai-data-processing/Chattisgarh_Road_defect_detection/input/2026-02-06_17-41-12/DJI_20260206174112_0754_D.SRT"
```

---

## Queue System

Jobs are processed **one at a time**. Submit multiple jobs freely — they will queue automatically:

```
📥 JOB QUEUED! — Video 1 — Position 1 — Processing now
📥 JOB QUEUED! — Video 2 — Position 2 — Waiting, 1 job ahead

▶  STARTING JOB: abc-123
✅ JOB COMPLETED! — 3 detections — 928.6m

▶  STARTING JOB: def-456
✅ JOB COMPLETED! — 7 detections — 1850.2m

All jobs done!
```

---

## Delete Temp Files from S3

After reviewing crop/frame images, delete temp to save storage:

```bash
aws s3 rm "s3://novametrics-ai-data-processing/Chattisgarh_Road_defect_detection/2026-02-06_DJI_20260206174112_0754_D/temp/" --recursive
```

---

## Ignored Classes (not detected/saved)

These classes are skipped completely:
`white_mark`, `water_mark`, `doubt`, `bump`, `guard_post`, `overhead_sign_board`, `sign_board`, `kerb_damage`

---

## Multi-Video Chainage Continuity

For continuous road surveys across multiple videos, pass the ending chainage of the previous job as starting chainage of the next:

```bash
# Video 1 — starts at 0
curl -X POST "http://localhost:8001/jobs" -d '{"starting_chainage_m": 0.0, ...}'

# Video 2 — starts where Video 1 ended (928.6m)
curl -X POST "http://localhost:8001/jobs" -d '{"starting_chainage_m": 928.6, ...}'
```
