from __future__ import annotations
import json
import uuid
import threading
from datetime import datetime
from pathlib import Path

from config.settings import get_settings
from core import database as db
from core.pipeline import VideoPipeline, load_class_names, IGNORED_CLASSES
from core.s3_manager import (
    build_s3_input_prefix,
    build_s3_temp_prefix,
    build_s3_output_prefix,
    download_from_s3,
    parse_s3_url,
    upload_directory_to_s3,
)
from utils.file_manager import FileManager
from utils.logger import get_logger

logger   = get_logger(__name__)
settings = get_settings()

_model        = None
_class_names  = None
_job_queue    = []
_queue_lock   = threading.Lock()
_queue_thread = None


def _get_model_and_classes():
    global _model, _class_names
    if _model is None:
        import torch
        import numpy as np
        from rfdetr import RFDETRSegPreview
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _class_names = load_class_names(settings.coco_json_path)
        _model = RFDETRSegPreview(
            pretrain_weights=settings.model_checkpoint_path,
            num_queries=settings.num_queries,
            device=device,
            image_size=settings.image_size,
            max_image_size=settings.image_size,
        )
        try:
            _model.optimize_for_inference()
        except Exception as e:
            logger.warning("optimize_for_inference skipped", extra={"reason": str(e)})
        dummy = [np.random.randint(0, 255, (settings.out_height, settings.out_width, 3), dtype=np.uint8)]
        with torch.no_grad():
            _model.predict(dummy, threshold=settings.global_threshold)
        logger.info("Model ready")
    return _model, _class_names


def _queue_worker():
    while True:
        job_id = None
        with _queue_lock:
            if _job_queue:
                job_id = _job_queue.pop(0)

        if job_id is None:
            threading.Event().wait(2)
            continue

        print(f"\n{'='*60}")
        print(f"▶  STARTING JOB : {job_id}")
        print(f"   Queue remaining : {len(_job_queue)}")
        print(f"{'='*60}\n")

        try:
            result = _run_single_job(job_id)
            print(f"\n{'='*60}")
            print(f"✅ JOB COMPLETED!")
            print(f"   Job ID      : {job_id}")
            print(f"   Video       : {result.get('video_basename')}")
            print(f"   Detections  : {result.get('total_detections')}")
            print(f"   Chainage    : {result.get('ending_chainage_m', 0):.1f} m")
            print(f"")
            print(f"   📁 S3 STRUCTURE:")
            print(f"   ├── input/  : {result.get('input_s3_prefix')}")
            print(f"   ├── temp/   : {result.get('temp_s3_prefix')}  <- delete when done")
            print(f"   └── output/ : {result.get('output_s3_prefix')}  <- keep forever")
            print(f"")
            print(f"   📹 Annotated Video : {result.get('annotated_video_s3')}")
            print(f"   📊 Report JSON     : {result.get('report_json_s3')}")
            print(f"   Queue left  : {len(_job_queue)}")
            if len(_job_queue) == 0:
                print(f"   All jobs done!")
            print(f"{'='*60}\n")

        except Exception as exc:
            print(f"\n{'='*60}")
            print(f"JOB FAILED!")
            print(f"   Job ID     : {job_id}")
            print(f"   Error      : {exc}")
            print(f"   Queue left : {len(_job_queue)}")
            print(f"{'='*60}\n")


def _ensure_queue_running():
    global _queue_thread
    if _queue_thread is None or not _queue_thread.is_alive():
        _queue_thread = threading.Thread(target=_queue_worker, daemon=True)
        _queue_thread.start()
        logger.info("Queue worker started")


def submit_job(
    video_s3_url: str,
    srt_s3_url: str,
    starting_chainage_m: float = 0.0,
) -> str:
    db.create_all_tables()
    job_id = str(uuid.uuid4())
    db.create_job(
        job_id=job_id,
        video_s3_url=video_s3_url,
        srt_s3_url=srt_s3_url,
        starting_chainage_m=starting_chainage_m,
    )

    with _queue_lock:
        _job_queue.append(job_id)
        position = len(_job_queue)

    _ensure_queue_running()

    print(f"\n{'='*60}")
    print(f"JOB QUEUED!")
    print(f"   Job ID   : {job_id}")
    print(f"   Video    : {video_s3_url.split('/')[-1]}")
    print(f"   Position : {position} in queue")
    print(f"   Status   : {'Processing now' if position == 1 else 'Waiting - ' + str(position-1) + ' job(s) ahead'}")
    print(f"{'='*60}\n")
    return job_id


def run_job(job_id: str) -> dict:
    pass


def _run_single_job(job_id: str) -> dict:
    try:
        model, class_names = _get_model_and_classes()
        job_record = db.get_job(job_id)
        if job_record is None:
            raise ValueError(f"Job {job_id} not found")

        video_s3_url        = job_record["input_video_s3_url"]
        srt_s3_url          = job_record["input_srt_s3_url"]
        starting_chainage_m = job_record.get("starting_chainage_m") or 0.0

        _, video_key   = parse_s3_url(video_s3_url)
        video_filename = Path(video_key).name
        video_basename = Path(video_key).stem
        _, srt_key     = parse_s3_url(srt_s3_url)
        srt_filename   = Path(srt_key).name

        input_s3_prefix  = build_s3_input_prefix(video_basename)
        temp_s3_prefix   = build_s3_temp_prefix(video_basename)
        output_s3_prefix = build_s3_output_prefix(video_basename)

        print(f"\n   S3 folders for this job:")
        print(f"   input/  : {input_s3_prefix}")
        print(f"   temp/   : {temp_s3_prefix}")
        print(f"   output/ : {output_s3_prefix}\n")

        db.update_job_status(
            job_id, "downloading",
            video_basename=video_basename,
            input_s3_prefix=input_s3_prefix,
            temp_s3_prefix=temp_s3_prefix,
            output_s3_prefix=output_s3_prefix,
        )

        file_manager = FileManager(
            job_id=job_id,
            class_names=class_names,
            ignored_classes=IGNORED_CLASSES,
        )

        local_video = download_from_s3(video_s3_url, file_manager.temp_video_path(video_filename))
        local_srt   = download_from_s3(srt_s3_url,   file_manager.temp_srt_path(srt_filename))

        db.update_job_status(job_id, "processing", started_at=datetime.utcnow())

        pipeline = VideoPipeline(model=model, class_names=class_names)
        all_detections, ending_chainage = pipeline.run(
            video_path=local_video,
            srt_path=local_srt,
            file_manager=file_manager,
            video_basename=video_basename,
            starting_chainage_m=starting_chainage_m,
        )

        file_manager.move_outputs_to_final(video_basename)

        summary = {}
        for d in all_detections:
            summary.setdefault(d["defect_type"], {"count": 0})["count"] += 1

        report_path = file_manager.final_report_path(video_basename)
        with open(report_path, "w") as f:
            json.dump({
                "job_id":              job_id,
                "video_name":          video_filename,
                "processing_date":     datetime.utcnow().isoformat(),
                "model":               "RFDETRSegPreview",
                "s3_structure": {
                    "input":  input_s3_prefix,
                    "temp":   temp_s3_prefix,
                    "output": output_s3_prefix,
                },
                "input_video_s3":      video_s3_url,
                "input_srt_s3":        srt_s3_url,
                "starting_chainage_m": starting_chainage_m,
                "ending_chainage_m":   ending_chainage,
                "total_detections":    len(all_detections),
                "summary":             summary,
                "detections":          all_detections,
            }, f, indent=2, default=str)

        db.update_job_status(job_id, "uploading")

        temp_local   = file_manager.final_temp_dir(video_basename)
        output_local = file_manager.final_output_dir(video_basename)

        temp_url_map   = upload_directory_to_s3(temp_local,   temp_s3_prefix,   max_workers=8)
        output_url_map = upload_directory_to_s3(output_local, output_s3_prefix, max_workers=8)

        for det in all_detections:
            crop_rel  = file_manager.final_crop_path(
                det["defect_type"], det["id"], video_basename
            ).relative_to(temp_local).as_posix()
            frame_rel = file_manager.final_frame_path(
                det["defect_type"], det["id"], video_basename
            ).relative_to(temp_local).as_posix()
            det["s3_urls"] = {
                "crop":  temp_url_map.get(crop_rel,  ""),
                "frame": temp_url_map.get(frame_rel, ""),
            }

        annotated_video_s3 = output_url_map.get(
            file_manager.final_annotated_video_path(video_basename)
            .relative_to(output_local).as_posix(), ""
        )
        report_json_s3 = output_url_map.get(
            report_path.relative_to(output_local).as_posix(), ""
        )

        db.save_detections_bulk(job_id, all_detections)
        db.update_job_status(
            job_id, "completed",
            input_s3_prefix=input_s3_prefix,
            temp_s3_prefix=temp_s3_prefix,
            output_s3_prefix=output_s3_prefix,
            annotated_video_s3=annotated_video_s3,
            report_json_s3=report_json_s3,
            ending_chainage_m=ending_chainage,
            total_detections=len(all_detections),
        )

        file_manager.cleanup_temp()

        return {
            "job_id":             job_id,
            "status":             "completed",
            "video_basename":     video_basename,
            "total_detections":   len(all_detections),
            "ending_chainage_m":  ending_chainage,
            "input_s3_prefix":    input_s3_prefix,
            "temp_s3_prefix":     temp_s3_prefix,
            "output_s3_prefix":   output_s3_prefix,
            "annotated_video_s3": annotated_video_s3,
            "report_json_s3":     report_json_s3,
            "summary":            summary,
        }

    except Exception as exc:
        logger.error("Job failed", extra={"job_id": job_id, "error": str(exc)}, exc_info=True)
        db.update_job_status(job_id, "failed", error_message=str(exc))
        raise
