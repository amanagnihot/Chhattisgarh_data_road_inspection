-- ============================================================
-- Road Assessment — MySQL Schema
-- Run this once to set up the database, or let SQLAlchemy
-- create tables automatically via create_all_tables().
-- ============================================================

CREATE DATABASE IF NOT EXISTS road_assessment
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE road_assessment;

-- ── processing_jobs ───────────────────────────────────────────

CREATE TABLE IF NOT EXISTS processing_jobs (
    id                   VARCHAR(36)   NOT NULL PRIMARY KEY COMMENT 'UUID job identifier',
    status               ENUM(
                             'pending','downloading','processing',
                             'uploading','completed','failed'
                         ) NOT NULL DEFAULT 'pending',

    -- Input S3 references
    input_video_s3_url   VARCHAR(2048) NOT NULL,
    input_srt_s3_url     VARCHAR(2048) NOT NULL,
    video_basename       VARCHAR(255)  NULL,

    -- Output S3 references
    output_s3_prefix     VARCHAR(2048) NULL  COMMENT 'Base S3 prefix for all outputs',
    annotated_video_s3   VARCHAR(2048) NULL,
    report_json_s3       VARCHAR(2048) NULL,

    -- Video / road stats
    total_frames         INT           NULL,
    processed_frames     INT           NULL,
    video_fps            FLOAT         NULL,
    starting_chainage_m  FLOAT         NULL DEFAULT 0.0,
    ending_chainage_m    FLOAT         NULL,
    total_detections     INT           NULL DEFAULT 0,

    -- Timestamps
    created_at           DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at           DATETIME      NULL,
    completed_at         DATETIME      NULL,

    -- Error
    error_message        TEXT          NULL,

    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ── detections ────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS detections (
    id                   INT           NOT NULL AUTO_INCREMENT PRIMARY KEY,
    job_id               VARCHAR(36)   NOT NULL,
    det_sequence_id      INT           NOT NULL  COMMENT 'Per-job detection counter',
    track_id             INT           NULL,

    -- Defect info
    defect_type          VARCHAR(100)  NOT NULL,
    confidence           FLOAT         NULL,

    -- Video position
    video_name           VARCHAR(255)  NULL,
    frame_start          INT           NULL,
    frame_end            INT           NULL,
    timestamp_start      VARCHAR(50)   NULL,
    timestamp_end        VARCHAR(50)   NULL,

    -- Chainage
    chainage_start_m     FLOAT         NULL,
    chainage_end_m       FLOAT         NULL,
    chainage_avg_m       FLOAT         NULL,

    -- GPS
    gps_latitude         FLOAT         NULL,
    gps_longitude        FLOAT         NULL,

    -- S3 image URLs
    crop_image_s3_url    VARCHAR(2048) NULL  COMMENT 'S3 URL of the cropped defect image',
    frame_image_s3_url   VARCHAR(2048) NULL  COMMENT 'S3 URL of the full annotated frame',

    -- Polygon
    polygon              JSON          NULL,

    created_at           DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_detection_job
        FOREIGN KEY (job_id) REFERENCES processing_jobs(id) ON DELETE CASCADE,

    INDEX idx_job_id (job_id),
    INDEX idx_defect_type (defect_type),
    INDEX idx_chainage (chainage_avg_m)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;