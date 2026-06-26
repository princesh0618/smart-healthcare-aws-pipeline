-- ============================================================================
-- Smart Healthcare Data Pipeline — Athena Analytics Queries
-- Deliverable #3: Demo Script
-- Database: healthcare_db   |   Engine: Amazon Athena (Presto/Trino SQL)
--
-- These four queries power the QuickSight "Smart Hospital Operations Dashboard".
-- Run each in the Athena query editor with healthcare_db selected as the database.
-- ============================================================================


-- ----------------------------------------------------------------------------
-- QUERY 0 (sanity check) — preview the patients table
-- ----------------------------------------------------------------------------
SELECT *
FROM patients
LIMIT 10;


-- ----------------------------------------------------------------------------
-- QUERY 1 — Average occupancy & wait time by department
-- Insight: Emergency shows the highest occupancy and the longest waits,
--          identifying it as the operational bottleneck.
-- ----------------------------------------------------------------------------
SELECT
    department,
    ROUND(AVG(occupancy_rate) * 100, 1) AS avg_occupancy_pct,
    ROUND(AVG(avg_wait_minutes), 1)     AS avg_wait_min,
    MAX(patients_waiting)               AS peak_queue
FROM operations
GROUP BY department
ORDER BY avg_occupancy_pct DESC;


-- ----------------------------------------------------------------------------
-- QUERY 2 — 30-day readmission rate by department and age group
-- Insight: Readmission risk rises sharply with age (65+ highest), concentrated
--          in emergency/oncology/cardiac cases.
-- Note: AVG of a 0/1 flag returns the readmission RATE.
-- ----------------------------------------------------------------------------
SELECT
    department,
    CASE
        WHEN age < 18 THEN '0-17'
        WHEN age BETWEEN 18 AND 40 THEN '18-40'
        WHEN age BETWEEN 41 AND 65 THEN '41-65'
        ELSE '65+'
    END AS age_group,
    COUNT(*)                            AS total_patients,
    SUM(readmitted_30d)                 AS readmissions,
    ROUND(AVG(readmitted_30d) * 100, 1) AS readmit_rate_pct
FROM patients
GROUP BY
    department,
    CASE
        WHEN age < 18 THEN '0-17'
        WHEN age BETWEEN 18 AND 40 THEN '18-40'
        WHEN age BETWEEN 41 AND 65 THEN '41-65'
        ELSE '65+'
    END
ORDER BY readmit_rate_pct DESC;


-- ----------------------------------------------------------------------------
-- QUERY 3 — IoT vital signs vs admission type (cross-dataset join)
-- Insight: Emergency admissions show higher heart rate, lower SpO2 and higher
--          temperature/BP — IoT vitals carry a strong signal for triage.
-- ----------------------------------------------------------------------------
SELECT
    p.admission_type,
    COUNT(DISTINCT p.patient_id) AS patients,
    ROUND(AVG(v.heart_rate), 1)  AS avg_heart_rate,
    ROUND(AVG(v.spo2), 1)        AS avg_spo2,
    ROUND(AVG(v.temperature), 2) AS avg_temp_c,
    ROUND(AVG(v.bp_systolic), 0) AS avg_systolic_bp
FROM vitals v
JOIN patients p
    ON v.patient_id = p.patient_id
GROUP BY p.admission_type
ORDER BY avg_heart_rate DESC;


-- ----------------------------------------------------------------------------
-- QUERY 4 — Daily occupancy & wait-time trend (time series)
-- Insight: Occupancy is stable (~62-66%) with daily variation; the time series
--          lets managers spot demand spikes and staff proactively.
-- date_parse converts the string timestamp to a real date for grouping.
-- ----------------------------------------------------------------------------
SELECT
    date(date_parse(snapshot_ts, '%Y-%m-%d %H:%i:%s')) AS day,
    ROUND(AVG(occupancy_rate) * 100, 1) AS avg_occupancy_pct,
    ROUND(AVG(avg_wait_minutes), 1)     AS avg_wait_min,
    SUM(patients_waiting)               AS total_waiting
FROM operations
GROUP BY date(date_parse(snapshot_ts, '%Y-%m-%d %H:%i:%s'))
ORDER BY day;
