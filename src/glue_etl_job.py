"""
========================================================================
Glue ETL Job — Smart Healthcare Pipeline
Bronze (raw CSV)  ->  Silver (cleaned Parquet)  ->  Gold (curated Parquet)
========================================================================
What this job does:
  1. Reads the three Bronze CSV datasets from S3 (patients, vitals, operations)
  2. SILVER: cleans/types each dataset and writes it as Parquet
  3. GOLD: builds a small curated, analytics-ready table and writes it as Parquet

Run as a Glue ETL job (Spark) using the GlueHealthcareRole.
Replace BUCKET below if your bucket name is different.
"""
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
args = getResolvedOptions(sys.argv, ["JOB_NAME"])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

BUCKET = "smart-hospital-datalake-prince"
BRONZE = f"s3://{BUCKET}/bronze"
SILVER = f"s3://{BUCKET}/silver"
GOLD   = f"s3://{BUCKET}/gold"

# ----------------------------------------------------------------------
# 1. READ Bronze CSVs (header=true, let Spark infer types)
# ----------------------------------------------------------------------
patients = (spark.read.option("header", True).option("inferSchema", True)
            .csv(f"{BRONZE}/patients/"))
vitals = (spark.read.option("header", True).option("inferSchema", True)
          .csv(f"{BRONZE}/vitals/"))
operations = (spark.read.option("header", True).option("inferSchema", True)
              .csv(f"{BRONZE}/operations/"))

# ----------------------------------------------------------------------
# 2. SILVER — clean + type, write Parquet
# ----------------------------------------------------------------------
# patients: drop rows with no id, cast types, parse timestamps
patients_s = (patients
    .dropna(subset=["patient_id"])
    .withColumn("age", F.col("age").cast("int"))
    .withColumn("length_of_stay", F.col("length_of_stay").cast("int"))
    .withColumn("readmitted_30d", F.col("readmitted_30d").cast("int"))
    .withColumn("admission_ts", F.to_timestamp("admission_ts", "yyyy-MM-dd HH:mm:ss"))
    .withColumn("discharge_ts", F.to_timestamp("discharge_ts", "yyyy-MM-dd HH:mm:ss"))
    .dropDuplicates())

vitals_s = (vitals
    .dropna(subset=["patient_id"])
    .withColumn("heart_rate", F.col("heart_rate").cast("double"))
    .withColumn("spo2", F.col("spo2").cast("double"))
    .withColumn("temperature", F.col("temperature").cast("double"))
    .withColumn("bp_systolic", F.col("bp_systolic").cast("int"))
    .withColumn("bp_diastolic", F.col("bp_diastolic").cast("int"))
    .withColumn("reading_ts", F.to_timestamp("reading_ts", "yyyy-MM-dd HH:mm:ss"))
    .dropDuplicates())

operations_s = (operations
    .withColumn("beds_total", F.col("beds_total").cast("int"))
    .withColumn("beds_occupied", F.col("beds_occupied").cast("int"))
    .withColumn("occupancy_rate", F.col("occupancy_rate").cast("double"))
    .withColumn("patients_waiting", F.col("patients_waiting").cast("int"))
    .withColumn("avg_wait_minutes", F.col("avg_wait_minutes").cast("int"))
    .withColumn("snapshot_ts", F.to_timestamp("snapshot_ts", "yyyy-MM-dd HH:mm:ss"))
    .withColumn("snapshot_date", F.to_date("snapshot_ts"))
    .dropDuplicates())

# write Silver as Parquet (overwrite so the job is re-runnable)
patients_s.write.mode("overwrite").parquet(f"{SILVER}/patients/")
vitals_s.write.mode("overwrite").parquet(f"{SILVER}/vitals/")
# operations partitioned by department (useful, matches the brief's "partition by department")
operations_s.write.mode("overwrite").partitionBy("department").parquet(f"{SILVER}/operations/")

# ----------------------------------------------------------------------
# 3. GOLD — curated, analytics-ready tables
# ----------------------------------------------------------------------
# 3a. department_kpis: one row per department with the key metrics
dept_kpis = (operations_s.groupBy("department").agg(
                F.round(F.avg("occupancy_rate") * 100, 1).alias("avg_occupancy_pct"),
                F.round(F.avg("avg_wait_minutes"), 1).alias("avg_wait_min"),
                F.max("patients_waiting").alias("peak_queue")))

# 3b. readmission_by_age: rate by department + age band
patients_age = patients_s.withColumn("age_group",
    F.when(F.col("age") < 18, "0-17")
     .when(F.col("age") <= 40, "18-40")
     .when(F.col("age") <= 65, "41-65")
     .otherwise("65+"))
readmit_by_age = (patients_age.groupBy("department", "age_group").agg(
                    F.count("*").alias("total_patients"),
                    F.sum("readmitted_30d").alias("readmissions"),
                    F.round(F.avg("readmitted_30d") * 100, 1).alias("readmit_rate_pct")))

# 3c. daily_trend: occupancy/wait per day
daily_trend = (operations_s.groupBy("snapshot_date").agg(
                  F.round(F.avg("occupancy_rate") * 100, 1).alias("avg_occupancy_pct"),
                  F.round(F.avg("avg_wait_minutes"), 1).alias("avg_wait_min"),
                  F.sum("patients_waiting").alias("total_waiting"))
               .orderBy("snapshot_date"))

dept_kpis.write.mode("overwrite").parquet(f"{GOLD}/department_kpis/")
readmit_by_age.write.mode("overwrite").parquet(f"{GOLD}/readmission_by_age/")
daily_trend.write.mode("overwrite").parquet(f"{GOLD}/daily_trend/")

print("ETL complete: Silver and Gold zones written as Parquet.")
job.commit()
