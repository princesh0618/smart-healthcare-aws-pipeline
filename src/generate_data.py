"""
Smart Healthcare - Synthetic Data Generator
============================================
Generates three synthetic, correlated datasets for the Cloud/Big Data pipeline:

  1. patients.csv     - one row per admission (clinical + outcome data)
  2. vitals.csv       - IoT patient-monitoring readings (many per patient)
  3. operations.csv   - hourly hospital operations snapshots per department

The data is 100% synthetic (no real people) and is intentionally built with
realistic correlations so the downstream Athena analytics show real patterns:
  - the Emergency dept has higher occupancy and longer waits
  - older patients and certain diagnoses are readmitted more often
  - abnormal vitals (low SpO2 / high heart rate) cluster around emergency cases

Usage:  python generate_data.py
Output: ./data/patients.csv, ./data/vitals.csv, ./data/operations.csv
"""

import numpy as np
import pandas as pd

# Reproducible output so your screenshots/queries are stable
RNG = np.random.default_rng(42)

# ----------------------------------------------------------------------------
# Configuration  (kept small to stay comfortably inside AWS Free Tier limits)
# ----------------------------------------------------------------------------
N_PATIENTS        = 5000
READINGS_PER_PT   = 10          # avg IoT readings per monitored patient
START_DATE        = "2025-01-01"
N_DAYS            = 90

DEPARTMENTS = ["Emergency", "Cardiology", "Pediatrics",
               "Oncology", "Orthopedics", "General"]

# Per-department behaviour: (occupancy bias, base wait minutes, readmit bias)
DEPT_PROFILE = {
    "Emergency":   dict(occ=0.85, wait=55, readmit=0.22),
    "Cardiology":  dict(occ=0.70, wait=30, readmit=0.18),
    "Pediatrics":  dict(occ=0.55, wait=20, readmit=0.08),
    "Oncology":    dict(occ=0.75, wait=25, readmit=0.20),
    "Orthopedics": dict(occ=0.60, wait=35, readmit=0.10),
    "General":     dict(occ=0.50, wait=25, readmit=0.12),
}

DIAGNOSES = ["Hypertension", "Diabetes", "Fracture", "Pneumonia",
             "Arrhythmia", "Infection", "Asthma", "Cancer-followup",
             "Chest-pain", "Routine-checkup"]

start = pd.Timestamp(START_DATE)


# ----------------------------------------------------------------------------
# 1) PATIENTS  -- one row per hospital admission
# ----------------------------------------------------------------------------
def make_patients():
    depts = RNG.choice(DEPARTMENTS, size=N_PATIENTS,
                       p=[0.28, 0.16, 0.14, 0.12, 0.14, 0.16])
    ages = RNG.integers(1, 95, size=N_PATIENTS)
    genders = RNG.choice(["M", "F"], size=N_PATIENTS)
    admit_offset_days = RNG.integers(0, N_DAYS, size=N_PATIENTS)
    admit_time = (start
                  + pd.to_timedelta(admit_offset_days, unit="D")
                  + pd.to_timedelta(RNG.integers(0, 24 * 60, size=N_PATIENTS), unit="m"))

    rows = []
    for i in range(N_PATIENTS):
        dept = depts[i]
        prof = DEPT_PROFILE[dept]
        is_emergency = (dept == "Emergency") or (RNG.random() < 0.12)

        # length of stay: emergency cases & older patients stay a bit longer
        base_los = RNG.poisson(3) + 1
        los = int(base_los + (ages[i] > 65) * RNG.poisson(2) + is_emergency * RNG.poisson(2))
        los = max(1, min(los, 30))
        discharge_time = admit_time[i] + pd.Timedelta(days=int(los))

        diagnosis = RNG.choice(DIAGNOSES)

        # readmission probability rises with age, dept profile and certain diagnoses
        p = prof["readmit"]
        p += 0.10 if ages[i] > 70 else 0.0
        p += 0.08 if diagnosis in ("Cancer-followup", "Arrhythmia", "Pneumonia") else 0.0
        readmitted = int(RNG.random() < min(p, 0.6))

        rows.append({
            "patient_id":     f"P{i:05d}",
            "age":            int(ages[i]),
            "gender":         genders[i],
            "department":     dept,
            "admission_type": "Emergency" if is_emergency else "Planned",
            "diagnosis":      diagnosis,
            "admission_ts":   admit_time[i].strftime("%Y-%m-%d %H:%M:%S"),
            "discharge_ts":   discharge_time.strftime("%Y-%m-%d %H:%M:%S"),
            "length_of_stay": los,
            "readmitted_30d": readmitted,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 2) VITALS  -- IoT device readings, several per patient
# ----------------------------------------------------------------------------
def make_vitals(patients):
    rows = []
    for _, pt in patients.iterrows():
        n = max(1, RNG.poisson(READINGS_PER_PT))
        admit = pd.Timestamp(pt["admission_ts"])
        emergency = pt["admission_type"] == "Emergency"

        for r in range(n):
            ts = admit + pd.Timedelta(minutes=int(RNG.integers(0, pt["length_of_stay"] * 24 * 60)))

            # baseline vitals, shifted toward "abnormal" for emergency cases
            hr   = RNG.normal(105 if emergency else 78, 12)
            spo2 = RNG.normal(93 if emergency else 97, 2)
            temp = RNG.normal(37.6 if emergency else 36.8, 0.4)
            sysbp = RNG.normal(140 if emergency else 120, 15)
            diabp = RNG.normal(90 if emergency else 78, 10)

            rows.append({
                "device_id":   f"DEV{RNG.integers(0, 200):03d}",
                "patient_id":  pt["patient_id"],
                "department":  pt["department"],
                "reading_ts":  ts.strftime("%Y-%m-%d %H:%M:%S"),
                "heart_rate":  round(float(np.clip(hr, 40, 180)), 1),
                "spo2":        round(float(np.clip(spo2, 80, 100)), 1),
                "temperature": round(float(np.clip(temp, 35, 41)), 1),
                "bp_systolic": int(np.clip(sysbp, 80, 200)),
                "bp_diastolic": int(np.clip(diabp, 50, 130)),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 3) OPERATIONS  -- hourly snapshot per department
# ----------------------------------------------------------------------------
def make_operations():
    rows = []
    for d in range(N_DAYS):
        for h in range(24):
            ts = start + pd.Timedelta(days=d, hours=h)
            # busier during daytime hours
            day_factor = 1.0 + 0.4 * np.sin((h - 6) / 24 * 2 * np.pi)
            for dept in DEPARTMENTS:
                prof = DEPT_PROFILE[dept]
                beds_total = {"Emergency": 40, "Cardiology": 30, "Pediatrics": 25,
                              "Oncology": 20, "Orthopedics": 25, "General": 50}[dept]
                occ_rate = np.clip(RNG.normal(prof["occ"] * day_factor, 0.08), 0.1, 0.99)
                occupied = int(beds_total * occ_rate)
                waiting = max(0, int(RNG.normal(occ_rate * 10, 3)))
                wait_min = max(0, int(RNG.normal(prof["wait"] * day_factor, 8)))
                rows.append({
                    "snapshot_ts":   ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "department":    dept,
                    "beds_total":    beds_total,
                    "beds_occupied": occupied,
                    "occupancy_rate": round(occupied / beds_total, 3),
                    "patients_waiting": waiting,
                    "avg_wait_minutes": wait_min,
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Generating patients ...")
    patients = make_patients()
    print("Generating vitals (IoT) ...")
    vitals = make_vitals(patients)
    print("Generating operations ...")
    operations = make_operations()

    patients.to_csv("data/patients.csv", index=False)
    vitals.to_csv("data/vitals.csv", index=False)
    operations.to_csv("data/operations.csv", index=False)

    print("\nDone. Row counts:")
    print(f"  patients.csv   : {len(patients):>7,} rows")
    print(f"  vitals.csv     : {len(vitals):>7,} rows")
    print(f"  operations.csv : {len(operations):>7,} rows")
    print(f"\n  Readmission rate: {patients['readmitted_30d'].mean():.1%}")
    print(f"  Emergency share : {(patients['admission_type']=='Emergency').mean():.1%}")
