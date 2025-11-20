#!/usr/bin/env python3
"""
Script to inspect an Optuna SQLite database and check study configuration.
"""
import sqlite3
import json
from pathlib import Path

# Path to the study database
db_path = Path("results/de_uvtzrh_scf_NUTS2/study_cnn_de_uvtzrh_scf_NUTS2_20251106/study_cnn_de_uvtzrh_scf_NUTS2_20251106.db")

print(f"Inspecting database: {db_path}")

# Connect to database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get study information
print("\n=== STUDY INFORMATION ===")
cursor.execute("SELECT study_id, study_name FROM studies")
studies = cursor.fetchall()
for study_id, study_name in studies:
    print(f"Study ID: {study_id}")
    print(f"Study Name: {study_name}")

# Get best trial information - values are in separate table
print("\n=== BEST TRIAL ===")
cursor.execute("""
    SELECT t.trial_id, t.number, t.state, tv.value
    FROM trials t
    JOIN trial_values tv ON t.trial_id = tv.trial_id
    WHERE t.state = 'COMPLETE'
    ORDER BY tv.value ASC
    LIMIT 1
""")

best_trial = cursor.fetchone()
if best_trial:
    trial_id, trial_number, state, value = best_trial
    print(f"Trial ID: {trial_id}")
    print(f"Trial Number: {trial_number}")
    print(f"State: {state}")
    print(f"Best Value: {value}")
    
    # Get parameters for this trial
    print(f"\n=== PARAMETERS FOR BEST TRIAL ===")
    cursor.execute("""
        SELECT param_name, param_value, distribution_json
        FROM trial_params
        WHERE trial_id = ?
    """, (trial_id,))
    
    params = cursor.fetchall()
    for param_name, param_value, distribution in params:
        print(f"{param_name}: {param_value}")
        if distribution:
            dist_info = json.loads(distribution)
            print(f"  Distribution: {dist_info}")
    
    # Check for any user attributes that might contain config info
    cursor.execute("""
        SELECT key, value_json
        FROM trial_user_attributes
        WHERE trial_id = ?
    """, (trial_id,))
    
    user_attrs = cursor.fetchall()
    if user_attrs:
        print(f"\n=== USER ATTRIBUTES ===")
        for key, value_json in user_attrs:
            print(f"{key}: {value_json}")

# Count trials by state
print(f"\n=== TRIAL STATISTICS ===")
cursor.execute("SELECT state, COUNT(*) FROM trials GROUP BY state")
states = cursor.fetchall()
for state, count in states:
    print(f"{state}: {count} trials")

conn.close()