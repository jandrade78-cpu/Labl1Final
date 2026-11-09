"""Schema of the MIDTERM model (student_service_requests.csv).
Target is derived exactly as in the midterm: SLA_Breach = 1 when Processing_Days > SLA_Days
(only for requests that have a Processing_Days value, i.e. completed ones)."""

TARGET = "SLA_Breach"
ID_COL = "Request_ID"

# Columns the uploaded CSV must contain (raw midterm columns)
RAW_REQUIRED = ["Service_Type", "Requester_Type", "Submission_Date", "Requested_Channel", "Priority",
                "Assigned_Office", "Queue_Length", "Required_Documents_Complete", "SLA_Days"]

# Model inputs (known at submission/triage time)
CATEGORICAL = ["Service_Type", "Requester_Type", "Requested_Channel", "Priority", "Assigned_Office",
               "Required_Documents_Complete", "Submit_Weekday"]
NUMERIC = ["Queue_Length", "SLA_Days", "Submit_Month"]
FEATURES = CATEGORICAL + NUMERIC

# Only known AFTER handling -> leakage, never used as inputs
LEAKAGE_COLUMNS = ["Processing_Days", "Status", "Reopened", "Satisfaction_Score"]

MODEL_PATH = "trained_model.joblib"
SAMPLE_DATA_PATH = "student_service_requests.csv"
