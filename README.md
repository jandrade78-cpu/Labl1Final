# Student Service Request SLA Breach Predictor (Streamlit)

Deployment of the Section B midterm data-mining model (Systems Analysis and Design, BSIS).
Predicts `SLA_Breach` for student service requests and supports staff review. **Decision support only.**

## Structure (flat)
app.py | config.py | features.py | data_processing.py | exploratory_analysis.py | prediction.py | dashboard.py | system_design.py
train_model.py | make_diagram.py | requirements.txt | trained_model.joblib | use_case_diagram.png | student_service_requests.csv

## Run locally
    pip install -r requirements.txt
    python train_model.py
    streamlit run app.py

## Deploy
Push to GitHub -> https://share.streamlit.io -> New app -> pick repo/branch -> main file `app.py`.

## Data and ethics
Use synthetic or authorized data only. Never commit real student records or credentials.
Post-outcome fields (`Processing_Days`, `Status`, `Reopened`, `Satisfaction_Score`) are removed to prevent leakage.
Predictions never authorize denying or delaying a service; staff record the final decision.

## Limitations
Small dataset (~350 labeled rows), uncalibrated probabilities, no per-user login, decisions stored per session only.
