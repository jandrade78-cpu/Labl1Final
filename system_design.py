import streamlit as st

FRS = [
    ("FR1", "Upload a CSV of service requests and validate schema, nulls and duplicates.", "Dataset page"),
    ("FR2", "Show descriptive statistics, SLA class distribution and interactive charts.", "Data Analysis page"),
    ("FR3", "Predict SLA breach probability for a new request using the saved pipeline.", "Prediction page"),
    ("FR4", "Rank requests into a risk/review queue with recommended follow-up.", "Dashboard page"),
    ("FR5", "Let authorized staff record and export a final human decision with model version.", "Dashboard page"),
]
NFRS = [
    ("NFR1 Performance", "Model loads once via cache; predictions return in under 2 seconds."),
    ("NFR2 Security/Privacy", "No real student records or credentials in the repository; synthetic or authorized data only."),
    ("NFR3 Usability", "Six-page sidebar navigation with clear error messages for invalid input."),
]

WORKFLOW = """
digraph G {
  rankdir=LR; node [shape=box, style=rounded, fontsize=11];
  A [label="Student submits\\nservice request"];
  B [label="Request data\\n(CSV / form)"];
  C [label="Validate &\\nclean"];
  D [label="Saved pipeline\\npredicts breach risk"];
  E [label="Risk queue\\n(Low/Med/High)"];
  F [label="Staff review", shape=diamond];
  G [label="Record decision\\n& follow-up"];
  H [label="Request handled\\nwithin SLA"];
  A -> B -> C -> D -> E -> F;
  F -> G [label="Medium/High"];
  F -> H [label="Low / override"];
  G -> H;
}
"""


def render():
    st.header("System Design (SAD Integration)")
    st.subheader("Functional requirements")
    st.table({"ID": [f[0] for f in FRS], "Requirement": [f[1] for f in FRS], "Implemented in": [f[2] for f in FRS]})
    st.subheader("Non-functional requirements")
    st.table({"ID": [n[0] for n in NFRS], "Requirement": [n[1] for n in NFRS]})

    st.subheader("Actors and permissions")
    st.markdown("""
| Actor | Can do |
|---|---|
| **Student** | Submits service requests (outside this app); cannot view predictions |
| **Student Services Staff** | Upload data, run predictions, view dashboard, **record final decisions** |
| **Office Head / Supervisor** | View dashboard and risk queue, review/override decisions |
| **System Administrator** | Maintains model version, deployment and data policy |
| **Predictive Model** | Supporting system only: outputs risk, never decides |
""")
    st.subheader("Updated use case diagram")
    st.image("use_case_diagram.png", width="stretch")
    st.subheader("Decision workflow / analytical process flow")
    st.graphviz_chart(WORKFLOW)
    st.caption("Predictions are decision-support outputs. Humans make all final decisions.")
