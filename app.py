import streamlit as st
import requests  # to send HTTP requests

st.title("Alzheimer Prediction")
# Hero section 2 columns
col1, col2 = st.columns([1, 2])
with col1:
    st.image("image.png")
with col2:
    # Description of input fields and app
    st.write("This app predicts if someone has Alzheimers using the following fields. These fields were chosen by performing feature selection on https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset.")
    st.write("- Functional Assessment: A measure of the patient's ability to perform daily activities")
    st.write("- ADL: Activities of Daily Living")
    st.write("- MMSE: Mini-Mental State Examination")
    st.write("- Memory Complaints: Whether the patient has memory complaints")
    st.write("- Behavioral Problems: Whether the patient has behavioral problems")
    st.write("")

# Input fields
st.write("Please enter the following information:")
FunctionalAssessment = st.number_input("Functional Assessment", min_value=0.0, max_value=10.0, value=8.9)
ADL = st.number_input("ADL", min_value=0.0, max_value=10.0, value=6.4)
MMSE = st.number_input("MMSE", min_value=0.0, max_value=30.0, value=13.4)

col1, col2 = st.columns(2)
with col1:
    MemoryComplaints = st.checkbox("Memory Complaints")
with col2:
    BehavioralProblems = st.checkbox("Behavioral Problems")

# Convert the checkbox values to integers
MemoryComplaints = int(MemoryComplaints)
BehavioralProblems = int(BehavioralProblems)

# The URL of the Flask API
url = 'http://localhost:9696/predict'

# Send the request
if st.button("Predict"):
    customer = {
        "FunctionalAssessment": FunctionalAssessment,
        "ADL": ADL,
        "MMSE": MMSE,
        "MemoryComplaints": MemoryComplaints,
        "BehavioralProblems": BehavioralProblems,
    }
    response = requests.post(url, json=customer)

    # Check the status code and response
    st.write("Prediction Results:")
    if response.status_code == 200:
        result = response.json()  # Parse the JSON response
        for model, prediction in result.items():
            st.write(f"{model}: {prediction['pred']}")
            prob=prediction['prob'] if prediction['pred']==1 else 1-prediction['prob']
            # Confidence level
            st.progress(prob)


    else:
        st.write(f"Error: {response.status_code}")
        st.write(response.text)