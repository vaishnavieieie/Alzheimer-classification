import requests  # to send HTTP requests

# Customer information
customer = {
    "FunctionalAssessment": 8.9,
    "ADL": 6.4,
    "MMSE": 13.4,
    "MemoryComplaints": 1,
    "BehavioralProblems": 1,
}

# The URL of the Flask API
url = 'http://localhost:9696/predict'

# Send the request
response = requests.post(url, json=customer)

# Check the status code and response
if response.status_code == 200:
    result = response.json()  # Parse the JSON response
    print(result)
else:
    print(f"Error: {response.status_code}")
    print(response.text)
