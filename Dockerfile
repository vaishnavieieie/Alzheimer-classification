# Base image
FROM python:3.9-slim

# Working directory
WORKDIR /app

# Copy
COPY app.py /app/
COPY flask_api.py /app/
COPY random_forest_best_model.pkl /app/
COPY random_forest_model.pkl /app/
COPY gradient_boosting_model.pkl /app/
COPY adaboost_model.pkl /app/
COPY requirements.txt /app/
COPY image.png /app/


# Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose ports for Streamlit and Flask
EXPOSE 8501 5000

# Command to run both services
CMD ["sh", "-c", "streamlit run app.py & python flask_api.py"]
