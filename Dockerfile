FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose ports for Streamlit and FastAPI
EXPOSE 8501 8000

# By default, run Streamlit. 
# You can override the command to run FastAPI: CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
