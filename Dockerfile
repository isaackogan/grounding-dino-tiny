FROM runpod/pytorch:3.10-2.1.2-11.8

WORKDIR /app

# Copy project files
COPY server.py .
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose API port
EXPOSE 8000

# Start server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
