FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY api/app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install package in editable mode
RUN pip install -e .

# Expose port
EXPOSE 8089

# Default command
CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8089"]
