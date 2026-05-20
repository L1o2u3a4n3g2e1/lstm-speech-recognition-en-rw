FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Set working directory to web_app
WORKDIR /app/web_app

# Expose port
EXPOSE 5000

# Run the app
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:app"]
