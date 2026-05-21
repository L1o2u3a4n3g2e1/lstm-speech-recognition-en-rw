FROM python:3.11.9-slim
# v3 cache bust - force full rebuild

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

            # Expose port
            EXPOSE 5000

            # Run the app from the web_app subdirectory
            CMD ["gunicorn", "--chdir", "/app/web_app", "-w", "1", "-b", "0.0.0.0:5000", "app:app"]
