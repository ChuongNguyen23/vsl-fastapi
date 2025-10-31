# Use Python 3.10 (Debian-based image)
FROM python:3.10-slim

# Install system dependencies required by Mediapipe & OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    unzip \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements-server.txt .

# Upgrade pip & install dependencies
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements-server.txt

# Copy the rest of your project
COPY . .

# Expose the port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
