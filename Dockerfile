# Build image using python container
# Use a lightweight Python base image
FROM python:3.12

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory to /app
WORKDIR /app

#install dependencies in docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files to docker
COPY . .

# Command to run the application with Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:4000", "wsgi:app"]
