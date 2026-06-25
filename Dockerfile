FROM python:3.10-slim

# Prevent python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Set working directory inside container
WORKDIR /app

# Install system dependencies required by OpenCV, Git, and Git LFS
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1-mesa-glx \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Copy the production requirements and install dependencies
COPY requirements_prod.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_prod.txt

# Copy all project files into the container
COPY . /app/

# Ensure the uploads directory exists with correct permissions
RUN mkdir -p /app/app/static/uploads && chmod -R 777 /app/app/static/uploads

# Expose port 7860 as required by Hugging Face Spaces
EXPOSE 7860

# Run the Flask app via Gunicorn WSGI server
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "180", "app.main:app"]
