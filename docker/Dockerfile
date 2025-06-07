# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # PostgreSQL client libraries
    libpq-dev \
    # Build essentials for Python packages
    gcc \
    g++ \
    make \
    # pkg-config for building packages
    pkg-config \
    # Required for some Python packages
    libffi-dev \
    libssl-dev \
    # Required for h5py
    libhdf5-dev \
    hdf5-tools \
    # Required for matplotlib
    libfreetype6-dev \
    libpng-dev \
    # Required for lxml
    libxml2-dev \
    libxslt-dev \
    # Required for Pillow
    libjpeg-dev \
    zlib1g-dev \
    # Git for any git dependencies
    git \
    # Network utilities for health checks
    netcat-traditional \
    curl \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements/ ./requirements/

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    # Install critical dependencies first to control versions
    pip install --no-cache-dir "numpy<2.0.0" "h5py==3.10.0" && \
    # Install PyTorch with CPU-only builds (may upgrade numpy but that's ok)
    pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu && \
    # Install all other dependencies
    pip install --no-cache-dir -c requirements/constraints.txt -r requirements/docker.txt

# Copy project files
COPY . /app/

# Copy and set permissions for entrypoint script
COPY docker/docker-entrypoint.sh /usr/local/bin/

RUN sed -i 's/\r$//g' /usr/local/bin/docker-entrypoint.sh

RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Create necessary directories
RUN mkdir -p logs cache saved_models models

# Expose port for Django/Daphne
EXPOSE 8000

# Create a non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Set the entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]

# Default command
CMD ["python", "run/run.py"] 