#!/bin/bash
set -e

# Function to wait for PostgreSQL
wait_for_postgres() {
    echo "Waiting for PostgreSQL..."
    while ! nc -z ${Postgres_HOST:-host.docker.internal} 5432; do
        sleep 1
    done
    echo "PostgreSQL is ready!"
}

# Function to wait for Redis
wait_for_redis() {
    echo "Waiting for Redis..."
    while ! nc -z ${REDIS_HOST:-host.docker.internal} ${REDIS_PORT:-6379}; do
        sleep 1
    done
    echo "Redis is ready!"
}

# Wait for services
wait_for_postgres
wait_for_redis

# Run migrations if enabled
if [ "${RUN_MIGRATIONS}" = "true" ]; then
    echo "Running database migrations..."
    python manage.py migrate --noinput
fi

# Collect static files if enabled
if [ "${COLLECT_STATIC}" = "true" ]; then
    echo "Collecting static files..."
    python manage.py collectstatic --noinput
fi

# Create superuser if credentials are provided
if [ -n "${DJANGO_SUPERUSER_USERNAME}" ] && [ -n "${DJANGO_SUPERUSER_PASSWORD}" ] && [ -n "${DJANGO_SUPERUSER_EMAIL}" ]; then
    echo "Creating superuser..."
    python manage.py createsuperuser --noinput || true
fi

# Execute the main command
exec "$@" 