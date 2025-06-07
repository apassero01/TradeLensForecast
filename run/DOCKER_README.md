# TradeLens Docker Setup

This guide explains how to run the TradeLens backend in a Docker container while connecting to PostgreSQL and Redis running on the host machine.

## Prerequisites

1. Docker and Docker Compose installed
2. PostgreSQL running on host machine (port 5432)
3. Redis running on host machine (port 6379)
4. The database `tradelens` should exist in PostgreSQL

## Quick Start

1. **Copy the environment file:**
   ```bash
   cp docker.env.example .env
   ```

2. **Edit `.env` file with your settings:**
   - Update `POSTGRES_PASSWORD` if different from default
   - Add any API keys you need

3. **Build and run the container:**
   ```bash
   docker-compose up --build
   ```

   Or run in detached mode:
   ```bash
   docker-compose up -d --build
   ```

4. **Access the application:**
   - Backend API: http://localhost:8000
   - WebSocket: ws://localhost:8000/ws/

## Docker Commands

### Build the image:
```bash
docker build -t tradelens-backend .
```

### Run without docker-compose:
```bash
docker run -p 8000:8000 \
  -e DJANGO_SETTINGS_MODULE=TradeLens.settings_docker \
  -e Postgres_HOST=host.docker.internal \
  -e REDIS_HOST=host.docker.internal \
  --add-host=host.docker.internal:host-gateway \
  tradelens-backend
```

### View logs:
```bash
docker-compose logs -f
```

### Stop the container:
```bash
docker-compose down
```

### Remove volumes (careful - this deletes data):
```bash
docker-compose down -v
```

## Configuration

### Environment Variables

Key environment variables (see `docker.env.example`):

- `DJANGO_SETTINGS_MODULE`: Use `TradeLens.settings_docker` for Docker
- `Postgres_HOST`: Set to `host.docker.internal` to connect to host PostgreSQL
- `REDIS_HOST`: Set to `host.docker.internal` to connect to host Redis
- `RUN_MIGRATIONS`: Set to `true` to auto-run migrations on startup
- `COLLECT_STATIC`: Set to `true` to collect static files on startup
- `DEBUG_MODE`: Set to `1` for debug mode (single-threaded Celery)

### Volumes

The following directories are mounted as volumes:
- `./logs`: Application logs
- `./cache`: Cache files
- `./saved_models`: Saved ML models
- `./models`: Model files

### Resource Limits

Default resource limits in docker-compose.yml:
- CPU: 4 cores (limit), 2 cores (reservation)
- Memory: 8GB (limit), 4GB (reservation)

Adjust these in `docker-compose.yml` based on your needs.

## Troubleshooting

### Connection Issues

1. **Cannot connect to PostgreSQL/Redis:**
   - Ensure PostgreSQL and Redis are running on the host
   - Check that they're listening on all interfaces, not just localhost
   - For PostgreSQL, check `postgresql.conf`:
     ```
     listen_addresses = '*'
     ```
   - For Redis, check `redis.conf`:
     ```
     bind 0.0.0.0
     ```

2. **Linux users:** If `host.docker.internal` doesn't work:
   - The docker-compose.yml includes `extra_hosts` configuration
   - Alternatively, use the host's IP address directly

### Permission Issues

If you encounter permission errors:
```bash
# Fix ownership of mounted volumes
sudo chown -R 1000:1000 ./logs ./cache ./saved_models ./models
```

### Database Migrations

To run migrations manually:
```bash
docker-compose exec tradelens-backend python manage.py migrate
```

To create a superuser manually:
```bash
docker-compose exec tradelens-backend python manage.py createsuperuser
```

### Debugging

To run in debug mode with interactive shell:
```bash
docker-compose run --rm tradelens-backend bash
```

To check if services are accessible from container:
```bash
# Check PostgreSQL
docker-compose exec tradelens-backend nc -zv host.docker.internal 5432

# Check Redis
docker-compose exec tradelens-backend nc -zv host.docker.internal 6379
```

## Production Considerations

For production deployment:

1. **Security:**
   - Change `SECRET_KEY` in settings
   - Set `DEBUG=False`
   - Configure `ALLOWED_HOSTS` properly
   - Use environment-specific settings

2. **Performance:**
   - Remove source code volume mount
   - Use production-grade ASGI server configuration
   - Adjust worker counts based on CPU cores
   - Configure proper logging

3. **Database:**
   - Use connection pooling
   - Configure proper database credentials
   - Consider using a managed database service

4. **Monitoring:**
   - Add health check endpoints
   - Configure logging aggregation
   - Set up monitoring and alerting

## Development Tips

1. **Hot Reload:** The current setup mounts source code as read-only. For development with hot reload:
   ```yaml
   volumes:
     - ./:/app  # Remove :ro for read-write access
   ```

2. **Debugging:** Set `DEBUG_MODE=1` to run Celery in single-threaded mode for easier debugging

3. **Shell Access:**
   ```bash
   docker-compose exec tradelens-backend python manage.py shell
   ```

4. **Database Shell:**
   ```bash
   docker-compose exec tradelens-backend python manage.py dbshell
   ``` 