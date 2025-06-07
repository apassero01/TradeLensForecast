# TradeLens Docker Setup Guide

This directory contains all Docker-related files for running the TradeLens backend application.

## 📋 Prerequisites

Before you begin, ensure you have the following installed on your system:

1. **Docker Desktop** (includes Docker Engine and Docker Compose)
   - macOS: [Download Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/)
   - Windows: [Download Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
   - Linux: [Install Docker Engine](https://docs.docker.com/engine/install/)

2. **PostgreSQL** (running on host machine)
   - Default port: 5432
   - Database name: `tradelens` (must be created)
   - User: `postgres`

3. **Redis** (running on host machine)
   - Default port: 6379

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd TradeLensForcast
```

### 2. Set Up Environment Variables
```bash
# From the project root
cp docker/docker.env.example docker/.env

# Or if you're already in the docker directory
cd docker
cp docker.env.example .env
```

Edit `docker/.env` file with your settings:
- Update `POSTGRES_PASSWORD` if different from default
- Add any API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)

### 3. Start PostgreSQL and Redis
Make sure PostgreSQL and Redis are running on your host machine:

```bash
# macOS (using Homebrew)
brew services start postgresql
brew services start redis

# Linux
sudo systemctl start postgresql
sudo systemctl start redis
```

### 4. Create the Database
```bash
psql -U postgres -c "CREATE DATABASE tradelens;"
```

### 5. Build and Run the Application
```bash
# From the project root
./docker.sh up --build

# Or from the docker directory
cd docker
docker-compose up --build

# Or run in detached mode (background)
./docker.sh up -d --build
```

The first build will take 5-10 minutes to download and install all dependencies. Subsequent runs will be much faster due to Docker's layer caching.

### 6. Access the Application
- Backend API: http://localhost:8000
- Admin Interface: http://localhost:8000/admin (if configured)

## 📁 File Structure

```
docker/
├── README.md              # This file
├── Dockerfile            # Docker image definition
├── docker-compose.yml    # Service orchestration
├── docker-entrypoint.sh  # Container startup script
├── docker.env.example    # Environment variables template
└── .dockerignore        # Files to exclude from Docker build
```

## 🔧 Common Operations

### Starting and Stopping

```bash
# Start services
docker-compose up

# Stop services (Ctrl+C or)
docker-compose down

# Stop and remove volumes (careful - deletes data)
docker-compose down -v
```

### Viewing Logs

```bash
# View all logs
docker-compose logs

# Follow logs in real-time
docker-compose logs -f

# View specific service logs
docker-compose logs tradelens-backend
```

### Running Django Commands

```bash
# Run migrations
docker-compose exec tradelens-backend python manage.py migrate

# Create superuser
docker-compose exec tradelens-backend python manage.py createsuperuser

# Access Django shell
docker-compose exec tradelens-backend python manage.py shell

# Collect static files
docker-compose exec tradelens-backend python manage.py collectstatic
```

### Rebuilding After Changes

```bash
# After changing requirements
docker-compose up --build

# Force rebuild without cache
docker-compose build --no-cache
```

## 🐛 Troubleshooting

### Cannot Connect to PostgreSQL/Redis

1. **Check if services are running:**
   ```bash
   # PostgreSQL
   pg_isready -h localhost -p 5432
   
   # Redis
   redis-cli ping
   ```

2. **Check PostgreSQL configuration:**
   - Edit `postgresql.conf`: set `listen_addresses = '*'`
   - Edit `pg_hba.conf`: add `host all all 0.0.0.0/0 md5`
   - Restart PostgreSQL

3. **For macOS users:**
   - `host.docker.internal` should work automatically
   - If not, try using your machine's IP address

### Build Failures

1. **Package version conflicts:**
   - Check `requirements/*.txt` files for incompatible versions
   - See `requirements/constraints.txt` for version pins

2. **Platform-specific issues:**
   - Some packages may not have ARM64 wheels (Apple Silicon)
   - The Dockerfile includes necessary build tools

### Application Errors

1. **Missing modules:**
   - Add to appropriate `requirements/*.txt` file
   - Rebuild: `docker-compose up --build`

2. **Code changes not reflected:**
   - Code is mounted as volume, changes should be immediate
   - Restart container: `docker-compose restart`

## 🔄 Development Workflow

### Making Code Changes

1. **Python code changes:**
   - No rebuild needed!
   - Just restart the container: `docker-compose restart`

2. **Adding new dependencies:**
   - Add to appropriate file in `requirements/`
   - Rebuild: `docker-compose up --build`

3. **Changing Docker configuration:**
   - Modify `Dockerfile` or `docker-compose.yml`
   - Rebuild: `docker-compose up --build`

### Using with PyCharm

For debugging with PyCharm, see the remote debugging setup in `run/DOCKER_README.md`.

## 🏗️ Architecture

The Docker setup runs:
- **Django/Daphne**: ASGI server for HTTP and WebSocket
- **Celery Worker**: Background task processing
- **Volume Mounts**: Code, logs, models, and cache directories

External services (on host):
- **PostgreSQL**: Main database
- **Redis**: Cache and Celery broker

## 📝 Environment Variables

Key variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_PASSWORD` | PostgreSQL password | `postgres` |
| `POSTGRES_HOST` | PostgreSQL host | `host.docker.internal` |
| `REDIS_HOST` | Redis host | `host.docker.internal` |
| `DEBUG` | Django debug mode | `True` |
| `RUN_MIGRATIONS` | Auto-run migrations | `true` |
| `DEBUG_MODE` | Run Celery in debug | `0` |

## 🚨 Important Notes

1. **Environment File Location**: The `.env` file must be in the `docker/` directory, not the project root
2. **Data Persistence**: Logs, models, and cache are persisted in local directories
3. **Security**: Change `SECRET_KEY` and passwords for production
4. **Performance**: Adjust CPU/memory limits in `docker-compose.yml` as needed
5. **Networking**: Uses `host.docker.internal` to connect to host services

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Django Docker Best Practices](https://docs.djangoproject.com/en/4.2/howto/deployment/)

## 🆘 Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Review logs: `docker-compose logs`
3. Ensure all prerequisites are installed
4. Check that PostgreSQL and Redis are accessible from Docker

For more detailed information about the Docker setup, see `run/DOCKER_README.md`. 