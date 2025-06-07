# PyCharm Remote Debugging with Docker

This guide explains how to set up and use PyCharm remote debugging with your TradeLens Docker container.

## Prerequisites

1. **PyCharm Professional** (Community edition doesn't support remote debugging)
2. **Docker and Docker Compose** installed
3. **TradeLens project** opened in PyCharm

## Setup Instructions

### 1. Configure PyCharm Remote Debug Configuration

1. In PyCharm, go to **Run** → **Edit Configurations...**
2. Click the **+** button and select **Python Debug Server**
3. Configure the debug server:
   - **Name**: `TradeLens Docker Debug`
   - **IDE host name**: `localhost` (or your machine's IP if Docker is on a different machine)
   - **Port**: `5678`
   - **Path mappings**: 
     - Local path: `/path/to/your/TradeLensForcast` (your project root)
     - Remote path: `/app`

### 2. Start the Debug Server in PyCharm

1. Select your debug configuration from the dropdown
2. Click the **Debug** button (bug icon) to start listening for connections
3. PyCharm will show "Waiting for process connection..." in the debug console

### 3. Run Docker Container in Debug Mode

Choose one of these methods:

#### Method A: Using docker-compose with debug override
```bash
# From the project root directory
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.debug.yml up --build
```

#### Method B: Set environment variables manually
```bash
# Set debug environment variables
export PYCHARM_DEBUG=true
export DEBUG_MODE=1

# Run with regular docker-compose
docker-compose -f docker/docker-compose.yml up --build
```

#### Method C: Using docker run directly
```bash
docker build -f docker/Dockerfile -t tradelens-debug .

docker run -it --rm \
  -p 8000:8000 -p 5678:5678 \
  -e PYCHARM_DEBUG=true \
  -e DEBUG_MODE=1 \
  -e PYCHARM_DEBUG_HOST=host.docker.internal \
  -e PYCHARM_DEBUG_PORT=5678 \
  -v $(pwd):/app \
  --add-host host.docker.internal:host-gateway \
  tradelens-debug
```

## How It Works

### Debug Mode Features

When `DEBUG_MODE=1` is set:
- **Single-threaded execution**: Celery runs with `concurrency=1` and `pool=solo`
- **In-process execution**: Both Daphne and Celery run in the same process
- **Full debugging support**: All breakpoints and step-through debugging work

### Connection Flow

1. **Container starts** and `run.py` executes
2. **PyCharm debugger connects** to your IDE via `pydevd_pycharm.settrace()`
3. **Breakpoints activate** - you can now debug your Django views, Celery tasks, etc.
4. **Code changes** are reflected immediately (volume mounted)

## Debugging Tips

### Setting Breakpoints

1. **Django Views**: Set breakpoints in your view functions
2. **Celery Tasks**: Set breakpoints in your `@shared_task` functions
3. **Models**: Set breakpoints in model methods
4. **Middleware**: Set breakpoints in custom middleware

### Common Issues and Solutions

#### "Failed to connect to PyCharm debugger"
- Ensure PyCharm debug server is running and listening on port 5678
- Check that port 5678 is not blocked by firewall
- Verify `host.docker.internal` resolves correctly

#### Breakpoints not hitting
- Ensure you're using the debug docker-compose configuration
- Check that `DEBUG_MODE=1` is set
- Verify path mappings in PyCharm debug configuration

#### Performance issues
- Debug mode runs single-threaded for debugging compatibility
- For performance testing, use regular mode without `DEBUG_MODE=1`

#### Container exits immediately
- Check container logs: `docker-compose logs tradelens-backend`
- Ensure all required environment variables are set
- Verify database and Redis connections

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PYCHARM_DEBUG` | `false` | Enable PyCharm remote debugging |
| `PYCHARM_DEBUG_HOST` | `host.docker.internal` | PyCharm debug server host |
| `PYCHARM_DEBUG_PORT` | `5678` | PyCharm debug server port |
| `DEBUG_MODE` | `0` | Enable single-threaded debug mode |

## Production Notes

⚠️ **Never enable debugging in production!**

- Debug mode significantly impacts performance
- Remote debugging exposes your application internally
- Always use the regular `docker-compose.yml` for production deployments

## Troubleshooting

### Check Debug Connection
```bash
# Inside the container, verify the debugger connection
docker exec -it tradelens-backend python -c "
import socket
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('host.docker.internal', 5678))
    print('Debug port reachable' if result == 0 else 'Debug port not reachable')
    sock.close()
except Exception as e:
    print(f'Connection test failed: {e}')
"
```

### View Container Logs
```bash
# View real-time logs
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.debug.yml logs -f tradelens-backend

# View specific service logs
docker logs tradelens-backend
```

### Restart Debug Session
```bash
# Stop containers
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.debug.yml down

# Rebuild and restart
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.debug.yml up --build
``` 