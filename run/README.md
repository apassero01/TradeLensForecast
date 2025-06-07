# TradeLens Runner - Duplicate Worker Prevention

This directory contains hardened launcher scripts that prevent the duplicate Celery worker issue described in the post-mortem report.

## Files

- **`run.py`** - Main development launcher (recommended)
- **`celery_worker.py`** - Standalone Celery worker launcher

## Key Features

### 🛡️ Duplicate Prevention
- **Environment Guard**: Uses `TL_CELERY_STARTED` environment variable to prevent multiple instances
- **Process Cleanup**: Automatically detects and terminates existing Celery workers on startup
- **Unique Node Names**: Each worker gets a unique name like `api<pid>@<hostname>`

### 📊 Monitoring & Logging
- **Health Checks**: Verifies exactly one worker is running after startup
- **Per-Worker Logs**: Each worker writes to `logs/celery_worker_<pid>.log`
- **Detailed Logging**: Enhanced logging for debugging and monitoring

### 🔧 Process Management
- **Graceful Shutdown**: Attempts SIGTERM before SIGKILL
- **Process Group Management**: Uses `os.setsid()` for clean process tree management
- **Automatic Cleanup**: Environment guard is cleaned up on exit

## Usage

### Recommended: Use run.py
```bash
# Normal mode (production-like)
python run/run.py

# Debug mode (single-threaded, in-process)
DEBUG_MODE=1 python run/run.py
```

### Standalone Worker
```bash
# Basic usage
python run/celery_worker.py worker --loglevel=info

# With custom options
python run/celery_worker.py worker --loglevel=debug --concurrency=2
```

## What Changed

### From Post-Mortem Analysis
The original issue was caused by:
1. **Stray workers** from PyCharm debug sessions
2. **Default node names** causing silent duplicates
3. **No cleanup** at startup
4. **Hidden processes** not detected by simple `ps` commands

### Implemented Solutions
1. **✅ Environment Guard**: Prevents multiple worker spawns
2. **✅ Unique Node Names**: `api<pid>@<hostname>` format
3. **✅ Process Cleanup**: Kills existing workers at startup
4. **✅ Health Checks**: Verifies single worker after startup
5. **✅ Enhanced Logging**: Per-worker log files and detailed output

## Debugging

### Check for Duplicate Workers
```bash
# Manual check
celery -A TradeLens inspect ping

# Process check
ps aux | grep "celery.*TradeLens"
```

### Log Files
- Main runner logs: Console output
- Worker logs: `logs/celery_worker_<pid>.log`

### Environment Variables
- `TL_CELERY_STARTED=1`: Indicates worker is running
- `DEBUG_MODE=1`: Enables debug mode with in-process worker

## IDE Configuration

### PyCharm
To prevent stray workers in PyCharm:
1. Use the main `run.py` script instead of direct Celery commands
2. Ensure "Kill process tree" is enabled in run configurations
3. Consider running workers in external terminal for debugging

### VS Code
Similar precautions apply - use the launcher scripts rather than direct Celery commands.

## Monitoring in Production

For production deployments, consider:
1. **Systemd/Launchctl**: Use system service managers
2. **Process Monitoring**: Tools like Supervisor or systemd
3. **Health Checks**: Regular `celery inspect ping` checks
4. **Log Aggregation**: Centralized logging for multiple workers

## Troubleshooting

### "TL_CELERY_STARTED=1" Error
If you see this error, it means:
1. Another worker is already running, OR
2. A previous worker didn't clean up properly

**Solution**: 
```bash
# Clear the environment variable
unset TL_CELERY_STARTED

# Or restart your terminal/IDE
```

### Multiple Workers Detected
If health checks fail with multiple workers:
1. Check for stray processes: `ps aux | grep celery`
2. Kill manually: `pkill -f "celery.*TradeLens"`
3. Restart with `run.py` (it will auto-cleanup)

### Worker Not Starting
1. Check logs in `logs/celery_worker_<pid>.log`
2. Verify Django settings are correct
3. Ensure Redis/broker is running
4. Check for port conflicts

## Next Steps

- [ ] Add CI health checks to verify single worker
- [ ] Implement systemd service files for production
- [ ] Add Prometheus metrics for worker monitoring
- [ ] Create IDE run configuration templates 