# Requirements Management

This directory contains organized Python requirements files for the TradeLens project.

## Structure

- **base.txt**: Core dependencies (Django, Celery, database drivers, etc.)
- **ml.txt**: Machine learning and data science packages
- **langchain.txt**: LangChain and AI provider SDKs
- **docker.txt**: Main file that includes all production dependencies
- **dev.txt**: Development-only dependencies (testing, linting, debugging)
- **constraints.txt**: Version constraints to resolve conflicts

## Usage

### For Docker (Production)
```bash
pip install -c requirements/constraints.txt -r requirements/docker.txt
```

### For Development
```bash
pip install -c requirements/constraints.txt -r requirements/dev.txt
```

### Installing Specific Components
```bash
# Just core dependencies
pip install -r requirements/base.txt

# Add ML capabilities
pip install -r requirements/ml.txt

# Add LangChain
pip install -r requirements/langchain.txt
```

## Managing Dependencies

### Adding a New Dependency

1. Determine which file it belongs in:
   - Web framework, database, or utility? → `base.txt`
   - ML/Data science? → `ml.txt`
   - AI/LLM related? → `langchain.txt`
   - Development only? → `dev.txt`

2. Add with a specific version:
   ```
   package-name==1.2.3
   ```

3. If there are conflicts, add constraints to `constraints.txt`

### Updating Dependencies

1. Update the version in the appropriate file
2. Test in Docker:
   ```bash
   docker-compose build --no-cache
   ```
3. Run tests to ensure compatibility

### Resolving Conflicts

1. Check error messages for conflicting versions
2. Add constraints to `constraints.txt`
3. Consider using compatible version ranges:
   ```
   package>=1.0,<2.0
   ```

## Best Practices

1. **Always pin versions** in production files
2. **Group related packages** with comments
3. **Document special cases** (like PyTorch CPU builds)
4. **Test changes** in Docker before committing
5. **Keep constraints minimal** - only add when needed

## Common Issues

### TensorFlow/Keras Version Mismatch
- TensorFlow 2.18+ requires Keras 3.5+
- Solution: Use compatible versions or downgrade TensorFlow

### PyTorch Installation
- CPU builds need special index URL
- Solution: Install PyTorch separately in Dockerfile

### Platform-Specific Packages
- Some packages have different wheels for ARM64 vs x86_64
- Solution: Let pip resolve automatically, avoid platform-specific versions 