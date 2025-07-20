# GEMINI.md

This file provides guidance to GEMINI Code when working with code in this repository.

## Development Commands

### Frontend (React)
```bash
cd react_frontend
npm install              # Install dependencies
npm start               # Start development server (port 3000)
npm run build           # Build for production
npm test                # Run tests
```

### Backend (Django)
```bash
# Standard Django development
python manage.py runserver                    # Start development server (port 8000)
python manage.py migrate                      # Run database migrations  
python manage.py makemigrations               # Create new migrations
python manage.py shell                        # Django shell
python manage.py collectstatic                # Collect static files
python manage.py createsuperuser             # Create admin user

# Testing
python manage.py test                         # Run all tests
python manage.py test app_name               # Run specific app tests

# Background tasks
celery -A TradeLens worker --loglevel=info   # Start Celery worker
```

### Docker Development
```bash
# Quick start
./docker.sh up --build                       # Build and start all services
./docker.sh up -d --build                   # Run in background

# From docker directory
cd docker
docker-compose up --build                    # Build and start
docker-compose down                          # Stop services
docker-compose logs -f                       # Follow logs

# Django commands in Docker
docker-compose exec tradelens-backend python manage.py migrate
docker-compose exec tradelens-backend python manage.py createsuperuser
docker-compose exec tradelens-backend python manage.py shell
```

## Architecture Overview

TradeLens is a full-stack application with a Django backend and React frontend, built around an entity-strategy pattern for extensible functionality.

### Core Architecture Patterns

**Entity-Strategy Pattern**: The system uses a sophisticated entity-strategy architecture where:
- **Entities** are data models (documents, recipes, meal plans, visualizations, API models, training sessions)
- **Strategies** are stateless classes that operate on entities (CreateEntityStrategy, SetAttributesStrategy, AddChildStrategy)
- **StrategyRequests** are the communication envelope containing strategy name, target entity, and parameters

**Entity Types**: Main entity types include:
- `DocumentEntity`: File/document management with IDE-like interface
- `ViewEntity`: UI components that render entity data
- `RecipeEntity`/`MealPlanEntity`: Meal planning functionality
- `CalendarEntity`/`CalendarEventEntity`: Calendar system
- `TrainingSessionEntity`: ML model training
- `ApiModelEntity`: LLM integration

### Backend Structure

**Django Apps**:
- `shared_utils/`: Core entity system, strategy executor, base classes
- `dataset_manager/`: Data handling and stock configuration
- `sequenceset_manager/`: Sequence data management
- `training_session/`: ML training workflows with WebSocket support
- `model_stage/`: Model staging and reinforcement learning utilities
- `data_bundle_manager/`: Feature set and data bundle management

**Key Backend Patterns**:
- WebSocket support via Django Channels for real-time updates
- Celery for background task processing
- Redis for caching and task broker
- PostgreSQL with pgvector for vector storage
- Entity relationships managed through parent-child links

### Frontend Structure

**React Architecture**:
- Component-based with functional components and hooks
- State management using both Recoil and Redux Toolkit
- Real-time updates via WebSocket connections
- Drag-and-drop interfaces using @dnd-kit and @xyflow/react

**Key Frontend Patterns**:
- Canvas-based entity rendering system in `components/Canvas/`
- View entity system that creates specialized UI components
- IDE-like document editing interface with file trees and search
- Real-time strategy execution and result handling

### Entity-Strategy Workflow

1. **Strategy Execution**: Form strategy requests via `create_strategy_request` tool
2. **Entity Operations**: Use strategies like CreateEntityStrategy, SetAttributesStrategy, AddChildStrategy
3. **Data Flow**: Strategies return results that can chain into additional operations
4. **View Rendering**: ViewEntity creates specialized UI components for different entity types

### LLM Integration

The system includes sophisticated LLM integration through `ApiModelEntity`:
- Supports multiple LLM providers (OpenAI, Anthropic, Google)
- Context management for documents and entities
- Agent-like functionality with tool calling
- Instructions and context stored in `shared_utils/entities/api_model/strategy/instructions.txt`

## Environment Setup

### Prerequisites
- Docker Desktop
- PostgreSQL (running on host, port 5432)
- Redis (running on host, port 6379)
- Node.js and npm

### Configuration
- Environment variables in `docker/.env` (copy from `docker.env.example`)
- Database: PostgreSQL database named `tradelens`
- Cache/Broker: Redis on default ports
- Frontend dev server: http://localhost:3000
- Backend API: http://localhost:8000

### First-time Setup
```bash
# 1. Set up environment
cp docker/docker.env.example docker/.env

# 2. Start external services  
brew services start postgresql redis  # macOS
sudo systemctl start postgresql redis # Linux

# 3. Create database
psql -U postgres -c "CREATE DATABASE tradelens;"

# 4. Build and start backend
./docker.sh up --build

# 5. Run migrations (first time only)
docker-compose exec tradelens-backend python manage.py migrate

# 6. Start frontend
cd react_frontend && npm install && npm start
```

## Testing Strategy

- **Frontend**: React Testing Library with Jest (`npm test`)
- **Backend**: Django test framework (`python manage.py test`)
- **Integration**: Docker-based testing with separate test database
- **Entity Tests**: Located in respective `tests/` directories within each app

## Key Development Notes

- The system is designed around creating, linking, and manipulating entities through strategies
- Views are themselves entities that render other entity data
- Real-time updates flow through WebSocket connections
- The IDE views (`ide_app_dashboard`) provide document management for any entity
- Calendar and meal planning demonstrate the entity-view pattern
- LLM agents can execute strategies to manipulate the system programmatically
