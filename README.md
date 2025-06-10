# TradeLens

Definition of application here? 

## 📋 Prerequisites

- Docker Desktop
- PostgreSQL (running on host)
- Redis (running on host)
- npm

## 🚀 Quick Start with Docker

The easiest way to run TradeLens is using Docker:

See [docker/README.md](docker/README.md) for detailed Docker setup instructions.
Specifically you need to configure variables in .env

First time running the project apply backend db migrations: 
```bash
docker-compose exec tradelens-backend python manage.py migrate
```

Ensure that redis and postgres are running on local machine

Then build and start the backend with Django.

```bash
# From project root
./docker.sh up --build

# Or manually
cd docker
docker-compose up --build
```

Finally Launch the frontend: 

```bash
cd react_frontend
npm install 
npm start
```


## 📋 Prerequisites

- Docker Desktop
- PostgreSQL (running on host)
- Redis (running on host)
