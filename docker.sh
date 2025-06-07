#!/bin/bash
# Convenience script to run Docker commands from project root

# Change to docker directory
cd docker

# Pass all arguments to docker-compose
docker-compose "$@" 