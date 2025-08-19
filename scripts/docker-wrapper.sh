#!/bin/bash

# Docker wrapper script for macOS with Docker Desktop
# Usage: ./scripts/docker-wrapper.sh [docker-commands]

# Set the correct PATH for Docker Desktop
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running or not accessible."
    echo "Please start Docker Desktop and try again."
    exit 1
fi

# Execute the docker command with all arguments
exec docker "$@"
