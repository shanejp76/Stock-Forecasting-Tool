# Docker Deployment Guide

This guide explains how to containerize and deploy the Stock Forecasting Tool using Docker.

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- API keys for Alpha Vantage and Finnhub
- (Optional) Google Cloud SDK for cloud deployment

### Local Development

1. **Copy environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Add your API keys to `.env`:**
   ```bash
   ALPHA_VANTAGE_API_KEY=your_actual_api_key
   FINNHUB_API_KEY=your_actual_api_key
   ```

3. **Deploy locally:**
   ```bash
   ./scripts/deploy-local.sh
   ```

4. **Access the application:**
   Open http://localhost:8501 in your browser

## Docker Configuration

### Dockerfile Structure

The project includes two Dockerfiles:

- **`Dockerfile`**: Development version with debugging capabilities
- **`Dockerfile.prod`**: Production-optimized multi-stage build

### Key Features

- **Multi-stage builds** for smaller production images
- **Non-root user** for security
- **Health checks** for monitoring
- **Environment variable** configuration
- **Optimized caching** for faster builds

### Docker Compose

The `docker-compose.yml` file provides:

- **Streamlit app** service with auto-restart
- **Environment variable** injection
- **Volume mounting** for development
- **Health monitoring**
- **Network isolation**

Optional services (commented out):
- PostgreSQL database
- Redis caching
- Future microservices

## Deployment Options

### 1. Local Development

```bash
# Start in development mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 2. Production Build

```bash
# Build production image
docker build -f Dockerfile.prod -t stock-forecasting-app:prod .

# Run production container
docker run -d \
  --name stock-app \
  -p 8080:8080 \
  -e ALPHA_VANTAGE_API_KEY=your_key \
  -e FINNHUB_API_KEY=your_key \
  stock-forecasting-app:prod
```

### 3. Google Cloud Run

```bash
# Deploy to Cloud Run
./scripts/deploy-cloud.sh
```

## Environment Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage API key | `ABC123...` |
| `FINNHUB_API_KEY` | Finnhub API key | `XYZ789...` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `STREAMLIT_SERVER_PORT` | Server port | `8501` |
| `STREAMLIT_SERVER_ADDRESS` | Server address | `0.0.0.0` |
| `ENVIRONMENT` | Environment mode | `development` |

## Cloud Deployment

### Google Cloud Run

1. **Setup Google Cloud:**
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **Create secrets:**
   ```bash
   echo "your_alpha_vantage_key" | gcloud secrets create alpha-vantage-key --data-file=-
   echo "your_finnhub_key" | gcloud secrets create finnhub-key --data-file=-
   ```

3. **Deploy:**
   ```bash
   ./scripts/deploy-cloud.sh
   ```

### AWS ECS/Fargate

```bash
# Build and tag image
docker build -f Dockerfile.prod -t stock-forecasting-app .
docker tag stock-forecasting-app:latest YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/stock-forecasting-app:latest

# Push to ECR
aws ecr get-login-password --region REGION | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com
docker push YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/stock-forecasting-app:latest
```

### Azure Container Instances

```bash
# Build and push to Azure Container Registry
docker build -f Dockerfile.prod -t stock-forecasting-app .
docker tag stock-forecasting-app YOUR_REGISTRY.azurecr.io/stock-forecasting-app:latest
docker push YOUR_REGISTRY.azurecr.io/stock-forecasting-app:latest

# Deploy to ACI
az container create \
  --resource-group YOUR_RG \
  --name stock-forecasting-app \
  --image YOUR_REGISTRY.azurecr.io/stock-forecasting-app:latest \
  --ports 8080 \
  --environment-variables ALPHA_VANTAGE_API_KEY=your_key FINNHUB_API_KEY=your_key
```

## Monitoring and Maintenance

### Health Checks

The containers include built-in health checks:

```bash
# Check container health
docker ps

# View health check logs
docker inspect CONTAINER_ID | grep Health -A 20
```

### Logs

```bash
# View application logs
docker-compose logs -f streamlit-app

# Follow logs with timestamps
docker-compose logs -f -t streamlit-app
```

### Updates

```bash
# Rebuild and restart
docker-compose build --no-cache
docker-compose up -d
```

## Troubleshooting

### Common Issues

1. **Port already in use:**
   ```bash
   # Change port in docker-compose.yml or stop conflicting service
   docker-compose down
   ```

2. **API key errors:**
   ```bash
   # Verify environment variables
   docker-compose exec streamlit-app env | grep API
   ```

3. **Build failures:**
   ```bash
   # Clean build cache
   docker system prune -a
   ```

### Debug Mode

```bash
# Run with debug shell
docker run -it --entrypoint /bin/bash stock-forecasting-app:latest

# Check environment
docker-compose exec streamlit-app env
```

## Security Considerations

- **API keys** stored as environment variables
- **Non-root user** in containers
- **Minimal base images** reduce attack surface
- **No secrets** in image layers
- **Network isolation** via Docker networks

## Performance Optimization

- **Multi-stage builds** for smaller images
- **Python package caching** for faster builds
- **Resource limits** prevent resource exhaustion
- **Health checks** enable auto-recovery

## Next Steps

After successful containerization:

1. **Implement CI/CD** with GitHub Actions
2. **Add database** services (PostgreSQL/Redis)
3. **Configure monitoring** (Prometheus/Grafana)
4. **Setup auto-scaling** for cloud deployments
5. **Add backup strategies** for data persistence

## Support

For issues with containerization:
1. Check the logs: `docker-compose logs`
2. Verify environment variables
3. Ensure Docker daemon is running
4. Check port availability
