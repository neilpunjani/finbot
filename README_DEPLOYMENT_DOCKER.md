# Docker Layer Optimization for Faster Deployments

## Problem
Using `az acr build` rebuilds the entire Docker image each time, which is slow and inefficient. This guide shows how to use Docker layer caching to only update changed files.

## Solution Overview
- Build Docker image locally with optimized layers
- Push only changed layers to Azure Container Registry
- Deploy updated image to Azure Container Apps

## Optimized Dockerfile Structure

Create or update your `Dockerfile` with proper layer caching:

```dockerfile
# Use Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (rarely change)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (changes less frequently than code)
COPY requirements.txt .

# Install Python dependencies (cached unless requirements.txt changes)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (changes most frequently - put last)
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "api.py"]
```

## Local Docker Build & Push Workflow

### 1. Initial Setup (One-time)
```powershell
# Login to Azure and Docker
az login
az acr login --name testbot

# Set environment variables
$REGISTRY_NAME = "testbot"
$IMAGE_NAME = "chatbot-api"
$TAG = "latest"
```

### 2. Build and Push with Layer Caching
```powershell
# Build image locally (uses layer caching)
docker build -t $REGISTRY_NAME.azurecr.io/$IMAGE_NAME:$TAG .

# Push to Azure Container Registry (only pushes changed layers)
docker push $REGISTRY_NAME.azurecr.io/$IMAGE_NAME:$TAG
```

### 3. Deploy to Container Apps
```powershell
# Update the container app with new image
az containerapp update `
  --name test-bot-api `
  --resource-group test-bot-rg `
  --image $REGISTRY_NAME.azurecr.io/$IMAGE_NAME:$TAG
```

## Advanced Layer Optimization

### Multi-stage Build (Even Faster)
```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Add local packages to PATH
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

EXPOSE 8000
CMD ["python", "api.py"]
```

### Ignore Unnecessary Files (.dockerignore)
Create/update `.dockerignore`:
```
__pycache__
*.pyc
*.pyo
*.pyd
.git
.gitignore
README*.md
.env
.vscode
tests/
docs/
frontend/
node_modules
.DS_Store
Thumbs.db
```

## Automated Deployment Script

Create `scripts/deploy.ps1`:
```powershell
param(
    [string]$Tag = "latest",
    [string]$RegistryName = "testbot",
    [string]$ImageName = "chatbot-api",
    [string]$ResourceGroup = "test-bot-rg",
    [string]$ContainerApp = "test-bot-api"
)

# Login to Azure Container Registry
Write-Host "Logging into Azure Container Registry..." -ForegroundColor Green
az acr login --name $RegistryName

# Build Docker image with caching
Write-Host "Building Docker image..." -ForegroundColor Green
docker build -t "$RegistryName.azurecr.io/$ImageName:$Tag" .

if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker build failed"
    exit 1
}

# Push image (only changed layers)
Write-Host "Pushing image to registry..." -ForegroundColor Green
docker push "$RegistryName.azurecr.io/$ImageName:$Tag"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker push failed"
    exit 1
}

# Update Container App
Write-Host "Updating Container App..." -ForegroundColor Green
az containerapp update `
  --name $ContainerApp `
  --resource-group $ResourceGroup `
  --image "$RegistryName.azurecr.io/$ImageName:$Tag"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Container App update failed"
    exit 1
}

Write-Host "Deployment completed successfully!" -ForegroundColor Green
```

### Usage:
```powershell
# Deploy with default settings
./scripts/deploy.ps1

# Deploy with custom tag
./scripts/deploy.ps1 -Tag "v1.2.3"
```

## Layer Caching Benefits

### Before (az acr build):
```
Building entire image: 5-10 minutes
- Downloads base image
- Installs system packages
- Installs Python packages
- Copies application code
```

### After (local build with layers):
```
First build: 5-10 minutes
Subsequent builds: 30 seconds - 2 minutes
- Reuses base image layer (cached)
- Reuses system packages layer (cached)
- Reuses Python packages layer (cached if requirements.txt unchanged)
- Only rebuilds application code layer
```

## Development Workflow

### 1. Make Code Changes
Edit your Python files in `src/`, update `api.py`, etc.

### 2. Quick Deploy
```powershell
./scripts/deploy.ps1
```

### 3. When Dependencies Change
Only when you modify `requirements.txt`:
```powershell
# Force rebuild of dependency layers
docker build --no-cache -t testbot.azurecr.io/chatbot-api:latest .
docker push testbot.azurecr.io/chatbot-api:latest
az containerapp update --name test-bot-api --resource-group test-bot-rg --image testbot.azurecr.io/chatbot-api:latest
```

## Monitoring Build Performance

### Check Layer Sizes
```powershell
# View image layers and sizes
docker history testbot.azurecr.io/chatbot-api:latest
```

### Monitor Push Progress
```powershell
# Push with verbose output
docker push testbot.azurecr.io/chatbot-api:latest --verbose
```

## Troubleshooting

### Cache Not Working
```powershell
# Clear Docker cache if needed
docker system prune -a

# Rebuild with no cache
docker build --no-cache -t testbot.azurecr.io/chatbot-api:latest .
```

### Registry Authentication Issues
```powershell
# Re-authenticate
az acr login --name testbot

# Check credentials
az acr credential show --name testbot
```

### Container App Not Updating
```powershell
# Force restart
az containerapp revision restart --name test-bot-api --resource-group test-bot-rg

# Check deployment status
az containerapp revision list --name test-bot-api --resource-group test-bot-rg --query "[].{Name:name,Active:properties.active,CreatedTime:properties.createdTime}"
```

## Best Practices

1. **Order Dockerfile layers** from least to most frequently changed
2. **Use .dockerignore** to exclude unnecessary files
3. **Keep images small** by using slim base images
4. **Version your images** with meaningful tags
5. **Monitor layer sizes** to optimize build times
6. **Use multi-stage builds** for production images

## Time Savings

| Deployment Method | First Build | Subsequent Builds | Time Saved |
|-------------------|-------------|-------------------|------------|
| `az acr build` | 8 minutes | 8 minutes | 0% |
| Local Docker | 8 minutes | 1 minute | 87% |

This optimization reduces deployment time from 8 minutes to 1 minute for code changes!