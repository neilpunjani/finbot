# Deployment Guide

## Overview
This guide covers deploying the Finaptive AI Chatbot with a React frontend and FastAPI backend to Azure.

## Architecture
- **Frontend**: React app deployed to Azure Static Web Apps
- **Backend**: FastAPI app deployed to Azure Container Apps
- **Database**: Your existing data sources (SQL, Excel, CSV, Email)

## Prerequisites
1. Azure account with active subscription
2. Azure CLI installed
3. Node.js 18+ installed
4. Python 3.11+ installed
5. Docker installed (for backend deployment)

## Local Development

### 1. Start the Backend API
```bash
# Install Python dependencies
pip install -r requirements.txt

# Start the FastAPI server
python api.py
```
The API will be available at `http://localhost:8000`

### 2. Start the Frontend
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the React app
npm start
```
The frontend will be available at `http://localhost:3000`

## Azure Deployment

### 1. Deploy Backend to Azure Container Apps

```bash
# Login to Azure
az login

# Create resource group
az group create --name test-bot-rg --location eastus

# Create container registry
az acr create --resource-group test-bot-rg --name testbot --sku Basic

# Build and push Docker image
az acr build --registry testbot --image chatbot-api:latest .

# Create Container Apps environment
az containerapp env create \
  --name test-bot-env \
  --resource-group test-bot-rg \
  --location eastus

# Deploy the container app
az containerapp create \
  --name test-bot-api \
  --resource-group test-bot-rg \
  --environment test-bot-env \
  --image testbot.azurecr.io/chatbot-api:latest \
  --target-port 8000 \
  --ingress 'external' \
  --registry-server testbot.azurecr.io
```

### 2. Deploy Frontend to Azure Static Web Apps

#### Option A: Using Azure Portal
1. Go to Azure Portal > Create Resource > Static Web App
2. Connect to your GitHub repository
3. Set build details:
   - App location: `/frontend`
   - Output location: `build`
4. Deploy

#### Option B: Using Azure CLI
```bash
# Create static web app
az staticwebapp create \
  --name test-bot-frontend \
  --resource-group test-bot-rg \
  --source https://github.com/yourusername/test-bot \
  --location eastus \
  --branch main \
  --app-location "/frontend" \
  --output-location "build"
```

### 3. Configure Environment Variables

#### Backend (Container Apps)
```bash
# Add environment variables to container app
az containerapp update \
  --name test-bot-api \
  --resource-group test-bot-rg \
  --set-env-vars \
    OPENAI_API_KEY=your_openai_key \
    DATABASE_URL=your_database_url \
    EMAIL_ADDRESS=your_email \
    EMAIL_PASSWORD=your_email_password
```

#### Frontend (Static Web Apps)
1. Go to Azure Portal > Your Static Web App > Configuration
2. Add application setting:
   - Name: `REACT_APP_API_URL`
   - Value: `https://your-container-app-url.azurecontainerapps.io`

## Production Considerations

### Security
- Use Azure Key Vault for sensitive environment variables
- Enable HTTPS only
- Configure CORS properly in production
- Use managed identity for Azure resources

### Monitoring
- Enable Application Insights for both frontend and backend
- Set up alerts for errors and performance issues
- Configure log analytics

### Scaling
- Container Apps auto-scale based on HTTP requests
- Static Web Apps scale automatically
- Monitor costs and set budget alerts

## Troubleshooting

### Common Issues
1. **CORS errors**: Update CORS settings in `api.py`
2. **Environment variables**: Check Azure configuration
3. **Build failures**: Verify Node.js version and dependencies
4. **API connection**: Ensure backend URL is correct in frontend

### Logs
```bash
# View container app logs
az containerapp logs show \
  --name test-bot-api \
  --resource-group test-bot-rg
```

## Cost Optimization
- Use Azure Free Tier where possible
- Monitor usage with Azure Cost Management
- Set up budget alerts
- Consider using consumption-based pricing

## Next Steps
1. Set up CI/CD pipeline with GitHub Actions
2. Configure custom domains
3. Add authentication (Azure AD B2C)
4. Implement caching strategies
5. Add monitoring and alerting