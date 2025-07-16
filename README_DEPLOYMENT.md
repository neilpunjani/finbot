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
```powershell
# Install Python dependencies
pip install -r requirements.txt

# Start the FastAPI server
python api.py
```
The API will be available at `http://localhost:8000`

### 2. Start the Frontend
```powershell
# Navigate to frontend directory
Set-Location frontend

# Install dependencies
npm install

# Start the React app
npm start
```
The frontend will be available at `http://localhost:3000`

## Azure Deployment

### 1. Deploy Backend to Azure Container Apps

```powershell
# Login to Azure
az login

# Create resource group
az group create --name finaptive-llmbot --location eastus

# Create container registry
az acr create --resource-group finaptive-llmbot --name finbotcr --sku Basic

# Enable admin access for the registry
az acr update --name finbotcr --admin-enabled true

# Build and push Docker image
az acr build --registry finbotcr --image chatbot-api:latest .

# Get registry credentials (copy the password from output)
az acr credential show --name finbotcr

# Create Container Apps environment
az containerapp env create `
  --name finaptive-llmbot-env `
  --resource-group finaptive-llmbot `
  --location eastus

# Deploy the container app (replace YOUR_PASSWORD with password from credential show command)
az containerapp create `
  --name finaptive-llmbot-api `
  --resource-group finaptive-llmbot `
  --environment finaptive-llmbot-env `
  --image finbotcr.azurecr.io/chatbot-api:latest `
  --target-port 8000 `
  --ingress 'external' `
  --registry-server finbotcr.azurecr.io `
  --registry-username finbotcr `
  --registry-password YOUR_PASSWORD
```

**Note**: Replace `YOUR_PASSWORD` with the actual password from the `az acr credential show` command output.

### 2. Deploy Frontend to Azure Static Web Apps

#### Option A: Using Azure Portal
1. Go to Azure Portal > Create Resource > Static Web App
2. Connect to your GitHub repository
3. Set build details:
   - App location: `/frontend`
   - Output location: `build`
4. Deploy

#### Option B: Using Azure CLI
```powershell
# Create static web app
az staticwebapp create `
  --name finaptive-llmbot-frontend `
  --resource-group finaptive-llmbot `
  --source https://github.com/yourusername/finaptive_chatbot `
  --location eastus `
  --branch main `
  --app-location "/frontend" `
  --output-location "build"
```

### 3. Configure Environment Variables

#### Backend (Container Apps)

##### Option 1: Direct Environment Variables (Simple)
```powershell
# Add environment variables to container app
az containerapp update `
  --name finaptive-llmbot-api `
  --resource-group finaptive-llmbot `
  --set-env-vars `
    OPENAI_API_KEY=your_openai_key `
    DATABASE_URL=your_database_url `
    EMAIL_ADDRESS=your_email `
    EMAIL_PASSWORD=your_email_password
```

##### Option 2: Azure Key Vault (Recommended for Production)

**Step 1: Create Key Vault and Add Secrets**
```powershell
# Create Key Vault
az keyvault create `
  --name finaptive-llmbot-kv `
  --resource-group finaptive-llmbot `
  --location eastus

# Grant yourself access to Key Vault first
$userObjectId = (az ad signed-in-user show --query id -o tsv)
$subscriptionId = (az account show --query id -o tsv)

az role assignment create `
  --role "Key Vault Secrets Officer" `
  --assignee $userObjectId `
  --scope "/subscriptions/$subscriptionId/resourcegroups/finaptive-llmbot/providers/microsoft.keyvault/vaults/finaptive-llmbot-kv"

# Wait 2-3 minutes for role propagation, then add secrets to Key Vault
az keyvault secret set --vault-name finaptive-llmbot-kv --name "openai-api-key" --value "your_openai_key"
az keyvault secret set --vault-name finaptive-llmbot-kv --name "database-url" --value "your_database_url"
az keyvault secret set --vault-name finaptive-llmbot-kv --name "email-address" --value "your_email"
az keyvault secret set --vault-name finaptive-llmbot-kv --name "email-password" --value "your_email_password"
```

**Step 2: Enable Managed Identity and Grant Permissions**
```powershell
# Enable managed identity for Container App
az containerapp identity assign `
  --name finaptive-llmbot-api `
  --resource-group finaptive-llmbot `
  --system-assigned

# Get the managed identity principal ID
$principalId = (az containerapp identity show --name finaptive-llmbot-api --resource-group finaptive-llmbot --query principalId -o tsv)

# Grant Key Vault access to managed identity using RBAC
az role assignment create `
  --role "Key Vault Secrets User" `
  --assignee $principalId `
  --scope "/subscriptions/$subscriptionId/resourcegroups/finaptive-llmbot/providers/microsoft.keyvault/vaults/finaptive-llmbot-kv"
```

**Step 3: Connect Key Vault to Container App via Azure Portal**

Since CLI has syntax issues, use the Azure Portal:

1. **Go to Azure Portal** → Navigate to your Container App: `finaptive-llmbot-api`

2. **Add Key Vault Secrets:**
   - Go to **Settings** → **Secrets**
   - Click **Add** for each secret:
     
     **Secret 1:**
     - Key: `openai-api-key`
     - Type: `Key Vault reference`
     - Key Vault Secret URL: `https://finaptive-llmbot-kv.vault.azure.net/secrets/openai-api-key`
     - Managed Identity: `System assigned`
     
     **Secret 2:**
     - Key: `database-url`
     - Type: `Key Vault reference`
     - Key Vault Secret URL: `https://finaptive-llmbot-kv.vault.azure.net/secrets/database-url`
     - Managed Identity: `System assigned`
     
     **Secret 3:**
     - Key: `email-address`
     - Type: `Key Vault reference`
     - Key Vault Secret URL: `https://finaptive-llmbot-kv.vault.azure.net/secrets/email-address`
     - Managed Identity: `System assigned`
     
     **Secret 4:**
     - Key: `email-password`
     - Type: `Key Vault reference`
     - Key Vault Secret URL: `https://finaptive-llmbot-kv.vault.azure.net/secrets/email-password`
     - Managed Identity: `System assigned`

3. **Add Environment Variables:**
   - Go to **Settings** → **Environment variables**
   - Add these variables:
     - `OPENAI_API_KEY` = `secretref:openai-api-key`
     - `DATABASE_URL` = `secretref:database-url`
     - `EMAIL_ADDRESS` = `secretref:email-address`
     - `EMAIL_PASSWORD` = `secretref:email-password`

4. **Save and Deploy** - The Container App will restart with Key Vault integration

#### Frontend (Static Web Apps)
1. Go to Azure Portal > Your Static Web App > Configuration
2. Add application setting:
   - Name: `REACT_APP_API_URL`
   - Value: `https://your-container-app-url.azurecontainerapps.io`

## Production Considerations

### Security
- **Use Azure Key Vault** for sensitive environment variables (see Option 2 above)
- **Enable HTTPS only**
- **Configure CORS** properly in production
- **Use managed identity** for Azure resources (eliminates need for connection strings)
- **Rotate secrets regularly** in Key Vault

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
```powershell
# View container app logs
az containerapp logs show `
  --name finaptive-llmbot-api `
  --resource-group finaptive-llmbot
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