# Streamlit Cloud BigQuery Deployment Guide

## Overview
This guide provides step-by-step instructions for deploying the Stock Forecasting Tool to Streamlit Cloud with proper BigQuery authentication.

## Problem Solved
- **Timeout Errors**: Fixes "Compute Engine Metadata server unavailable" errors
- **Authentication Failures**: Resolves BigQuery connection issues on Streamlit Cloud
- **Service Account Integration**: Enables proper service account authentication

## Prerequisites
1. Google Cloud Project with BigQuery enabled
2. Service Account with BigQuery permissions
3. Streamlit Cloud account
4. GitHub repository with your code

## Step 1: Create Service Account (if not already done)

### 1.1 Create Service Account
```bash
# Via gcloud CLI
gcloud iam service-accounts create stock-forecasting-sa \
    --description="Service account for Stock Forecasting Tool" \
    --display-name="Stock Forecasting SA"
```

### 1.2 Grant BigQuery Permissions
```bash
# Grant BigQuery permissions
gcloud projects add-iam-policy-binding stock-forecasting-tool-2025 \
    --member="serviceAccount:stock-forecasting-sa@stock-forecasting-tool-2025.iam.gserviceaccount.com" \
    --role="roles/bigquery.dataEditor"

gcloud projects add-iam-policy-binding stock-forecasting-tool-2025 \
    --member="serviceAccount:stock-forecasting-sa@stock-forecasting-tool-2025.iam.gserviceaccount.com" \
    --role="roles/bigquery.jobUser"
```

### 1.3 Generate Service Account Key
```bash
# Download service account key
gcloud iam service-accounts keys create credentials.json \
    --iam-account=stock-forecasting-sa@stock-forecasting-tool-2025.iam.gserviceaccount.com
```

## Step 2: Configure Streamlit Cloud Secrets

### 2.1 Prepare Service Account JSON
1. Open the `credentials.json` file
2. Copy the entire JSON content
3. Format as a single line (remove newlines within the JSON structure)
4. Escape the private key newlines with `\\n`

### 2.2 Add to Streamlit Cloud
1. Go to your Streamlit Cloud app dashboard
2. Click on "Settings" → "Secrets"
3. Add the following secrets:

```toml
# BigQuery Service Account Authentication
GOOGLE_APPLICATION_CREDENTIALS_JSON = """{"type": "service_account", "project_id": "stock-forecasting-tool-2025", "private_key_id": "YOUR_KEY_ID", "private_key": "-----BEGIN PRIVATE KEY-----\\nYOUR_PRIVATE_KEY\\n-----END PRIVATE KEY-----\\n", "client_email": "stock-forecasting-sa@stock-forecasting-tool-2025.iam.gserviceaccount.com", "client_id": "YOUR_CLIENT_ID", "auth_uri": "https://accounts.google.com/o/oauth2/auth", "token_uri": "https://oauth2.googleapis.com/token", "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs", "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/stock-forecasting-sa%40stock-forecasting-tool-2025.iam.gserviceaccount.com"}"""

# Alpha Vantage API Key (if needed)
ALPHA_VANTAGE_API_KEY = "your-alpha-vantage-api-key"
```

## Step 3: Verify Deployment

### 3.1 Check Application Logs
After deployment, check the Streamlit Cloud logs for:
- ✅ "Using service account credentials from Streamlit secrets"
- ✅ "BigQuery client initialized successfully"
- ✅ "BigQuery connection test successful"

### 3.2 Test BigQuery Integration
1. Launch the deployed app
2. Select a stock symbol
3. Choose "BigQuery (Production)" as data source
4. Verify that data loads without timeout errors

## Step 4: Troubleshooting

### Common Issues and Solutions

#### Issue: "Streamlit secrets not accessible"
**Solution**: Ensure secrets are added to Streamlit Cloud, not just local `.streamlit/secrets.toml`

#### Issue: "BigQuery connection test failed"
**Possible Causes**:
- Service account permissions insufficient
- JSON format incorrect (check escaping)
- Project ID mismatch

**Debug Steps**:
1. Verify service account has BigQuery Editor and Job User roles
2. Check JSON format in secrets (no extra quotes or formatting issues)
3. Ensure project ID matches in both service account and application config

#### Issue: "Invalid private key format"
**Solution**: 
- Ensure private key includes `\\n` for line breaks
- Verify no extra spaces or characters in the JSON

### Debug Mode
To enable more detailed logging, add this to your secrets:
```toml
LOG_LEVEL = "DEBUG"
```

## Step 5: Local Development Setup

For local development, create your own `secrets.toml` file:

### 5.1 Copy Template
```bash
# Copy the template file
cp .streamlit/secrets.toml.template .streamlit/secrets.toml
```

### 5.2 Configure Local Secrets
Edit `.streamlit/secrets.toml` with your actual credentials:
```toml
# Your actual service account JSON (single line, properly escaped)
GOOGLE_APPLICATION_CREDENTIALS_JSON = """{"type": "service_account", ...}"""

# Your Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = "your-actual-key"
```

### 5.3 Verify Local Setup
Run the diagnostic to ensure everything works:
```bash
python debug_scripts/streamlit_auth_diagnostic.py
```

**IMPORTANT**: The `secrets.toml` file is gitignored and will never be committed to version control.

## Benefits of This Setup

1. **No Timeout Errors**: Eliminates metadata service timeouts
2. **Proper Authentication**: Uses service account instead of default auth
3. **Environment Isolation**: Separate local and production credentials
4. **Graceful Degradation**: Falls back to Alpha Vantage if BigQuery fails
5. **Security**: Credentials stored securely in Streamlit Cloud secrets

## Monitoring and Maintenance

### Regular Checks
- Monitor service account key expiration (keys expire after 10 years by default)
- Review BigQuery usage and costs
- Check application logs for authentication warnings

### Key Rotation
When rotating service account keys:
1. Generate new key in Google Cloud Console
2. Update Streamlit Cloud secrets
3. Delete old key after verifying new one works

## Security Best Practices

1. **Never Commit Secrets**: The `secrets.toml` file is in `.gitignore` and should never be committed
2. **Use Template File**: Use `.streamlit/secrets.toml.template` as a reference for format
3. **Principle of Least Privilege**: Service account has only necessary BigQuery permissions
4. **Key Management**: Store keys only in secure secret management systems
5. **Access Logging**: Monitor BigQuery access logs for unusual activity
6. **Regular Audits**: Review service account usage quarterly
7. **Environment Separation**: Use different service accounts for development and production

---

## Support and Next Steps

After successful deployment:
1. Consider implementing connection pooling for better performance
2. Set up monitoring dashboards for BigQuery usage
3. Implement automated testing for BigQuery connectivity
4. Review and optimize BigQuery query performance

For issues not covered in this guide, check the application logs and ensure all prerequisites are met.
