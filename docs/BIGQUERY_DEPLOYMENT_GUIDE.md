# BigQuery Authentication for Deployed Environments

## Problem Description
When deploying the Streamlit app to cloud platforms (like Streamlit Community Cloud), BigQuery authentication fails because the app tries to use Google Compute Engine metadata service, which is only available on Google Cloud infrastructure.

## Error Symptoms
```
WARNING:google.auth.compute_engine._metadata:Compute Engine Metadata server unavailable on attempt 1 of 3. Reason: timed out
503 GET /script-health-check (127.0.0.1) 60141.31ms
```

## Solution Implementation Status: ✅ COMPLETED

The authentication timeout issues have been resolved with the following improvements:

### 1. Enhanced Authentication Priority
The BigQueryClient now uses the following priority order:
1. **Streamlit Secrets** (highest priority for Streamlit Cloud)
2. **Environment Variables** (for other cloud platforms)  
3. **Default Authentication** with timeout (local development only)

### 2. Timeout Protection
- Added 5-second timeout for Google Cloud metadata service calls
- Prevents hanging on deployed environments where metadata service isn't available
- Graceful fallback to Alpha Vantage API when BigQuery is unavailable

### 3. Streamlit Secrets Integration
- Automatic detection of `st.secrets.GOOGLE_APPLICATION_CREDENTIALS_JSON`
- Proper JSON parsing and validation
- Fallback chain ensures maximum compatibility

## Solution Implementation

### 1. Environment Detection
The app now automatically detects the deployment environment and adjusts behavior accordingly:
- **Local Development**: Uses default Google Cloud authentication
- **Deployed Environments**: Falls back to Alpha Vantage API only (unless service account is configured)

### 2. Service Account Authentication (Optional)
For deployed environments that need BigQuery access:

#### Step 1: Create Service Account
1. Go to Google Cloud Console → IAM & Admin → Service Accounts
2. Create a new service account with BigQuery permissions
3. Download the JSON key file

#### Step 2: Configure Environment Variable
Add the service account key as an environment variable in your deployment platform:

**Environment Variable Name**: `GOOGLE_APPLICATION_CREDENTIALS_JSON`
**Value**: The entire contents of the service account JSON key file

Example for Streamlit Community Cloud:
```json
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "key-id",
  "private_key": "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n",
  "client_email": "service-account@your-project.iam.gserviceaccount.com",
  "client_id": "client-id",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token"
}
```

### 3. Graceful Fallback
When BigQuery is unavailable:
- The app automatically uses Alpha Vantage API only
- BigQuery option is hidden from the UI
- Users see an informational message about the data source
- No errors or timeouts occur

## Files Modified
- `app_modules/bigquery_client.py`: Enhanced authentication with fallback
- `app_modules/deployment_config.py`: Environment detection and configuration
- `app_modules/data_sources.py`: Graceful handling of BigQuery unavailability
- `app_modules/ui_intro.py`: Conditional UI display based on environment

## Testing
1. **Local Development**: Should work as before with BigQuery if authenticated
2. **Deployed Without Service Account**: Should use Alpha Vantage only, no errors
3. **Deployed With Service Account**: Should use BigQuery if properly configured

## Benefits
- No more timeout errors in deployed environments
- Automatic environment detection
- Graceful degradation to Alpha Vantage API
- Optional BigQuery support for production deployments
- Better user experience with clear status messages
