# Full Universe Bulk Loading Instructions

## Overview
Run this on your other machine to complete the BigQuery data warehouse with all ~9,000 remaining stock symbols. Estimated time: 2.5 hours with automatic progress tracking.

## Prerequisites Setup

### 1. Environment Setup
```bash
# Clone/pull latest repository
git clone https://github.com/shanejp76/Stock-Forecasting-Tool.git
cd Stock-Forecasting-Tool
git pull origin main

# Create conda environment
conda create -n stock-forecasting python=3.11 -y
conda activate stock-forecasting

# Install dependencies
pip install -r requirements.txt
```

### 2. Authentication Setup
```bash
# Copy your .env file to this machine with:
# ALPHA_VANTAGE_API_KEY=your_key_here
# GOOGLE_CLOUD_PROJECT=stock-forecasting-tool-prod

# Copy your Google Cloud service account key file
# Place it as: ~/.config/gcloud/stock-forecasting-service-account.json

# Set authentication
export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/stock-forecasting-service-account.json
```

### 3. Verify Setup
```bash
# Test BigQuery connection
conda activate stock-forecasting
python -c "
from app_modules.bigquery_client import BigQueryClient
client = BigQueryClient()
symbols = client.get_available_symbols()
print(f'Current symbols: {len(symbols)}')
print('✅ BigQuery connection verified')
"
```

## Run Full Universe Loading

### Start the Bulk Load
```bash
conda activate stock-forecasting
python scripts/initial_bulk_load.py --full-universe --yes
```

### What to Expect
- **Total symbols**: ~9,000 remaining (329 already loaded)
- **Estimated time**: 2.5 hours (free API: 5 calls/minute)
- **Progress tracking**: Live progress bar with ETA
- **Checkpointing**: Automatic resume if interrupted
- **Rate limiting**: Built-in Alpha Vantage compliance

### Monitor Progress
The script provides:
- Real-time progress bar
- Success/failure counters
- Throughput metrics
- Checkpoint saves every 10 symbols

### If Interrupted
Simply re-run the same command - it will resume from the last checkpoint:
```bash
python scripts/initial_bulk_load.py --full-universe --yes
```

## Expected Final Results
- **Total symbols**: ~9,329 in BigQuery
- **Data range**: 2+ years (August 2023 - August 2025)
- **Success rate**: 95%+ (some symbols may fail due to API issues)
- **Storage**: ~500MB in BigQuery

## Verification Commands

### Check Final Symbol Count
```bash
python -c "
from app_modules.bigquery_client import BigQueryClient
client = BigQueryClient()
symbols = client.get_available_symbols()
print(f'Final symbol count: {len(symbols)}')
"
```

### Test Random Symbols
```bash
python -c "
from app_modules.data_handler import load_bigquery_data
from datetime import date, timedelta
import random

# Test a few random symbols
test_symbols = ['AAPL', 'GOOG', 'MSFT', 'TSLA', 'META']
start_date = date.today() - timedelta(days=30)

for symbol in test_symbols:
    data, source = load_bigquery_data(symbol, start_date)
    print(f'{symbol}: {len(data)} rows from {source}')
"
```

## Troubleshooting

### API Rate Limit Errors
- Script automatically handles rate limiting
- If you see persistent errors, verify your Alpha Vantage API key

### BigQuery Connection Issues
- Verify GOOGLE_APPLICATION_CREDENTIALS path
- Check service account permissions
- Test with: `gcloud auth application-default login`

### Memory Issues
- Script is optimized for low memory usage
- Each symbol processed individually
- Checkpoints prevent data loss

## Next Steps After Completion

1. **Commit the progress**:
   ```bash
   git add data/bulk_load_progress.pkl
   git commit -m "Complete full universe bulk loading - ~9,329 symbols in BigQuery"
   git push origin main
   ```

2. **Test the application**:
   ```bash
   streamlit run main.py
   ```

3. **Ready for Phase 2**: Research integration and advanced features

## Contact
If you encounter issues, check the bulk_load_progress.pkl file for current status and refer to the MODERNIZATION_ROADMAP.md for context.

---
**Estimated completion**: 2.5 hours | **Auto-resume**: Yes | **Progress tracking**: Real-time
