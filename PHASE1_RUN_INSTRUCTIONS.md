# Phase 1: Data Foundation - Run Instructions

## Quick Start

### 1. Environment Setup

```bash
# Ensure you're in the project directory
cd c:\Users\uthac\OneDrive\Documents\programing\Python\Hackathon

# Activate virtual environment
.venv\Scripts\activate

# Install dependencies (if needed)
pip install -r requirements.txt
```

### 2. Configure EM-DAT Data Access

**Option A: File Download (Recommended)**

1. Register at https://www.emdat.be/ (free, takes 5 minutes)
2. Go to "Public Data" → "Download"
3. Download the Excel or JSON file
4. Place the file in the `data/` directory
5. The system will automatically detect and load it

**Option B: API Access (If available)**

Copy the dev environment file and add your EM-DAT API key:

```bash
copy config\dev.env .env
```

Edit `.env` and set:

```env
EMDAT_API_KEY=your_emdat_api_key_here
```

**Note**: EM-DAT primarily provides file downloads. API access may be limited. The file download method is recommended and more reliable.

---

## Running the Data Pipeline

### Option A: Run Once (Testing/Manual)

```bash
python scripts/schedule_data_collection.py --once
```

This will:

1. Fetch EM-DAT disaster records
2. Fetch weather data for key locations
3. Clean and validate data
4. Engineer features
5. Save master dataset to `data/processed/disasters_master.csv`

### Option B: Run Scheduled (Production)

```bash
python scripts/schedule_data_collection.py
```

This starts a daily scheduler that runs at 2 AM. Customize the time:

```bash
python scripts/schedule_data_collection.py --hour 3 --minute 30
```

---

## Running Tests

```bash
# Run Phase 1 tests
python -m pytest tests/test_phase1_data_pipeline.py -v

# Run specific test class
python -m pytest tests/test_phase1_data_pipeline.py::TestDataCleaner -v
```

---

## Output Files

After a successful run:

| File                                  | Description                |
| ------------------------------------- | -------------------------- |
| `data/raw/emdat_*.csv`                | Raw EM-DAT disaster data   |
| `data/raw/weather_*.csv`              | Raw weather data           |
| `data/processed/disasters_master.csv` | Cleaned & featured dataset |
| `data/quality_reports.jsonl`          | Quality score history      |
| `data/collection.log`                 | Pipeline execution logs    |
| `data/manifest.json`                  | Data versioning manifest   |

---

## Checking Data Quality

```bash
# View latest quality report (PowerShell)
Get-Content data\quality_reports.jsonl -Tail 1 | ConvertFrom-Json

# View quality trend
python -c "from services.data_quality_monitor import DataQualityMonitor; print(DataQualityMonitor().get_trend())"
```

---

## Troubleshooting

| Issue             | Solution                                                      |
| ----------------- | ------------------------------------------------------------- |
| EM-DAT API fails  | Use file download method instead (see Option A in setup)      |
| No EM-DAT data    | Download file from https://www.emdat.be/ and place in `data/` |
| No weather data   | Verify internet connection; Open-Meteo is free, no key needed |
| Low quality score | Review `data/quality_reports.jsonl` for recommendations       |
| Import errors     | Ensure you're in project root and venv is activated           |

---

## Validate Phase 1 Completion

```bash
python scripts/validate_phase1.py
```

Expected output: All checks pass ✅
