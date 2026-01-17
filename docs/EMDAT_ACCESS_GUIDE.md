# EM-DAT Access Guide

## Overview

The Phase 1 plan discussed using EM-DAT (Emergency Events Database) as a primary data source. This guide clarifies the actual access methods and how the implementation handles them.

## EM-DAT Access Reality

### What the Plan Assumed
- Simple REST API with API key authentication
- Direct API calls to fetch disaster records

### What EM-DAT Actually Provides

1. **File Downloads** (Primary method):
   - Excel (.xlsx) or JSON files
   - Available after free registration
   - Updated weekly
   - Most reliable access method

2. **API Access** (Limited):
   - GraphQL API (requires specific setup)
   - Python API packages
   - Public REST API may be limited or require special access
   - Not as straightforward as the plan assumed

## Implementation Approach

The codebase implements **both** methods:

### 1. API Fetcher (`services/data_sources/emdat_fetcher.py`)
- Attempts REST API calls
- Requires `EMDAT_API_KEY` in environment
- Handles pagination and rate limiting
- Falls back gracefully if API unavailable

### 2. File Loader (`services/data_sources/emdat_file_loader.py`)
- Loads manually downloaded Excel/JSON files
- Automatically detects files in `data/` directory
- Handles column name variations
- Filters to Southern Africa region
- More reliable and recommended

### 3. Smart Fallback in Scheduler

The data collection scheduler (`scripts/schedule_data_collection.py`) now:

1. **First**: Tries API if `EMDAT_API_KEY` is set
2. **Then**: Falls back to file loader if API fails or no key
3. **Checks**: For EM-DAT files in `data/` directory automatically

## Recommended Workflow

### For Phase 1 Completion:

1. **Register with EM-DAT** (5 minutes):
   - Visit https://www.emdat.be/
   - Create free account
   - Verify email

2. **Download Data**:
   - Go to "Public Data" → "Download"
   - Select date range (2010-2024 recommended)
   - Download Excel file
   - Save to `data/` directory as `emdat_*.xlsx` or `data_disaster.xlsx`

3. **Run Pipeline**:
   ```bash
   python scripts/schedule_data_collection.py --once
   ```

4. **System Will**:
   - Detect the file automatically
   - Load and process it
   - Create master dataset
   - Generate quality reports

## Why This Approach?

- **More Reliable**: File downloads are always available
- **No API Dependencies**: Works without API keys
- **Better Control**: You control which data version to use
- **Offline Capable**: Can work with downloaded files
- **Matches Reality**: Aligns with how EM-DAT actually provides data

## Data Quality Considerations

From EM-DAT documentation:

- **Pre-2000 data**: Marked as "Historic", lower quality
- **2000+ data**: Better completeness and accuracy
- **Recent updates**: Schema changed in 2023, new columns added
- **Missing fields**: Some fields may be incomplete for older events

The data cleaning pipeline handles these issues:
- Flags historic data
- Handles missing values intelligently
- Validates data quality
- Generates quality scores

## Legal/Licensing

- **Non-commercial use**: Free with registration
- **Commercial use**: Requires paid license
- **Attribution**: Must credit EM-DAT/CRED
- **No derivatives**: Archive license restricts some transformations

For Phase 1 (research/non-commercial), free access is sufficient.

## Summary

The Phase 1 plan's discussion of EM-DAT was correct in spirit but assumed a simpler API than what's available. The implementation handles this by:

1. ✅ Supporting both API and file-based access
2. ✅ Prioritizing the more reliable file method
3. ✅ Providing clear instructions for both methods
4. ✅ Automatically detecting and loading files

**Bottom line**: The plan's goal of using EM-DAT is met, with a more practical implementation that uses file downloads as the primary method.
