# Phase 1: Data Foundation Documentation

## Overview

Phase 1 establishes a robust, multi-source historical disaster dataset with clean, validated, and engineered features for Crisis Connect's flood prediction system.

**Success Metric**: 1,000+ validated disaster records with quality score >75%

---

## Data Sources

### 1. EM-DAT (Emergency Events Database)

- **Provider**: Centre for Research on the Epidemiology of Disasters (CRED)
- **URL**: https://www.emdat.be/
- **Coverage**: Global disasters since 1900 (~27,000 events)
- **Update Frequency**: Weekly (public data)
- **Access**: Free with registration (non-commercial use)

**Access Methods:**

1. **File Download** (Primary method):
   - Register at https://www.emdat.be/
   - Download Excel or JSON file from Public Data section
   - Place file in `data/` directory
   - System will automatically detect and load it

2. **API Access** (If available):
   - Requires API key from EM-DAT
   - Set `EMDAT_API_KEY` in `.env` file
   - System will attempt API fetch first, fallback to file loader

**Note**: EM-DAT provides GraphQL/Python APIs, but the public REST API may be limited. The file download method is more reliable and is the recommended approach.

**Fields collected:**

- Disaster ID, event name, date
- Country, location, coordinates
- Disaster type/subtype
- Deaths, injured, affected, homeless
- Economic damage (USD)

**Data Quality Notes:**
- Pre-2000 data is marked as "Historic" and may have lower quality
- Recent data (2000+) has better completeness and accuracy
- Some fields may be missing for older events

### 2. Open-Meteo Archive API

- **Provider**: Open-Meteo
- **URL**: https://archive-api.open-meteo.com/
- **Coverage**: Global historical weather data
- **Update Frequency**: Daily
- **Access**: Free, no registration needed

**Variables collected:**

- Temperature (max, min, mean)
- Precipitation (rain, snow, total)
- Wind (speed, gusts, direction)
- Solar radiation
- Evapotranspiration

### 3. Local Historical Data

- **File**: `data/data_disaster.xlsx`
- **Coverage**: Regional disasters
- **Used for**: Baseline validation and gap-filling

---

## Feature Dictionary

### Temporal Features

| Feature                  | Description                                  | Type  |
| ------------------------ | -------------------------------------------- | ----- |
| `year`                   | Calendar year                                | int   |
| `month`                  | Calendar month (1-12)                        | int   |
| `day_of_year`            | Day of year (1-365)                          | int   |
| `day_of_week`            | Day of week (0=Monday)                       | int   |
| `quarter`                | Calendar quarter (1-4)                       | int   |
| `season`                 | Season (summer/autumn/winter/spring)         | str   |
| `month_sin`, `month_cos` | Cyclic encoding of month                     | float |
| `is_rainy_season`        | Southern Africa rainy season (Oct-Mar)       | int   |
| `days_since_last_event`  | Days since previous disaster in same country | int   |

### Geographic Features

| Feature                  | Description               | Type  |
| ------------------------ | ------------------------- | ----- |
| `is_southern_hemisphere` | 1 if latitude < 0         | int   |
| `distance_to_equator`    | Absolute latitude         | float |
| `latitude_band`          | Categorized latitude zone | str   |
| `is_likely_coastal`      | Rough coastal indicator   | int   |
| `grid_cell`              | 2° x 2° grid cell ID      | str   |

### Pattern Features

| Feature                   | Description                             | Type  |
| ------------------------- | --------------------------------------- | ----- |
| `disaster_type_frequency` | Total count of this disaster type       | int   |
| `type_avg_deaths`         | Average deaths for this disaster type   | float |
| `type_avg_affected`       | Average affected for this disaster type | float |
| `country_event_count`     | Total disasters in this country         | int   |
| `events_per_year`         | Annual disaster count for country       | int   |

### Aggregation Features

| Feature           | Description                 | Type |
| ----------------- | --------------------------- | ---- |
| `event_count_7d`  | Events in past 7 days       | int  |
| `event_count_14d` | Events in past 14 days      | int  |
| `event_count_30d` | Events in past 30 days      | int  |
| `deaths_sum_7d`   | Total deaths in past 7 days | int  |

### Interaction Features

| Feature                     | Description                     | Type  |
| --------------------------- | ------------------------------- | ----- |
| `rainy_coastal_interaction` | Rainy season × coastal location | int   |
| `total_impact`              | deaths × 100 + affected         | int   |
| `severity_score`            | log(1 + total_impact)           | float |

---

## Data Quality Metrics

Quality is assessed across 5 dimensions:

1. **Completeness** (25%): Percentage of non-null values
2. **Validity** (25%): Values within expected ranges
3. **Consistency** (20%): Cross-field logical checks
4. **Accuracy Proxy** (15%): Absence of obvious errors
5. **Coverage** (15%): Geographic and temporal diversity

**Target**: Overall score ≥ 75%

---

## Known Limitations

1. **Coordinate Precision**: Some EM-DAT records have country-level coordinates only
2. **Historical Gaps**: Pre-2000 data is less comprehensive
3. **Delayed Reporting**: Recent disasters may take weeks to appear in EM-DAT
4. **Impact Estimates**: Death/affected counts are often rounded estimates
5. **Weather Resolution**: Daily weather data; hourly not available historically

---

## Data Pipeline Architecture

```
EM-DAT API ──┐
             ├──► raw/    ──► DataCleaner ──► FeatureEngineer ──► processed/
Open-Meteo ──┤                                                        │
             │                                                        ▼
Local Excel ─┘                                        DataQualityMonitor
                                                              │
                                                              ▼
                                                   quality_reports.jsonl
```

---

## Usage Examples

```python
from services.data_storage import DataStorage
from services.feature_engineering import FeatureEngineer

# Load processed data
storage = DataStorage()
df = storage.load_processed("disasters_master")

# Get feature names
engineer = FeatureEngineer()
features = engineer.get_numeric_features(df)

# Use for ML
X = df[features]
y = df["severity_score"]
```

---

## References

- EM-DAT: https://www.emdat.be/
- Open-Meteo: https://open-meteo.com/
- CRED Disaster Classification: https://www.emdat.be/classification
