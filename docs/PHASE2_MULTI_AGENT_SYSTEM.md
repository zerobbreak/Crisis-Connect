# Phase 2: Multi-Agent System Documentation

**Version**: 1.0  
**Last Updated**: January 2026

---

## Overview

The Crisis Connect Multi-Agent System is an advanced disaster prediction architecture that combines multiple specialized agents to provide comprehensive risk assessments. Instead of relying on a single model, the system uses five specialized agents that each analyze different aspects of disaster risk, then combines their predictions through an ensemble coordinator.

### Key Benefits

1. **Temporal Understanding**: Moves beyond static snapshots to understand how disasters develop over time
2. **Multiple Perspectives**: Different agents catch different patterns
3. **Robust Predictions**: Ensemble approach handles individual agent failures gracefully
4. **Explainable Results**: Each agent provides reasoning for its predictions
5. **Anomaly Detection**: Catches unprecedented conditions that historical models miss

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Input Layer                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ Weather Data │    │ Disaster DB  │    │ Real-time    │          │
│  │   (14 days)  │    │ (Historical) │    │   Alerts     │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Temporal Processor                                │
│  • Converts raw data to labeled sequences                           │
│  • Creates positive examples (pre-disaster weather)                 │
│  • Creates negative examples (non-disaster periods)                 │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Multi-Agent System                               │
│                                                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │    Pattern      │  │   Progression   │  │    Anomaly      │     │
│  │   Detection     │  │    Analyzer     │  │   Detection     │     │
│  │     Agent       │  │     Agent       │  │     Agent       │     │
│  │                 │  │                 │  │                 │     │
│  │ Finds recurring │  │ Tracks stages:  │  │ Flags unusual   │     │
│  │ disaster        │  │ DORMANT →       │  │ conditions      │     │
│  │ patterns        │  │ ESCALATING →    │  │ not seen        │     │
│  │                 │  │ CRITICAL →      │  │ before          │     │
│  │                 │  │ PEAK            │  │                 │     │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘     │
│           │                    │                    │               │
│           └──────────┬─────────┴────────────────────┘               │
│                      ▼                                               │
│           ┌─────────────────────┐                                   │
│           │   Forecast Agent    │                                   │
│           │                     │                                   │
│           │ Combines:           │                                   │
│           │ • LSTM predictions  │                                   │
│           │ • Pattern matches   │                                   │
│           │ • Progression data  │                                   │
│           └──────────┬──────────┘                                   │
│                      │                                               │
│                      ▼                                               │
│           ┌─────────────────────┐                                   │
│           │     Ensemble        │                                   │
│           │    Coordinator      │                                   │
│           │                     │                                   │
│           │ • Weighted average  │                                   │
│           │ • Disagreement      │                                   │
│           │   detection         │                                   │
│           │ • Final decision    │                                   │
│           └─────────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Output Layer                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ Risk Score   │    │ Risk Level   │    │ Actionable   │          │
│  │   (0-100%)   │    │ (LOW-CRIT)   │    │ Recommend.   │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Agent Details

### 1. Pattern Detection Agent

**Location**: `agents/pattern_detection_agent.py`

**Purpose**: Discovers recurring weather patterns that precede disasters by analyzing historical sequences.

**How It Works**:
1. Normalizes historical weather sequences
2. Groups sequences by disaster type
3. Uses cosine similarity to find clusters of similar pre-disaster conditions
4. Creates "Pattern" objects representing typical pre-disaster weather signatures

**Key Methods**:
```python
# Discover patterns from historical data
patterns = agent.discover_patterns(
    sequences_data=df,
    labels=y,
    feature_columns=['rainfall', 'humidity', 'wind_speed']
)

# Match current conditions against known patterns
matches = agent.match_current_sequence(current_weather)
```

**Output**: Pattern objects with:
- Pattern ID and type
- Frequency (how often it occurred)
- Reliability (% of times it led to disaster)
- Days to disaster estimate
- Feature importance

---

### 2. Progression Analyzer Agent

**Location**: `agents/progression_analyzer_agent.py`

**Purpose**: Tracks how conditions evolve from safe to dangerous, analyzing the trajectory of change.

**Key Insight**: Just because conditions are bad doesn't mean disaster is imminent. What matters is the TRAJECTORY:
- Stable bad conditions → Lower risk
- Rapidly worsening conditions → HIGH risk
- Accelerating deterioration → CRITICAL risk

**Progression Stages**:
| Stage | Severity | Description |
|-------|----------|-------------|
| DORMANT | 0-25% | Normal conditions |
| ESCALATING | 25-50% | Conditions worsening |
| CRITICAL | 50-75% | High risk |
| PEAK | 75-100% | Maximum danger |

**Key Methods**:
```python
analysis = agent.analyze_progression(
    weather_history=df,
    hazard_type="flood",
    lookback_days=14
)

# Returns:
# - current_stage: ProgressionStage
# - severity_score: 0-1
# - worsening_velocity: rate of change
# - is_accelerating: bool
# - days_to_peak_estimate: int
```

---

### 3. Anomaly Detection Agent

**Location**: `agents/anomaly_detection_agent.py`

**Purpose**: Detects unusual weather patterns not seen before using Isolation Forest.

**Why This Matters**:
- Climate change creates unprecedented weather patterns
- Traditional models fail on "never before seen" conditions
- Anomaly detection provides early warning even when patterns don't match historical disasters

**How It Works**:
1. Trains Isolation Forest on historical "normal" weather
2. Calculates anomaly scores for new data
3. Identifies which features are most unusual

**Key Methods**:
```python
# Train on historical data
agent.train(X_train, feature_columns=['rainfall', 'temp', 'humidity'])

# Detect anomalies
result = agent.detect_single(current_weather)
# Returns: AnomalyResult with is_anomalous, score, contributing_features
```

---

### 4. Forecast Agent

**Location**: `agents/forecast_agent.py`

**Purpose**: Synthesizes predictions from multiple methods into individual forecasts.

**Methods Used**:
1. **LSTM**: Deep learning on temporal sequences
2. **Pattern Matching**: Similarity to historical disaster patterns
3. **Progression Analysis**: Current trajectory extrapolation

**Key Methods**:
```python
forecasts = agent.forecast(
    current_sequence=weather_array,
    weather_history=df,
    hazard_type="flood"
)

# Returns list of Forecast objects, each with:
# - risk_score: 0-1
# - hours_to_peak: estimated time
# - confidence: 0-1
# - method: which approach generated this
# - reasoning: human-readable explanation
```

---

### 5. Ensemble Coordinator Agent

**Location**: `agents/ensemble_coordinator_agent.py`

**Purpose**: Combines predictions from all agents into a final decision.

**Weighting Strategy**:
| Method | Default Weight |
|--------|---------------|
| LSTM | 40% |
| Pattern Matching | 30% |
| Progression | 30% |

**Key Features**:
- Adjusts weights by confidence (more confident predictions get more weight)
- Detects disagreement between methods
- Handles anomaly flags
- Generates actionable recommendations

**Output**: EnsembleDecision with:
```python
{
    "risk_score": 0.72,
    "risk_level": "HIGH",
    "hours_to_peak": 24,
    "primary_method": "lstm",
    "method_breakdown": {"lstm": 0.75, "pattern": 0.68, "progression": 0.70},
    "confidence": 0.82,
    "warnings": [],
    "recommendation": "ALERT: Activate emergency protocols..."
}
```

---

## Services

### Agent Orchestrator

**Location**: `services/agent_orchestrator.py`

**Purpose**: Coordinates all agents and provides a unified prediction interface.

**Usage**:
```python
from services.agent_orchestrator import AgentOrchestrator

# Initialize
orchestrator = AgentOrchestrator(data_dir="data")

# Train agents on historical data
stats = orchestrator.train_agents(historical_disasters_df)

# Make predictions
result = orchestrator.predict(
    current_weather=weather_df,
    location="Durban",
    hazard_type="flood"
)

print(result['prediction']['risk_level'])  # "HIGH"
print(result['prediction']['recommendation'])  # Action to take
```

---

### Temporal Data Processor

**Location**: `services/temporal_processor.py`

**Purpose**: Converts raw disaster and weather data into labeled sequences for training.

**Usage**:
```python
from services.temporal_processor import TemporalDataProcessor

processor = TemporalDataProcessor(lookback_days=14)

# From master disaster dataset
X, y, features = processor.create_sequences_from_master_dataset(disasters_df)

# With weather data
sequences_df, labels = processor.create_training_sequences(
    disaster_data=disasters_df,
    weather_data=weather_df
)
```

---

### Agent Performance Monitor

**Location**: `services/agent_performance_monitor.py`

**Purpose**: Tracks prediction performance and calculates accuracy metrics.

**Usage**:
```python
from services.agent_performance_monitor import AgentPerformanceMonitor

monitor = AgentPerformanceMonitor(log_dir="data")

# Log predictions
monitor.log_prediction("Durban", "flood", prediction_result)

# Update with actual outcome (for validation)
monitor.update_outcome("Durban", timestamp, actual_outcome=True)

# Get metrics
metrics = monitor.calculate_metrics(days=30)

# Generate report
report = monitor.generate_report(days=30)
```

---

## API Integration

### Forecast Router Extension

The multi-agent system can be integrated into the existing forecast API:

```python
# routers/forecast.py

from services.agent_orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator()

@router.get("/multi-agent/{location_id}")
async def multi_agent_forecast(location_id: str, hazard_type: str = "flood"):
    """Get multi-agent ensemble forecast"""
    
    # Get weather data for location
    weather_data = await get_recent_weather(location_id)
    
    # Run multi-agent prediction
    result = orchestrator.predict(
        current_weather=weather_data,
        location=location_id,
        hazard_type=hazard_type
    )
    
    return {
        "location_id": location_id,
        "risk_score": result['prediction']['risk_score'],
        "risk_level": result['prediction']['risk_level'],
        "hours_to_peak": result['prediction']['hours_to_peak'],
        "recommendation": result['prediction']['recommendation'],
        "method_breakdown": result['prediction']['method_breakdown'],
        "forecasts": result['forecasts'],
        "is_anomalous": result['prediction']['is_anomalous']
    }
```

---

## Training the System

### Step 1: Prepare Data

Use the Phase 1 master dataset:

```python
import pandas as pd
from services.agent_orchestrator import AgentOrchestrator

# Load master dataset
disasters_df = pd.read_csv("data/processed/disasters_master.csv")

# Initialize orchestrator
orchestrator = AgentOrchestrator(data_dir="data")
```

### Step 2: Train Agents

```python
# Train all agents
stats = orchestrator.train_agents(disasters_df)

print(f"Training completed in {stats['training_time_seconds']:.1f}s")
print(f"Patterns discovered: {stats['pattern_agent']['patterns_discovered']}")
print(f"Anomaly detector trained: {stats['anomaly_agent']['trained']}")
```

### Step 3: Save Models

```python
# Models are automatically saved during training
# Or manually save:
orchestrator.save_all_models("data/models")
```

---

## Example Prediction Output

```json
{
    "location": "Durban",
    "hazard_type": "flood",
    "timestamp": "2026-01-17T14:30:00",
    "prediction": {
        "risk_score": 0.72,
        "risk_level": "HIGH",
        "hours_to_peak": 24,
        "primary_method": "progression",
        "method_breakdown": {
            "lstm": 0.68,
            "pattern_matching": 0.75,
            "progression": 0.73
        },
        "confidence": 0.81,
        "warnings": [],
        "recommendation": "ALERT: Activate emergency protocols. Peak flood conditions expected in 24 hours. Prepare evacuation routes, alert emergency services, and notify at-risk communities.",
        "is_anomalous": false
    },
    "forecasts": [
        {
            "method": "progression",
            "risk_score": 0.73,
            "confidence": 0.85,
            "reasoning": "Conditions in CRITICAL stage (severity 68%), worsening at 15%/day"
        },
        {
            "method": "pattern_matching",
            "risk_score": 0.75,
            "confidence": 0.78,
            "reasoning": "Matches historical pattern 'flood_3' (12 occurrences, 83% led to flood)"
        }
    ],
    "explanation": "=== Risk Assessment Summary ===\nOverall Risk: 72% (HIGH)\nTime to Peak: 24 hours\n...",
    "processing_time_seconds": 1.23
}
```

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Prediction Accuracy | >70% | On validated outcomes |
| Processing Time | <5 seconds | Per prediction |
| Patterns Discovered | >10 | From historical data |
| Agent Agreement | >70% | Between methods |

---

## Troubleshooting

### Common Issues

**1. "No patterns discovered"**
- Ensure sufficient training data (100+ disaster records)
- Lower similarity threshold: `PatternDetectionAgent(similarity_threshold=0.7)`
- Check that feature columns are correctly specified

**2. "Anomaly detector not trained"**
- Call `orchestrator.train_agents()` before making predictions
- Check that training data has numeric features

**3. "Low confidence predictions"**
- Increase training data
- Check for data quality issues
- Consider adjusting ensemble weights

**4. "Method disagreement warnings"**
- This is expected for edge cases
- Review individual method predictions
- Consider expert review for critical decisions

---

## Files Reference

| File | Purpose |
|------|---------|
| `agents/__init__.py` | Agent exports |
| `agents/pattern_detection_agent.py` | Pattern discovery |
| `agents/progression_analyzer_agent.py` | Stage tracking |
| `agents/forecast_agent.py` | Multi-method forecasting |
| `agents/anomaly_detection_agent.py` | Unusual pattern detection |
| `agents/ensemble_coordinator_agent.py` | Prediction combination |
| `services/agent_orchestrator.py` | Agent coordination |
| `services/temporal_processor.py` | Data preparation |
| `services/agent_performance_monitor.py` | Metrics tracking |
| `tests/test_phase2_agents.py` | Test suite |

---

## Next Steps (Phase 3)

1. **Action-Oriented Alerts**: Integrate with notification systems
2. **Real-time Monitoring**: Continuous prediction updates
3. **Feedback Loop**: Learn from prediction outcomes
4. **Geographic Expansion**: Multi-region support
