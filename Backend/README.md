# Crisis Connect API

A comprehensive flood risk prediction and alerting system built with FastAPI, featuring real-time weather data collection, machine learning-based risk assessment, and multi-language alert generation.

## 🏗️ Project Structure

```
Backend/
├── 📁 config/                    # Configuration files
│   ├── dev.env                   # Development environment template
│   ├── env.example               # Environment configuration example
│   ├── docker-compose.yml        # Docker development setup
│   └── Dockerfile.improved       # Production Docker configuration
│
├── 📁 data/                      # Data files and ML models
│   ├── rf_model.pkl             # Trained Random Forest model
│   ├── data_disaster.xlsx        # Historical disaster data
│   ├── latest_data.csv          # Latest weather data
│   ├── weather_data_scored.csv   # Scored weather data
│   ├── alerts_log.csv           # Alert history
│   └── weather_risk_map.html    # Risk visualization
│
├── 📁 docs/                      # Documentation
│   ├── DEPLOYMENT.md            # Deployment guide
│   └── TESTING_SUMMARY.md       # Testing documentation
│
├── 📁 middleware/               # Custom middleware
│   ├── __init__.py
│   └── logging_middleware.py    # Request/response logging
│
├── 📁 models/                   # Pydantic data models
│   └── model.py                 # API request/response models
│
├── 📁 scripts/                  # Utility scripts
│   ├── simple_test.py           # Basic functionality test
│   ├── test_offline.py          # Component testing
│   ├── test_backend.py          # Comprehensive test suite
│   ├── test_backend.bat         # Windows test script
│   └── start_dev.py             # Development server startup
│
├── 📁 services/                 # Business logic services
│   ├── alert_generate.py        # Alert generation logic
│   ├── alert_service.py         # Alert management service
│   ├── predict.py               # Prediction algorithms
│   └── weather_service.py       # Weather data service
│
├── 📁 tests/                    # Test suite
│   ├── __init__.py
│   ├── conftest.py              # Test configuration
│   ├── requirements-test.txt    # Test dependencies
│   ├── test_alert_generate.py   # Alert generation tests
│   ├── test_db.py               # Database tests
│   ├── test_improved_api.py     # API endpoint tests
│   ├── test_main.py             # Main application tests
│   ├── test_models.py           # Model tests
│   ├── test_predict.py          # Prediction tests
│   └── tests_flow.py            # Integration tests
│
├── 📁 utils/                    # Utility functions
│   └── db.py                    # Database utilities
│
├── main.py                      # Main FastAPI application
├── config.py                    # Configuration management
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Copy environment template
cp config/dev.env .env

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Tests
```bash
# Test all components
python scripts/test_offline.py

# Run comprehensive tests
python scripts/test_backend.py
```

### 3. Start Development Server
```bash
# Option 1: Direct Python
python main.py

# Option 2: Using Uvicorn
uvicorn main:app --reload

# Option 3: Using development script
python scripts/start_dev.py
```

### 4. Access API
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Interactive Docs**: http://localhost:8000/redoc

## 📋 Key Features

### 🌤️ Weather Data Collection
- Real-time weather data from Open-Meteo API
- Marine weather data integration
- Historical data analysis
- Custom location support

### 🤖 Machine Learning
- Random Forest flood risk prediction
- Real-time risk scoring
- Historical pattern analysis
- Anomaly detection

### 🚨 Alert System
- Multi-language alert generation (English, Afrikaans, Zulu)
- Risk level classification (LOW, MODERATE, HIGH)
- Historical alert tracking
- Alert statistics and analytics

### 🔒 Security & Performance
- API key authentication
- Rate limiting
- Redis caching
- CORS configuration
- Structured logging

## 🔧 Configuration

### Environment Variables
Key configuration options in `.env`:

```bash
# API Configuration
DEBUG=true
API_VERSION=1.0.0

# Database
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=crisis_connect

# Redis (Optional)
REDIS_URL=redis://localhost:6379

# Security
API_KEY=your-secret-key
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# ML Models
MODEL_PATH=data/rf_model.pkl
HISTORICAL_DATA_PATH=data/data_disaster.xlsx
```

## 🧪 Testing

### Component Tests
```bash
# Test individual components
python scripts/simple_test.py

# Test offline functionality
python scripts/test_offline.py
```

### Integration Tests
```bash
# Run full test suite
python scripts/test_backend.py

# Run specific test modules
python -m pytest tests/test_models.py
```

## 📊 API Endpoints

### Core Endpoints
- `GET /` - API information
- `GET /health` - System health check
- `GET /metrics` - Performance metrics

### Weather & Prediction
- `GET /collect` - Collect weather data
- `POST /collect` - Collect data for custom locations
- `GET /risk-assessment` - Get latest risk assessments
- `POST /predict` - Run risk prediction

### Alerts
- `POST /alerts` - Create new alert
- `GET /alerts/history` - Get alert history
- `GET /alerts/statistics` - Alert analytics
- `POST /alerts/generate` - Generate alerts from predictions

### Resources
- `GET /resources` - Calculate household resources
- `POST /resources/calculate` - Calculate for specific location

## 🐳 Docker Deployment

### Development
```bash
# Start all services
docker-compose -f config/docker-compose.yml up -d

# View logs
docker-compose -f config/docker-compose.yml logs -f
```

### Production
```bash
# Build production image
docker build -f config/Dockerfile.improved -t crisis-connect-api .

# Run container
docker run -p 8000:8000 crisis-connect-api
```

## 📈 Monitoring

### Health Checks
- **System Health**: `/health`
- **Service Status**: MongoDB, Redis, ML Model, External APIs
- **Performance Metrics**: `/metrics`

### Logging
- Structured JSON logging with request tracking
- Request/response middleware for monitoring
- Error tracking and alerting

## 🔄 Development Workflow

1. **Setup**: Copy `config/dev.env` to `.env`
2. **Test**: Run `python scripts/test_offline.py`
3. **Develop**: Make changes to services/models
4. **Test**: Run `python scripts/test_backend.py`
5. **Start**: Run `python main.py`
6. **Deploy**: Use Docker or cloud deployment

## 📚 Documentation

- **[Deployment Guide](docs/DEPLOYMENT.md)** - Production deployment instructions
- **[Testing Summary](docs/TESTING_SUMMARY.md)** - Test results and setup
- **[API Documentation](http://localhost:8000/docs)** - Interactive API docs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python scripts/test_backend.py`
5. Submit a pull request

## 📄 License

This project is part of the Crisis Connect system for disaster management and flood risk prediction.

---

**Status**: ✅ Production Ready  
**Version**: 1.0.0  
**Last Updated**: January 2025