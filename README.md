# Crisis Connect

## Overview
Crisis Connect is a comprehensive AI-powered flood detection and community alert system designed for high-risk regions in South Africa, particularly KwaZulu-Natal. This multi-component platform leverages real-time weather and marine data from the Open-Meteo API to predict flood risks using machine learning, generates multilingual alerts (English, isiZulu, isiXhosa), and provides multiple user interfaces for different stakeholders.

The system consists of three main components:
- **Backend API**: FastAPI-based service with machine learning models and data processing
- **CrisisConnect Dashboard**: React Router-based interactive dashboard with maps and alerts
- **Next.js Frontend**: Modern web interface for public access and information

The system fetches data for predefined districts (e.g., eThekwini, Ugu), scores risk using a Random Forest model, and supports location-based risk assessment with geocoding. It's designed for local governments, NGOs, emergency services, and communities to act proactively against flood threats.

## Features

### Core System Capabilities
- **Real-Time Data Collection**: Fetches comprehensive weather (temperature, humidity, rainfall, wind speed) and marine (wave height, sea conditions) data from Open-Meteo API for predefined South African districts
- **AI-Powered Risk Assessment**: Pre-trained Random Forest model generates accurate risk scores and categories (Low, Moderate, High, Critical) based on multiple environmental factors
- **Multilingual Support**: Generates and manages alerts in English, isiZulu, and isiXhosa with Google Gemini API integration for accurate translations
- **Location Intelligence**: Advanced geocoding with geopy for custom locations and nearby area risk analysis
- **Historical Analysis**: Comprehensive historical disaster data integration for pattern recognition and risk profiling

### Backend Services
- **FastAPI REST API**: High-performance, auto-documented API with comprehensive endpoints for all system operations
- **Streamlit Admin Dashboard**: Interactive administrative interface for data management, system monitoring, and configuration
- **MongoDB Integration**: Scalable NoSQL database with optimized queries for real-time data and historical analysis
- **Machine Learning Pipeline**: Automated model loading, prediction scoring, and result storage with joblib serialization
- **Comprehensive Testing**: Full test suite with pytest covering all components and edge cases

### CrisisConnect Dashboard (React Router)
- **Interactive Risk Maps**: Real-time flood risk visualization using Mapbox GL JS with customizable layers and markers
- **Alert Management**: Complete alert lifecycle management with filtering, search, and status tracking
- **Data Visualization**: Dynamic charts and graphs for weather patterns, risk trends, and historical analysis
- **Offline Capability**: IndexedDB integration for offline data access and synchronization
- **Responsive Design**: Mobile-first design with TailwindCSS for optimal user experience across devices

### Public Frontend (Next.js)
- **Public Information Portal**: Accessible information about flood risks and emergency procedures
- **Real-Time Updates**: Live data feeds and alert notifications for public awareness
- **Accessibility Features**: WCAG compliant design for inclusive access
- **SEO Optimized**: Server-side rendering for better search engine visibility

### Advanced Features
- **Resource Calculator**: Estimates household resource needs during flood events
- **Risk Heatmaps**: Interactive visualization of flood risks across South African regions
- **Alert Statistics**: Comprehensive analytics on alert patterns and effectiveness
- **API Integration**: RESTful endpoints for third-party integrations and mobile applications
- **Container Support**: Docker configuration for easy deployment and scaling

## Project Structure
```
CrisisConnect/
├── Backend/                    # FastAPI backend service
│   ├── main.py                # Main FastAPI application
│   ├── dashboard.py           # Streamlit dashboard interface
│   ├── models/                # Data models and schemas
│   │   └── model.py          # Pydantic models
│   ├── services/              # Core business logic
│   │   ├── predict.py        # Risk prediction and scoring
│   │   └── alert_generate.py # Alert generation and management
│   ├── utils/                 # Utility functions
│   │   └── db.py             # Database connection and operations
│   ├── tests/                 # Comprehensive test suite
│   ├── data_disaster.xlsx     # Historical disaster data
│   ├── rf_model.pkl          # Pre-trained Random Forest model
│   └── requirements.txt       # Python dependencies
├── CrisisConnect/             # React Router dashboard
│   ├── app/                  # Application components
│   │   ├── components/       # Reusable UI components
│   │   ├── lib/             # API and database utilities
│   │   ├── pages/           # Page components
│   │   └── routes/          # Route definitions
│   ├── public/              # Static assets
│   ├── package.json         # Node.js dependencies
│   └── Dockerfile          # Container configuration
├── frontend/                 # Next.js public interface
│   ├── app/                 # Next.js app directory
│   ├── public/              # Static assets
│   └── package.json         # Node.js dependencies
├── rf_model.pkl             # Shared ML model
└── README.md                # This file
```

## Tech Stack
- **Backend**: FastAPI (Python), MongoDB (via pymongo), Streamlit for admin dashboard
- **Data Fetching**: Open-Meteo API, geopy for geocoding, Google Gemini API for translations
- **Machine Learning**: scikit-learn (Random Forest), joblib for model serialization
- **Frontend Applications**:
  - CrisisConnect: React Router with TailwindCSS, Mapbox GL JS, IndexedDB
  - Public Frontend: Next.js with TailwindCSS
- **Data Visualization**: Folium, Matplotlib, Seaborn for maps and charts
- **Other**: Requests-cache, Retry-requests for API reliability, Comprehensive logging
- **Deployment**: Docker support, Render (backend), Vercel (frontend)

## Installation

### Prerequisites
- Python 3.8+ (for backend)
- Node.js 18+ (for frontend applications)
   - MongoDB (local or Atlas)
- Git

### Quick Start
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd CrisisConnect
   ```

2. **Backend Setup**:
   ```bash
   cd Backend
   
   # Create virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
     pip install -r requirements.txt
   
   # Set up environment variables (create .env file)
   echo "MONGO_CONNECTION_STRING=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0" > .env
   echo "GEMINI_API_KEY=your_gemini_api_key_here" >> .env  # Optional
   
   # Run the FastAPI backend
     uvicorn main:app --reload
     ```
   - **API Documentation**: http://localhost:8000/docs
   - **Admin Dashboard**: `streamlit run dashboard.py` → http://localhost:8501

3. **CrisisConnect Dashboard Setup** (React Router):
   ```bash
   cd CrisisConnect
   
   # Install dependencies
   npm install
   
   # Run development server
   npm run dev
   ```
   - **Dashboard**: http://localhost:5173

4. **Public Frontend Setup** (Next.js):
   ```bash
   cd frontend
   
   # Install dependencies
     npm install
   
   # Run development server
     npm run dev
     ```
   - **Public Interface**: http://localhost:3000

### Docker Setup (Alternative)
Each component includes Docker support for containerized deployment:
```bash
# Build and run backend
cd Backend
docker build -t crisis-connect-backend .
docker run -p 8000:8000 crisis-connect-backend

# Build and run CrisisConnect dashboard
cd CrisisConnect
docker build -t crisis-connect-dashboard .
docker run -p 3000:3000 crisis-connect-dashboard
```

## Usage
1. **Collect Data**:
   - Call `/collect` to fetch and store weather data for districts (e.g., eThekwini).
   - Response: `{"message": "Data collected", "count": 5}`

2. **Generate Risk Scores**:
   - Call `/predict` to score stored weather data and update MongoDB with `risk_score` and `risk_category`.
   - Response: Scored records.

3. **Create Alert**:
   - POST to `/alerts` with body:
     ```json
     {
       "location": "eThekwini",
       "risk_level": "HIGH",
       "message": "Severe flood warning",
       "language": "English",
       "timestamp": "2025-08-07 12:00:00"
     }
     ```
   - Response: `{"message": "Alert created", "alert": {...}}`

4. **Retrieve Alerts**:
   - GET `/alerts/history?location=eThekwini&language=English&limit=10`
   - Response: Filtered alerts.

5. **Location Risk Assessment**:
   - POST to `/risk/location` with body:
     ```json
     {
       "place_name": "eThekwini",
       "is_coastal": true
     }
     ```
   - Response: Risk score, wave height, and nearby locations.

6. **Historical Data**:
   - GET `/api/historical` for all historical records.
   - GET `/api/risk/eThekwini` for risk profile.

## Testing
1. **Insert Dummy Data**:
   - Run `insert_dummy_data.py` to populate `alerts`, `weatherdata`, and `historicaldata`.

2. **Run Tests**:
   - Use Swagger UI (`http://localhost:8000/docs`) for endpoint testing.
   - Verify data in MongoDB Compass or `mongo` shell.
   - Run frontend and check alerts, map, and chart with dummy data.

3. **Edge Cases**:
   - Empty database: Call `/collect` to populate.
   - Invalid location: Test `/risk/location` with invalid `place_name` (expect 400 error).
   - Duplicate alert: Test `/alerts` with existing timestamp/location (expect 400 error).

## System Architecture

### Component Overview
```
┌─────────────────────────────────────────────────────────────────┐
│                        Crisis Connect System                    │
├─────────────────────────────────────────────────────────────────┤
│  Frontend Applications                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  Next.js Public │  │ CrisisConnect   │  │ Streamlit Admin │ │
│  │  Interface      │  │ Dashboard       │  │ Dashboard       │ │
│  │  (Port 3000)    │  │ (Port 5173)     │  │ (Port 8501)     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Backend Services                                               │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  FastAPI REST API                                          │ │
│  │  • Risk Prediction Engine                                  │ │
│  │  • Alert Management System                                 │ │
│  │  • Data Collection Service                                 │ │
│  │  • Resource Calculator                                     │ │
│  │  (Port 8000)                                               │ │
│  └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  External Services                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  Open-Meteo API │  │ Google Gemini   │  │ MongoDB Atlas   │ │
│  │  (Weather Data) │  │ (Translations)  │  │ (Data Storage)  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow
1. **Data Collection**: Open-Meteo API provides real-time weather and marine data
2. **Risk Assessment**: FastAPI processes data through Random Forest model
3. **Alert Generation**: System creates multilingual alerts based on risk levels
4. **Data Storage**: MongoDB stores all data, alerts, and historical information
5. **User Interfaces**: Multiple frontend applications provide different user experiences
6. **External Integration**: Google Gemini API handles language translations

## Deployment

### Production Deployment Options

#### Option 1: Cloud Platform Deployment
**Backend (Render/Railway/DigitalOcean)**:
```bash
# Deploy FastAPI backend
- Connect your GitHub repository
- Set environment variables:
  MONGO_CONNECTION_STRING=mongodb+srv://...
  GEMINI_API_KEY=your_key_here
- Upload model files: rf_model.pkl, data_disaster.xlsx
- Start command: uvicorn main:app --host 0.0.0.0 --port $PORT
```

**CrisisConnect Dashboard (Vercel/Netlify)**:
```bash
# Deploy React Router dashboard
- Connect repository
- Build command: npm run build
- Output directory: build/
- Environment variables: API_URL=https://your-backend-url.com
```

**Public Frontend (Vercel)**:
```bash
# Deploy Next.js frontend
- Connect repository
- Build command: npm run build
- Output directory: .next/
- Environment variables: NEXT_PUBLIC_API_URL=https://your-backend-url.com
```

#### Option 2: Docker Deployment
```bash
# Build and run all services
docker-compose up -d

# Individual service deployment
docker build -t crisis-connect-backend ./Backend
docker build -t crisis-connect-dashboard ./CrisisConnect
docker build -t crisis-connect-frontend ./frontend
```

#### Option 3: Kubernetes Deployment
```yaml
# Example Kubernetes configuration for scalable deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: crisis-connect-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: crisis-connect-backend
  template:
    spec:
      containers:
      - name: backend
        image: crisis-connect-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: MONGO_CONNECTION_STRING
          valueFrom:
            secretKeyRef:
              name: mongo-secret
              key: connection-string
```

### Environment Configuration
```bash
# Production environment variables
MONGO_CONNECTION_STRING=mongodb+srv://username:password@cluster.mongodb.net/
GEMINI_API_KEY=your_gemini_api_key
API_RATE_LIMIT=1000
LOG_LEVEL=INFO
ENVIRONMENT=production
```

### Monitoring and Maintenance
- **Health Checks**: Built-in endpoints for service monitoring
- **Logging**: Comprehensive logging with structured format
- **Metrics**: Performance monitoring and alerting
- **Backup**: Automated MongoDB backups
- **Updates**: Rolling deployment strategy for zero-downtime updates

## API Documentation

### Core Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Score weather data and generate risk assessments |
| `/alerts/history` | GET | Retrieve historical alerts with filtering |
| `/alerts` | POST | Create new alerts |
| `/resources` | POST | Calculate household resource needs |
| `/risk-assessment` | GET | Get risk assessment data for visualization |
| `/api/historical` | GET | Access historical disaster data |
| `/api/risk/{location}` | GET | Get risk profile for specific location |
| `/collect` | POST | Fetch and store weather data for districts |
| `/risk/location` | POST | Assess risk for custom locations |

### Example API Calls
```bash
# Get risk assessment for eThekwini
curl -X GET "http://localhost:8000/api/risk/eThekwini"

# Create a new alert
curl -X POST "http://localhost:8000/alerts" \
  -H "Content-Type: application/json" \
  -d '{
    "location": "eThekwini",
    "risk_level": "HIGH",
    "message": "Severe flood warning",
    "language": "English"
  }'

# Calculate resources for household
curl -X POST "http://localhost:8000/resources" \
  -H "Content-Type: application/json" \
  -d '{
    "household_size": 4,
    "flood_duration": 72,
    "risk_level": "HIGH"
  }'
```

## Troubleshooting

### Common Issues

#### Backend Issues
```bash
# MongoDB connection error
Error: pymongo.errors.ServerSelectionTimeoutError
Solution: Check MONGO_CONNECTION_STRING and network connectivity

# Model loading error
Error: FileNotFoundError: rf_model.pkl
Solution: Ensure model file is in Backend directory

# API rate limiting
Error: 429 Too Many Requests
Solution: Implement request caching or reduce API call frequency
```

#### Frontend Issues
```bash
# Build errors
Error: Module not found
Solution: Run npm install in respective directories

# API connection issues
Error: Network Error
Solution: Check API_URL configuration and backend availability

# Map loading issues
Error: Mapbox token invalid
Solution: Update Mapbox access token in environment variables
```

#### Environment Setup
```bash
# Python dependency conflicts
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall

# Node.js version issues
nvm use 18  # or appropriate version
npm install --legacy-peer-deps
```

### Performance Optimization
- **Database Indexing**: Ensure proper MongoDB indexes for frequent queries
- **API Caching**: Implement Redis for frequently accessed data
- **Frontend Optimization**: Use lazy loading and code splitting
- **Monitoring**: Set up application performance monitoring (APM)

## Security Considerations

### Production Security
- **Authentication**: Implement JWT-based authentication for API access
- **Authorization**: Role-based access control for different user types
- **HTTPS**: Use SSL certificates for all production deployments
- **Rate Limiting**: Implement API rate limiting to prevent abuse
- **Input Validation**: Sanitize all user inputs and API parameters
- **Secrets Management**: Use secure secret management for API keys and credentials

### Data Privacy
- **Data Encryption**: Encrypt sensitive data at rest and in transit
- **GDPR Compliance**: Implement data retention and deletion policies
- **Access Logging**: Log all data access for audit purposes
- **Backup Security**: Encrypt database backups

## Contributing

### Development Workflow
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Set up development environment following installation instructions
4. Make changes and ensure all tests pass: `pytest`
5. Update documentation if needed
6. Commit changes: `git commit -m "Add new feature"`
7. Push branch: `git push origin feature/new-feature`
8. Submit a pull request with detailed description

### Code Standards
- **Python**: Follow PEP 8 style guidelines
- **JavaScript/TypeScript**: Use ESLint and Prettier configurations
- **Documentation**: Update README and API docs for new features
- **Testing**: Maintain test coverage above 80%
- **Commits**: Use conventional commit messages

### Pull Request Guidelines
- Provide clear description of changes
- Include screenshots for UI changes
- Ensure all CI checks pass
- Request review from maintainers
- Update version numbers if applicable

## License
MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments
- **Open-Meteo API** for comprehensive weather and marine data
- **scikit-learn** for machine learning capabilities
- **FastAPI** and **MongoDB** for robust backend infrastructure
- **React Router** and **Next.js** for modern frontend development
- **Mapbox** for mapping and geospatial visualization
- **Google Gemini** for AI-powered translation services
- **South African Weather Service** for historical disaster data
- **Open source community** for the tools and libraries that make this project possible

## Support
For support, please:
1. Check the troubleshooting section above
2. Review existing GitHub issues
3. Create a new issue with detailed information
4. Contact the development team for urgent matters

## Roadmap
- [ ] Mobile application development
- [ ] Advanced machine learning models
- [ ] Integration with more weather data sources
- [ ] Real-time notification system
- [ ] Community reporting features
- [ ] Advanced analytics and reporting
- [ ] Multi-language support expansion
- [ ] Integration with emergency services APIs
