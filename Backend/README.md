# Crisis Connect
A flood risk assessment and alert system for South African communities.

## Project Overview
Crisis Connect is a comprehensive platform designed to assess flood risks, generate alerts, and calculate resource needs for communities in South Africa. The system combines weather data, historical disaster information, and machine learning to provide accurate risk assessments and timely alerts.

## Features
- Flood Risk Assessment : Analyzes weather data to predict flood risks for different locations
- Interactive Heatmap : Visualizes flood risk levels across South Africa
- Alert System : Generates and manages flood alerts based on risk assessments
- Resource Calculator : Estimates household resource needs during flood events
- Historical Data Analysis : Provides insights based on past disaster events
- REST API : Offers programmatic access to all system features
## Project Structure
```
├── main.py                 # FastAPI 
backend application
├── dashboard.py            # Streamlit 
dashboard interface
├── models/                 # Data models 
and schemas
├── services/               # Core business 
logic
│   ├── predict.py          # Risk 
prediction and scoring
│   └── alert_generate.py   # Alert 
generation and management
├── utils/                  # Utility 
functions
│   └── db.py               # Database 
connection and operations
├── tests/                  # Test suite
├── data_disaster.xlsx      # Historical 
disaster data
├── rf_model.pkl            # Pre-trained 
machine learning model
└── requirements.txt        # Project 
dependencies
```
## Prerequisites
- Python 3.9+
- MongoDB (local or remote instance)
- OpenMeteo API access (free tier)
- Google Gemini API key (optional, for enhanced alert translations)
## Installation
1. 1.
   Clone the repository:
```
git clone https://github.com/your-username/
crisis-connect.git
cd crisis-connect
```
2. 1.
   Create and activate a virtual environment:
```
python -m venv venv
.\venv\Scripts\activate
```
3. 1.
   Install dependencies:
```
pip install -r requirements.txt
```
4. 1.
   Set up environment variables (create a .env file in the project root):
```
# Database Configuration
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=crisis_connect

# API Keys (Optional)
GEMINI_API_KEY=your_gemini_api_key_here
```
## Running the Application
### Start the Backend API
```
uvicorn main:app --host 0.0.0.0 --port 8000 
--reload
```
The API will be available at http://localhost:8000

### Start the Dashboard
```
streamlit run dashboard.py
```
The dashboard will be available at http://localhost:8501

## API Documentation
Once the backend is running, you can access the API documentation at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
## Key API Endpoints
- /predict - Score weather data and generate risk assessments
- /alerts/history - Retrieve historical alerts
- /resources - Calculate household resource needs
- /risk-assessment - Get risk assessment data for visualization
- /api/historical - Access historical disaster data
- /api/risk/{location} - Get risk profile for a specific location
## Dashboard Features
The Streamlit dashboard provides a user-friendly interface with the following tabs:

1. 1.
   Heatmap - Visualize flood risks across South Africa
2. 2.
   Recent Alerts - View recent flood alerts
3. 3.
   Resource Calculator - Calculate household resource needs
4. 4.
   Risk Predictions - View detailed risk assessment data
5. 5.
   Historical Data - Explore historical disaster information
6. 6.
   Summary - View summary statistics
7. 7.
   Alert Stats - Analyze alert patterns
## Testing
Run the test suite with pytest:

```
pip install -r tests/requirements-test.txt
pytest
```
## Security Considerations
For production deployment, consider implementing:

- Authentication and authorization
- HTTPS with proper SSL certificates
- Rate limiting
- Input validation and sanitization
- Secure database connections
## License
MIT License

## Contributors
- Your Name - Initial work
## Acknowledgments
- OpenMeteo for weather data
- MongoDB for database services
- FastAPI and Streamlit for application frameworks