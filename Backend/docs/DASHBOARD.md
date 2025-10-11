# Crisis Connect Admin Dashboard

A modern, professional Streamlit-based admin dashboard for managing the Crisis Connect flood risk prediction and alert system.

## 🌟 Features

### 🎨 Modern Design
- **Professional UI**: Clean, modern interface with custom CSS styling
- **Responsive Layout**: Optimized for desktop and tablet viewing
- **Dark/Light Theme**: Adaptive color scheme
- **Interactive Charts**: Plotly-based visualizations with hover effects
- **Real-time Updates**: Live data refresh with caching

### 📊 Dashboard Overview
- **System Health Monitoring**: Real-time status of all services
- **Key Metrics**: Total alerts, recent activity, risk levels, uptime
- **Risk Distribution Charts**: Pie charts and bar graphs for risk analysis
- **Alert Trends**: Time-series visualization of alert patterns
- **Recent Alerts Table**: Latest alerts with filtering and sorting

### 🚨 Alert Management
- **Advanced Filtering**: Filter by location, risk level, language, and date
- **Alert Statistics**: Comprehensive analytics on alert patterns
- **Bulk Operations**: Mass alert generation and management
- **Export Capabilities**: Download alert data for analysis

### 🌤️ Weather Data
- **Real-time Collection**: Manual weather data collection for all locations
- **Custom Locations**: Add custom coordinates for weather data
- **Data Visualization**: Weather patterns and trends
- **Historical Analysis**: Long-term weather data insights

### 📈 Analytics & Reports
- **Alert Analytics**: Risk level distribution and trends
- **Location Analytics**: Top locations by alert count
- **System Performance**: API response times, database metrics
- **Custom Reports**: Generate reports for specific time periods

### ⚙️ System Settings
- **API Configuration**: Configure backend API endpoints
- **Cache Management**: Control data caching duration
- **Connection Testing**: Verify API connectivity
- **System Information**: View detailed system status

## 🚀 Quick Start

### Option 1: Python Script
```bash
cd Backend
python scripts/start_dashboard.py
```

### Option 2: Direct Streamlit
```bash
cd Backend
streamlit run dashboard.py
```

### Option 3: Windows Batch File
```bash
# Double-click or run from command line
scripts/start_dashboard.bat
```

### Option 4: Docker (Coming Soon)
```bash
docker-compose up dashboard
```

## 📋 Prerequisites

### Required Packages
```bash
pip install streamlit plotly requests pandas
```

### Backend API
The dashboard works best when connected to the Crisis Connect API:
```bash
# Start the API first
python main.py

# Then start the dashboard
python scripts/start_dashboard.py
```

## 🎯 Navigation

### 📊 Dashboard Tab
- **System Overview**: Key metrics and system health
- **Risk Distribution**: Visual breakdown of risk levels
- **Alert Trends**: Time-series charts of alert activity
- **Recent Alerts**: Latest alerts in a sortable table

### 🚨 Alerts Tab
- **Alert Filters**: Filter by location, risk level, language
- **Alert Statistics**: Counts and distributions
- **Alert Table**: Detailed view of all alerts
- **Bulk Actions**: Mass operations on alerts

### 🌤️ Weather Data Tab
- **Data Collection**: Manual weather data collection
- **Custom Locations**: Add specific coordinates
- **Weather Visualization**: Charts and graphs
- **Historical Data**: Long-term weather analysis

### 📈 Analytics Tab
- **Alert Analytics**: Risk level and location analysis
- **System Performance**: API and database metrics
- **Custom Reports**: Generate specific reports
- **Export Options**: Download data for external analysis

### ⚙️ Settings Tab
- **API Configuration**: Backend connection settings
- **Cache Settings**: Data caching configuration
- **System Information**: Detailed system status
- **Quick Actions**: Cache clearing, connection testing

## 🎨 Customization

### Theme Colors
The dashboard uses a modern color palette:
- **Primary**: Blue (#2563eb)
- **Secondary**: Dark Blue (#1e40af)
- **Success**: Green (#10b981)
- **Warning**: Amber (#f59e0b)
- **Danger**: Red (#ef4444)

### Custom CSS
All styling is contained in the dashboard.py file and can be customized by modifying the CSS in the `st.markdown()` section.

### Chart Themes
Charts use Plotly's modern themes and can be customized by modifying the `fig.update_layout()` calls.

## 🔧 Configuration

### Environment Variables
```bash
# API Configuration
API_BASE_URL=http://localhost:8000

# Dashboard Settings
DASHBOARD_PORT=8501
DASHBOARD_HOST=0.0.0.0
```

### Cache Settings
- **Default Cache Duration**: 5 minutes (300 seconds)
- **Configurable**: Adjustable in the Settings tab
- **Auto-refresh**: Manual refresh button available

## 📊 Data Sources

### API Endpoints Used
- `GET /health` - System health status
- `GET /metrics` - System performance metrics
- `GET /alerts/history` - Alert history
- `GET /risk-assessment` - Risk assessment data
- `GET /collect` - Weather data collection
- `POST /alerts/generate` - Alert generation

### Data Refresh
- **Automatic**: Every 5 minutes (configurable)
- **Manual**: Refresh button in sidebar
- **Real-time**: Some metrics update in real-time

## 🚀 Advanced Features

### Real-time Monitoring
- System health indicators
- Service status monitoring
- Performance metrics tracking
- Alert generation monitoring

### Data Export
- CSV export of alert data
- JSON export of system metrics
- Chart image downloads
- Custom report generation

### Multi-language Support
- English, isiZulu, isiXhosa alert filtering
- Language-specific analytics
- Localized interface elements

### Responsive Design
- Mobile-friendly layout
- Tablet optimization
- Desktop full-featured view
- Adaptive sidebar

## 🔒 Security

### Access Control
- No authentication built-in (configure at server level)
- API key support for backend connections
- Secure data transmission (HTTPS recommended)

### Data Privacy
- No data stored locally
- All data fetched from API
- Cache clears automatically
- No persistent user data

## 🐛 Troubleshooting

### Common Issues

#### Dashboard Won't Start
```bash
# Check Python installation
python --version

# Install dependencies
pip install streamlit plotly requests pandas

# Check port availability
netstat -an | findstr 8501
```

#### API Connection Issues
```bash
# Check if API is running
curl http://localhost:8000/health

# Start API if needed
python main.py
```

#### Chart Display Issues
```bash
# Update Plotly
pip install --upgrade plotly

# Clear browser cache
# Or try incognito/private browsing
```

#### Performance Issues
- Reduce cache duration in Settings
- Limit data queries in API
- Use filters to reduce data volume
- Clear browser cache

### Debug Mode
```bash
# Enable debug logging
streamlit run dashboard.py --logger.level=debug
```

## 📈 Performance

### Optimization Tips
- Use filters to limit data queries
- Adjust cache duration based on needs
- Close unused browser tabs
- Use modern browser with hardware acceleration

### System Requirements
- **RAM**: 2GB minimum, 4GB recommended
- **CPU**: Modern multi-core processor
- **Browser**: Chrome, Firefox, Safari, Edge (latest versions)
- **Network**: Stable internet connection for API calls

## 🔄 Updates

### Version History
- **v1.0.0**: Initial release with core features
- **v1.1.0**: Added analytics and reporting
- **v1.2.0**: Enhanced weather data visualization
- **v1.3.0**: Mobile responsiveness improvements

### Upcoming Features
- Real-time WebSocket connections
- Advanced filtering and search
- Custom dashboard layouts
- Data export enhancements
- User authentication
- Multi-tenant support

## 📞 Support

### Documentation
- [Main README](../README.md)
- [API Documentation](http://localhost:8000/docs)
- [Deployment Guide](DEPLOYMENT.md)

### Issues
- Report bugs via GitHub issues
- Feature requests welcome
- Community support available

---

**Dashboard Version**: 1.0.0  
**Last Updated**: January 2025  
**Compatibility**: Crisis Connect API v1.0+
