# Crisis Connect API - Deployment Guide

This guide covers deploying the improved Crisis Connect API in various environments.

## Quick Start with Docker Compose

The easiest way to deploy the API is using the provided Docker Compose setup:

```bash
# Clone the repository
git clone <repository-url>
cd Backend

# Copy environment configuration
cp env.example .env
# Edit .env with your configuration

# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f crisis-connect-api

# Access services
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
# MongoDB Express: http://localhost:8081
# Redis Commander: http://localhost:8082
```

## Environment Configuration

### Required Environment Variables

```bash
# Database
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=crisis_connect

# Security (Required for production)
API_KEY=your-secret-api-key-here
CORS_ORIGINS=https://yourdomain.com
TRUSTED_HOSTS=yourdomain.com
```

### Optional Environment Variables

```bash
# Redis (for caching and rate limiting)
REDIS_URL=redis://localhost:6379

# External APIs
GEMINI_API_KEY=your-gemini-api-key

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# Logging
LOG_LEVEL=INFO
DEBUG=false
```

## Production Deployment

### 1. Cloud Platform Deployment (Render/Railway/Vercel)

#### Backend Deployment

1. **Connect Repository**: Link your GitHub repository
2. **Set Environment Variables**:
   ```bash
   MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/
   API_KEY=your-production-api-key
   DEBUG=false
   CORS_ORIGINS=https://yourdomain.com
   ```
3. **Upload Model Files**: Ensure `rf_model.pkl` and `data_disaster.xlsx` are in the repository
4. **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

#### Database Setup

**MongoDB Atlas**:
1. Create a MongoDB Atlas cluster
2. Whitelist your deployment platform's IP ranges
3. Create a database user with read/write permissions
4. Use the connection string in `MONGODB_URI`

**Redis** (Optional):
1. Use a managed Redis service (Redis Cloud, AWS ElastiCache)
2. Configure connection string in `REDIS_URL`

### 2. Kubernetes Deployment

Create the following Kubernetes manifests:

#### Namespace
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: crisis-connect
```

#### ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: crisis-connect-config
  namespace: crisis-connect
data:
  MONGODB_URI: "mongodb://mongodb:27017"
  MONGODB_DB: "crisis_connect"
  REDIS_URL: "redis://redis:6379"
  LOG_LEVEL: "INFO"
  DEBUG: "false"
```

#### Secret
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: crisis-connect-secrets
  namespace: crisis-connect
type: Opaque
data:
  API_KEY: <base64-encoded-api-key>
  GEMINI_API_KEY: <base64-encoded-gemini-key>
```

#### Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: crisis-connect-api
  namespace: crisis-connect
spec:
  replicas: 3
  selector:
    matchLabels:
      app: crisis-connect-api
  template:
    metadata:
      labels:
        app: crisis-connect-api
    spec:
      containers:
      - name: api
        image: crisis-connect-api:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: crisis-connect-config
        - secretRef:
            name: crisis-connect-secrets
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### Service
```yaml
apiVersion: v1
kind: Service
metadata:
  name: crisis-connect-api
  namespace: crisis-connect
spec:
  selector:
    app: crisis-connect-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 3. Manual Server Deployment

#### Prerequisites
- Python 3.11+
- MongoDB 7.0+
- Redis 7.0+ (optional)
- Nginx (for reverse proxy)

#### Installation Steps

1. **Clone and Setup**:
   ```bash
   git clone <repository-url>
   cd Backend
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Environment Configuration**:
   ```bash
   cp env.example .env
   # Edit .env with your settings
   ```

3. **Database Setup**:
   ```bash
   # Start MongoDB
   sudo systemctl start mongod
   
   # Start Redis (optional)
   sudo systemctl start redis
   ```

4. **Run Application**:
   ```bash
   # Development
   uvicorn main:app --reload
   
   # Production with Gunicorn
   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
   ```

5. **Nginx Configuration**:
   ```nginx
   server {
       listen 80;
       server_name yourdomain.com;
       
       location / {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

## Monitoring and Maintenance

### Health Checks

The API provides comprehensive health checks:

- **Health Endpoint**: `GET /health`
- **Metrics Endpoint**: `GET /metrics`

### Logging

Structured JSON logging is enabled by default:

```bash
# View logs
docker-compose logs -f crisis-connect-api

# Or in production
tail -f /var/log/crisis-connect-api.log
```

### Database Maintenance

```bash
# Connect to MongoDB
mongo mongodb://localhost:27017/crisis_connect

# Clean up old alerts (keep 90 days)
curl -X DELETE "http://localhost:8000/alerts/cleanup?days_to_keep=90" \
  -H "Authorization: Bearer your-api-key"

# Get database statistics
curl "http://localhost:8000/metrics"
```

### Performance Monitoring

Monitor these key metrics:

1. **Response Times**: Track API response times
2. **Error Rates**: Monitor 4xx/5xx error rates
3. **Database Performance**: Monitor MongoDB query performance
4. **Cache Hit Rates**: Monitor Redis cache effectiveness
5. **Resource Usage**: CPU, memory, disk usage

### Scaling Considerations

1. **Horizontal Scaling**: Deploy multiple API instances behind a load balancer
2. **Database Scaling**: Use MongoDB replica sets for read scaling
3. **Cache Scaling**: Use Redis Cluster for distributed caching
4. **Rate Limiting**: Adjust rate limits based on usage patterns

## Security Best Practices

1. **API Keys**: Use strong, unique API keys and rotate regularly
2. **HTTPS**: Always use HTTPS in production
3. **CORS**: Restrict CORS origins to your domains only
4. **Rate Limiting**: Enable rate limiting to prevent abuse
5. **Input Validation**: All inputs are validated using Pydantic
6. **Database Security**: Use MongoDB authentication and network restrictions
7. **Secrets Management**: Use proper secrets management (Kubernetes secrets, AWS Secrets Manager, etc.)

## Troubleshooting

### Common Issues

1. **Model Not Found**: Ensure `rf_model.pkl` is in the correct location
2. **Database Connection**: Check MongoDB URI and network connectivity
3. **Redis Connection**: Redis is optional; API works without it
4. **API Key Issues**: Ensure API key is set correctly for protected endpoints
5. **CORS Issues**: Check CORS origins configuration

### Debug Mode

Enable debug mode for detailed error information:

```bash
DEBUG=true LOG_LEVEL=DEBUG
```

### Log Analysis

Use structured logging for better analysis:

```bash
# Filter error logs
docker-compose logs crisis-connect-api | jq 'select(.level == "error")'

# Monitor specific operations
docker-compose logs crisis-connect-api | jq 'select(.event == "weather_data_collection")'
```

## Backup and Recovery

### Database Backup

```bash
# Backup MongoDB
mongodump --uri="mongodb://localhost:27017/crisis_connect" --out=/backup

# Restore MongoDB
mongorestore --uri="mongodb://localhost:27017/crisis_connect" /backup/crisis_connect
```

### Configuration Backup

Keep backups of:
- Environment configuration (`.env`)
- Model files (`rf_model.pkl`)
- Historical data (`data_disaster.xlsx`)
- Database indexes and schemas

## Updates and Maintenance

1. **Regular Updates**: Keep dependencies updated
2. **Security Patches**: Apply security updates promptly
3. **Model Updates**: Retrain and update ML models regularly
4. **Database Maintenance**: Regular cleanup of old data
5. **Monitoring**: Continuous monitoring of system health
