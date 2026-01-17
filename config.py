"""
Configuration management for Crisis Connect API
"""
import os
from typing import List, Optional
from pydantic import BaseModel, validator


class Settings(BaseModel):
    """Application settings with environment variable support"""
    
    # API Configuration
    api_title: str = "Crisis Connect API"
    api_description: str = "Real-time flood risk prediction, alerting, and resource planning"
    api_version: str = "1.0.0"
    debug: bool = False
    
    # Database Configuration
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "crisis_connect"
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379"
    redis_password: Optional[str] = None
    
    # Security Configuration
    api_key: Optional[str] = "mvp-secret-key-123"
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    trusted_hosts: List[str] = ["localhost", "127.0.0.1"]
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600
    
    # External APIs
    open_meteo_url: str = "https://api.open-meteo.com/v1/forecast"
    marine_api_url: str = "https://marine-api.open-meteo.com/v1/marine"
    gemini_api_key: Optional[str] = None
    
    # ML Model Configuration
    model_path: str = "data/rf_model.pkl"
    historical_data_path: str = "data/data_disaster.xlsx"
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Cache Configuration
    cache_ttl_seconds: int = 300
    weather_cache_ttl: int = 1800  # 30 minutes
    risk_cache_ttl: int = 300      # 5 minutes
    
    # EM-DAT API Configuration (Phase 1: Data Foundation)
    emdat_api_key: Optional[str] = None
    emdat_api_url: str = "https://public.emdat.be/api"
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @validator('trusted_hosts', pre=True)
    def parse_trusted_hosts(cls, v):
        if isinstance(v, str):
            return [host.strip() for host in v.split(',')]
        return v
    
    @validator('debug', pre=True)
    def parse_debug(cls, v):
        if isinstance(v, str):
            return v.lower() in ('true', '1', 'yes', 'on')
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()

