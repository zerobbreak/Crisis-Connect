import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import FastAPI
from utils.db import init_mongo, close_mongo, get_db, ensure_indexes

@pytest.mark.asyncio
class TestDatabaseUtils:
    @patch('utils.db.AsyncIOMotorClient')
    async def test_init_mongo(self, mock_client):
        """Test initializing MongoDB connection"""
        # Setup mock client
        mock_instance = MagicMock()
        mock_instance.admin.command = AsyncMock()
        mock_client.return_value = mock_instance
        
        # Create test app
        app = FastAPI()
        
        # Call init_mongo
        await init_mongo(app)
        
        # Verify client was created and ping was called
        mock_client.assert_called_once()
        mock_instance.admin.command.assert_called_once_with("ping")
        
        # Verify app state was updated
        assert hasattr(app.state, "mongo_client")
        assert hasattr(app.state, "db")
        assert app.state.mongo_client == mock_instance
        assert app.state.db == mock_instance["crisis_connect"]
    
    async def test_close_mongo(self):
        """Test closing MongoDB connection"""
        # Create test app with mock client
        app = FastAPI()
        mock_client = MagicMock()
        mock_client.close = MagicMock()
        app.state.mongo_client = mock_client
        app.state.db = MagicMock()
        
        # Call close_mongo
        await close_mongo(app)
        
        # Verify client was closed and state was cleared
        mock_client.close.assert_called_once()
        assert app.state.mongo_client is None
        assert app.state.db is None
    
    async def test_close_mongo_no_client(self):
        """Test closing MongoDB when no client exists"""
        # Create test app without client
        app = FastAPI()
        
        # Call close_mongo (should not raise exception)
        await close_mongo(app)
    
    def test_get_db(self):
        """Test getting database from app state"""
        # Create test app with mock db
        app = FastAPI()
        mock_db = MagicMock()
        app.state.db = mock_db
        
        # Call get_db
        db = get_db(app)
        
        # Verify correct db was returned
        assert db == mock_db
    
    def test_get_db_not_initialized(self):
        """Test getting database when not initialized"""
        # Create test app without db
        app = FastAPI()
        
        # Call get_db (should raise exception)
        with pytest.raises(RuntimeError, match="MongoDB is not initialized"):
            get_db(app)
    
    @patch('utils.db.AsyncIOMotorDatabase')
    async def test_ensure_indexes(self, mock_db_class):
        """Test ensuring indexes are created"""
        # Setup mock collections
        mock_db = MagicMock(spec=mock_db_class)
        mock_db.weather_data = MagicMock()
        mock_db.weather_data.create_index = AsyncMock()
        mock_db.predictions = MagicMock()
        mock_db.predictions.create_index = AsyncMock()
        mock_db.alerts = MagicMock()
        mock_db.alerts.create_index = AsyncMock()
        mock_db.historical_events = MagicMock()
        mock_db.historical_events.create_index = AsyncMock()
        mock_db.historical_summary = MagicMock()
        mock_db.historical_summary.create_index = AsyncMock()
        mock_db.location_risks = MagicMock()
        mock_db.location_risks.create_index = AsyncMock()
        
        # Call ensure_indexes
        await ensure_indexes(mock_db)
        
        # Verify indexes were created
        assert mock_db.weather_data.create_index.call_count >= 1
        assert mock_db.predictions.create_index.call_count >= 1
        assert mock_db.alerts.create_index.call_count >= 1
        assert mock_db.historical_events.create_index.call_count >= 1
        assert mock_db.historical_summary.create_index.call_count >= 1
        assert mock_db.location_risks.create_index.call_count >= 1
    
    @patch('utils.db.AsyncIOMotorDatabase')
    async def test_ensure_indexes_handles_exceptions(self, mock_db_class):
        """Test that ensure_indexes handles exceptions when creating unique indexes"""
        # Setup mock collections with one that raises exception
        mock_db = MagicMock(spec=mock_db_class)
        mock_db.weather_data = MagicMock()
        mock_db.weather_data.create_index = AsyncMock(side_effect=[None, Exception("Duplicate key error")])
        mock_db.predictions = MagicMock()
        mock_db.predictions.create_index = AsyncMock()
        mock_db.alerts = MagicMock()
        mock_db.alerts.create_index = AsyncMock()
        mock_db.historical_events = MagicMock()
        mock_db.historical_events.create_index = AsyncMock()
        mock_db.historical_summary = MagicMock()
        mock_db.historical_summary.create_index = AsyncMock()
        mock_db.location_risks = MagicMock()
        mock_db.location_risks.create_index = AsyncMock()
        
        # Call ensure_indexes (should not raise exception)
        await ensure_indexes(mock_db)
        
        # Verify function continued after exception
        assert mock_db.predictions.create_index.called