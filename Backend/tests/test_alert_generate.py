import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from services.alert_generate import generate_alerts, translate_with_gemini, generate_alerts_from_db
from datetime import datetime

class TestAlertGeneration:
    def test_generate_alerts_high_risk(self, sample_prediction_data):
        """Test that high risk alerts are generated correctly"""
        # Filter for high risk data only
        high_risk_data = sample_prediction_data[sample_prediction_data['risk_category'] == 'High']
        
        alerts = generate_alerts(high_risk_data)
        
        assert len(alerts) > 0
        for alert in alerts:
            assert alert['risk_level'].lower() == 'high'
            assert 'HIGH RISK' in alert['message']
            assert alert['location'] in high_risk_data['location'].values
    
    def test_generate_alerts_moderate_risk(self):
        """Test that moderate risk alerts are generated correctly"""
        # Create a dataframe with moderate risk data
        moderate_data = pd.DataFrame({
            'location': ['Moderate Risk Location'],
            'composite_risk_score': [55.0],
            'risk_category': ['Medium'],  # Note: code uses 'medium' for moderate risk
            'wave_height': [0.8],
            'household_resources': [
                {'food_packs': 8, 'water_gallons': 16, 'shelter_needed': False, 'boats_needed': 1}
            ],
            'timestamp': ['2023-10-15 12:00:00']
        })
        
        alerts = generate_alerts(moderate_data)
        
        assert len(alerts) > 0
        for alert in alerts:
            assert alert['risk_level'].lower() == 'moderate'
            assert 'MODERATE RISK' in alert['message']
            assert alert['location'] in moderate_data['location'].values
    
    def test_generate_alerts_low_risk(self):
        """Test that low risk data doesn't generate alerts"""
        # Create a dataframe with low risk data
        low_risk_data = pd.DataFrame({
            'location': ['Low Risk Location'],
            'composite_risk_score': [25.0],
            'risk_category': ['Low'],
            'wave_height': [0.3],
            'household_resources': [
                {'food_packs': 4, 'water_gallons': 8, 'shelter_needed': False, 'boats_needed': 0}
            ],
            'timestamp': ['2023-10-15 12:00:00']
        })
        
        alerts = generate_alerts(low_risk_data)
        
        # Low risk should not generate alerts
        assert len(alerts) == 0
    
    def test_generate_alerts_empty_data(self):
        """Test that empty data doesn't generate alerts"""
        empty_data = pd.DataFrame()
        
        alerts = generate_alerts(empty_data)
        
        assert len(alerts) == 0

    @patch('services.alert_generate.translate_with_gemini')
    def test_alert_translation(self, mock_translate):
        """Test that alerts are translated correctly"""
        # Mock the translation function
        mock_translate.return_value = "⚠️ [ALTO RIESGO] Clima severo en Test Location."
        
        # Create a dataframe with high risk data
        high_risk_data = pd.DataFrame({
            'location': ['Test Location'],
            'composite_risk_score': [75.0],
            'risk_category': ['High'],
            'wave_height': [1.2],
            'household_resources': [
                {'food_packs': 12, 'water_gallons': 24, 'shelter_needed': True, 'boats_needed': 2}
            ],
            'timestamp': ['2023-10-15 12:00:00']
        })
        
        # Generate alerts with translation
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'fake-key'}):
            alerts = generate_alerts(high_risk_data)
            
            # Verify translation was called
            mock_translate.assert_called()

    @patch('services.alert_generate.genai')
    def test_translate_with_gemini(self, mock_genai):
        """Test the translation function with Gemini API"""
        # Mock the Gemini client and response
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        
        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_content = MagicMock()
        mock_content.parts = [MagicMock(text="Texto traducido")]
        mock_candidate.content = mock_content
        mock_response.candidates = [mock_candidate]
        
        mock_client.models.generate_content.return_value = mock_response
        
        # Test translation
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'fake-key'}):
            result = translate_with_gemini("Test text", "Spanish")
            
            assert result == "Texto traducido"
            mock_client.models.generate_content.assert_called_once()

    @patch('services.alert_generate.genai', None)
    def test_translate_fallback_when_no_genai(self):
        """Test that translation falls back to original text when genai is not available"""
        original_text = "Test text without translation"
        result = translate_with_gemini(original_text, "Spanish")
        
        # Should return original text when genai is not available
        assert result == original_text

    @patch('services.alert_generate.generate_alerts')
    async def test_generate_alerts_from_db(self, mock_generate_alerts, mock_db):
        """Test generating alerts from database"""
        # Mock the database query result
        mock_cursor = MagicMock()
        mock_cursor.__aiter__.return_value = [
            {'location': 'Test Location', 'composite_risk_score': 75.0, 'risk_category': 'High'}
        ]
        mock_db.predictions.find.return_value = mock_cursor
        
        # Mock the generate_alerts function
        mock_generate_alerts.return_value = [
            {'location': 'Test Location', 'risk_level': 'high', 'message': 'Test alert'}
        ]
        
        # Call the function
        result = await generate_alerts_from_db(mock_db)
        
        # Verify results
        assert len(result) > 0
        mock_db.predictions.find.assert_called_once()
        mock_generate_alerts.assert_called_once()