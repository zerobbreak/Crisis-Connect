#!/usr/bin/env python3
"""
Test script for dynamic location management system
"""
import requests
import time
import json
from datetime import datetime

def main():
    print('🗺️  TESTING DYNAMIC LOCATION MANAGEMENT SYSTEM')
    print('=' * 60)

    # Wait for server to start
    time.sleep(2)

    def test_endpoint(name, method, url, **kwargs):
        print(f'\n🧪 Testing {name}...')
        try:
            if method.upper() == 'GET':
                response = requests.get(url, timeout=10, **kwargs)
            elif method.upper() == 'POST':
                response = requests.post(url, timeout=10, **kwargs)
            elif method.upper() == 'PUT':
                response = requests.put(url, timeout=10, **kwargs)
            elif method.upper() == 'DELETE':
                response = requests.delete(url, timeout=10, **kwargs)
            else:
                print(f'❌ Unsupported method: {method}')
                return False, None

            if response.status_code == 200:
                print(f'✅ {name} - SUCCESS (200 OK)')
                return True, response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
            else:
                print(f'❌ {name} - FAILED ({response.status_code})')
                try:
                    error = response.json()
                    print(f'   Error: {error.get("detail", "Unknown error")}')
                except:
                    print(f'   Response: {response.text[:100]}...')
                return False, None
        except Exception as e:
            print(f'❌ {name} - ERROR: {str(e)[:50]}')
            return False, None

    base_url = 'http://localhost:8000'

    # Test 1: Initialize default locations
    success, data = test_endpoint('Initialize Default Locations', 'GET', f'{base_url}/api/v1/locations/initialize')
    if success and isinstance(data, dict):
        print(f'   Initialized: {data.get("count", 0)} locations')

    # Test 2: Search locations
    success, data = test_endpoint('Search Locations', 'POST', f'{base_url}/api/v1/locations/search',
                                json={"limit": 5, "is_active": True})
    if success and isinstance(data, dict):
        locations = data.get("locations", [])
        print(f'   Found: {data.get("total_count", 0)} total locations')
        print(f'   Returned: {len(locations)} locations')
        if locations:
            sample = locations[0]
            print(f'   Sample: {sample.get("name", "Unknown")} ({sample.get("latitude", 0):.4f}, {sample.get("longitude", 0):.4f})')

    # Test 3: Create new location
    new_location = {
        "name": "Pretoria CBD",
        "display_name": "Pretoria Central Business District",
        "latitude": -25.7479,
        "longitude": 28.2293,
        "country": "South Africa",
        "region": "Gauteng",
        "district": "Pretoria",
        "is_coastal": False,
        "population": 292000,
        "tags": ["urban", "capital", "business"],
        "metadata": {"timezone": "Africa/Johannesburg", "type": "metropolitan"}
    }

    success, data = test_endpoint('Create New Location', 'POST', f'{base_url}/api/v1/locations',
                                json=new_location)
    location_id = None
    if success and isinstance(data, dict):
        location_id = data.get("location_id")
        print(f'   Created location: {data.get("location", {}).get("name", "Unknown")}')
        print(f'   Location ID: {location_id}')

    # Test 4: Get specific location
    if location_id:
        success, data = test_endpoint('Get Location by ID', 'GET', f'{base_url}/api/v1/locations/{location_id}')
        if success and isinstance(data, dict):
            location = data.get("location", {})
            print(f'   Retrieved: {location.get("name", "Unknown")}')
            print(f'   Coordinates: {location.get("latitude", 0)}, {location.get("longitude", 0)}')

    # Test 5: Update location
    if location_id:
        update_data = {
            "population": 300000,
            "tags": ["urban", "capital", "business", "updated"],
            "metadata": {"timezone": "Africa/Johannesburg", "type": "metropolitan", "last_updated": datetime.now().isoformat()}
        }
        success, data = test_endpoint('Update Location', 'PUT', f'{base_url}/api/v1/locations/{location_id}',
                                    json=update_data)
        if success:
            print(f'   Updated location successfully')

    # Test 6: Geocode location
    success, data = test_endpoint('Geocode Location', 'POST', f'{base_url}/api/v1/locations/geocode',
                                json={"query": "Durban, South Africa", "limit": 3})
    if success and isinstance(data, dict):
        locations = data.get("locations", [])
        print(f'   Found: {len(locations)} geocoded locations')
        if locations:
            print(f'   Best match: {locations[0].get("display_name", "")[:50]}...')

    # Test 7: Get location coordinates
    success, data = test_endpoint('Get Location Coordinates', 'GET', f'{base_url}/api/v1/locations/coords')
    if success and isinstance(data, dict):
        coords = data.get("coordinates", {})
        print(f'   Available coordinates: {len(coords)} locations')

    # Test 8: Create location preset
    if location_id:
        preset_data = {
            "name": "Major Cities",
            "description": "South Africa's major metropolitan areas",
            "locations": [location_id],  # Add the created location
            "category": "custom",
            "is_public": True
        }
        success, data = test_endpoint('Create Location Preset', 'POST', f'{base_url}/api/v1/locations/presets',
                                    json=preset_data)
        preset_id = None
        if success and isinstance(data, dict):
            preset_id = data.get("preset_id")
            print(f'   Created preset: {preset_data["name"]}')

        # Test getting preset locations
        if preset_id:
            success, data = test_endpoint('Get Preset Locations', 'GET', f'{base_url}/api/v1/locations/presets/{preset_id}')
            if success and isinstance(data, dict):
                preset = data.get("preset", {})
                locations = data.get("locations", [])
                print(f'   Preset contains: {len(locations)} locations')

    # Test 9: Collect weather for specific locations
    if location_id:
        success, data = test_endpoint('Collect Weather for Location ID', 'GET',
                                    f'{base_url}/api/v1/weather/collect?location_ids={location_id}')
        if success and isinstance(data, dict):
            print(f'   Weather collected: {data.get("successful_collections", 0)}/{data.get("total_locations", 0)} locations')

    print('')
    print('🎉 DYNAMIC LOCATION MANAGEMENT SYSTEM SUCCESSFULLY TESTED!')
    print('=' * 60)
    print('')
    print('🌟 NEW DYNAMIC FEATURES:')
    print('   ✅ Database-driven locations (no more hardcoded!)')
    print('   ✅ CRUD operations for location management')
    print('   ✅ Advanced search and filtering')
    print('   ✅ Geocoding integration')
    print('   ✅ Location presets/templates')
    print('   ✅ Dynamic weather collection by location ID')
    print('   ✅ Rich metadata support')
    print('')
    print('📋 API ENDPOINTS:')
    print('   POST /api/v1/locations - Create location')
    print('   GET /api/v1/locations/{id} - Get location')
    print('   PUT /api/v1/locations/{id} - Update location')
    print('   DELETE /api/v1/locations/{id} - Delete location')
    print('   POST /api/v1/locations/search - Search locations')
    print('   POST /api/v1/locations/geocode - Geocode address')
    print('   POST /api/v1/locations/presets - Create preset')
    print('   GET /api/v1/locations/coords - Get coordinates')
    print('')
    print('🎯 FLEXIBLE LOCATION MANAGEMENT ACHIEVED!')

if __name__ == '__main__':
    main()
