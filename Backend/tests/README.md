# Crisis Connect Backend Tests

This directory contains tests for the Crisis Connect backend application.

## Test Structure

- `conftest.py` - Contains test fixtures and utilities used across test files
- `test_models.py` - Tests for Pydantic models
- `test_predict.py` - Tests for prediction service
- `test_alert_generate.py` - Tests for alert generation service
- `test_db.py` - Tests for database utilities
- `test_main.py` - Tests for API endpoints

## Running Tests

### Install Test Dependencies

```bash
pip install -r tests/requirements-test.txt
```

### Run All Tests

```bash
pytest tests/
```

### Run Tests with Coverage

```bash
pytest --cov=. tests/
```

### Run Specific Test File

```bash
pytest tests/test_models.py
```

### Run Specific Test Function

```bash
pytest tests/test_models.py::TestAlertModel::test_valid_alert_model
```

## Test Configuration

The tests use pytest fixtures to mock database connections and external API calls. This ensures tests can run without actual external dependencies.