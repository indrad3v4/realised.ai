"""
Tests for Team-Realized API endpoints
Essential integration tests for hackathon demo
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json

from ..main import app  # Import from main.py
from ..core.entities import InvestmentRecommendation


@pytest.fixture
def client():
    """Test client for FastAPI app"""
    return TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_check(self, client):
        """Test health check returns 200"""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data


class TestAffordablePropertiesEndpoint:
    """Test affordable properties endpoint"""

    def test_get_affordable_properties_default(self, client):
        """Test getting affordable properties with default parameters"""
        response = client.get("/properties/affordable")

        assert response.status_code == 200
        data = response.json()

        assert "opportunities" in data
        assert "total_count" in data
        assert "user_budget" in data
        assert "cities_scanned" in data
        assert "analysis_time_seconds" in data

        assert data["user_budget"] == 500.0  # Default budget
        assert isinstance(data["opportunities"], list)
        assert data["total_count"] >= 0

    def test_get_affordable_properties_custom_budget(self, client):
        """Test getting properties with custom budget"""
        response = client.get("/properties/affordable?budget=750&max_results=5")

        assert response.status_code == 200
        data = response.json()

        assert data["user_budget"] == 750.0
        assert len(data["opportunities"]) <= 5

    def test_get_affordable_properties_with_location(self, client):
        """Test getting properties with location filter"""
        response = client.get("/properties/affordable?budget=500&location=Krakow")

        assert response.status_code == 200
        data = response.json()

        # All opportunities should be in Krakow (if any)
        for opp in data["opportunities"]:
            assert opp["city"] == "Krakow"

    def test_invalid_budget_too_low(self, client):
        """Test invalid budget (too low)"""
        response = client.get("/properties/affordable?budget=50")

        assert response.status_code == 422  # Validation error

    def test_invalid_budget_too_high(self, client):
        """Test invalid budget (too high)"""
        response = client.get("/properties/affordable?budget=6000")

        assert response.status_code == 422  # Validation error


class TestCityOpportunitiesEndpoint:
    """Test city opportunities endpoint"""

    def test_get_city_opportunities(self, client):
        """Test getting opportunities for a specific city"""
        response = client.get("/cities/Krakow/opportunities")

        assert response.status_code == 200
        data = response.json()

        assert data["city"] == "Krakow"
        assert "opportunities" in data
        assert "market_summary" in data
        assert "analysis_timestamp" in data

        # All opportunities should be in Krakow
        for opp in data["opportunities"]:
            assert opp["city"] == "Krakow"

    def test_get_city_opportunities_with_budget(self, client):
        """Test getting city opportunities with custom budget"""
        response = client.get("/cities/Berlin/opportunities?budget=800")

        assert response.status_code == 200
        data = response.json()

        assert data["city"] == "Berlin"
        # Check that opportunities respect budget constraints
        for opp in data["opportunities"]:
            assert opp["min_investment_amount"] <= 800


class TestPropertyAnalysisEndpoint:
    """Test property analysis endpoint"""

    def test_analyze_property(self, client):
        """Test detailed property analysis"""
        property_id = "test_property_123"
        response = client.post(f"/analyze/property/{property_id}")

        assert response.status_code == 200
        data = response.json()

        assert data["property_id"] == property_id
        assert "detailed_analysis" in data
        assert "undervaluation_percentage" in data
        assert "confidence_score" in data
        assert "investment_recommendation" in data
        assert "risk_assessment" in data
        assert "upside_factors" in data

        # Validate recommendation is valid enum value
        assert data["investment_recommendation"] in [
            "BUY", "HOLD", "SELL", "PASS", "REVIEW"
        ]

        # Validate score ranges
        assert 0 <= data["confidence_score"] <= 100
        assert data["undervaluation_percentage"] >= 0


class TestOneTapPurchaseEndpoint:
    """Test one-tap purchase endpoint"""

    def test_successful_purchase(self, client):
        """Test successful purchase"""
        purchase_request = {
            "property_id": "krakow_prop_1",
            "investment_amount": 500.0,
            "user_id": "test_user_123"
        }

        response = client.post(
            "/purchase/one-tap",
            json=purchase_request
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "transaction_id" in data
        assert "certificate_token" in data
        assert "ownership_percentage" in data
        assert data["ownership_percentage"] > 0
        assert "message" in data

    def test_purchase_invalid_amount_too_low(self, client):
        """Test purchase with amount too low"""
        purchase_request = {
            "property_id": "test_prop",
            "investment_amount": 50.0,  # Below minimum
            "user_id": "test_user"
        }

        response = client.post("/purchase/one-tap", json=purchase_request)

        assert response.status_code == 422  # Validation error

    def test_purchase_invalid_amount_too_high(self, client):
        """Test purchase with amount too high"""
        purchase_request = {
            "property_id": "test_prop",
            "investment_amount": 1500.0,  # Above maximum
            "user_id": "test_user"
        }

        response = client.post("/purchase/one-tap", json=purchase_request)

        assert response.status_code == 422  # Validation error

    def test_purchase_missing_fields(self, client):
        """Test purchase with missing required fields"""
        purchase_request = {
            "property_id": "test_prop",
            # Missing investment_amount and user_id
        }

        response = client.post("/purchase/one-tap", json=purchase_request)

        assert response.status_code == 422  # Validation error


class TestAPIInfoEndpoint:
    """Test API information endpoint"""

    def test_api_stats(self, client):
        """Test API statistics endpoint"""
        response = client.get("/api/stats")

        assert response.status_code == 200
        data = response.json()

        assert "name" in data
        assert "version" in data
        assert "environment" in data
        assert "features" in data
        assert "endpoints" in data

        # Validate features
        features = data["features"]
        assert "cities_supported" in features
        assert "min_investment" in features
        assert "max_investment" in features
        assert "ai_analysis" in features


class TestErrorHandling:
    """Test error handling"""

    def test_404_endpoint(self, client):
        """Test 404 for non-existent endpoint"""
        response = client.get("/non-existent-endpoint")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    def test_invalid_json(self, client):
        """Test invalid JSON in request body"""
        response = client.post(
            "/purchase/one-tap",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422  # Validation error


class TestDataValidation:
    """Test data validation"""

    def test_property_opportunity_structure(self, client):
        """Test that property opportunities have correct structure"""
        response = client.get("/properties/affordable?budget=500&max_results=1")

        assert response.status_code == 200
        data = response.json()

        if data["opportunities"]:
            opp = data["opportunities"][0]

            # Required fields
            required_fields = [
                "property_id", "city", "address", "total_property_value",
                "predicted_fair_value", "undervaluation_percentage",
                "min_investment_amount", "max_investment_amount",
                "is_undervalued", "is_affordable", "attractiveness_score",
                "confidence_score", "recommendation", "analysis_summary",
                "risk_factors", "upside_factors"
            ]

            for field in required_fields:
                assert field in opp, f"Missing required field: {field}"

            # Validate data types and ranges
            assert isinstance(opp["property_id"], str)
            assert isinstance(opp["city"], str)
            assert isinstance(opp["total_property_value"], (int, float))
            assert opp["total_property_value"] > 0
            assert isinstance(opp["is_undervalued"], bool)
            assert isinstance(opp["is_affordable"], bool)
            assert 0 <= opp["attractiveness_score"] <= 100
            assert 0 <= opp["confidence_score"] <= 100
            assert isinstance(opp["risk_factors"], list)
            assert isinstance(opp["upside_factors"], list)


# Integration test fixtures
@pytest.fixture
def mock_ai_service():
    """Mock AI service for testing"""
    mock = Mock()
    mock.find_affordable_opportunities = Mock(return_value=[])
    mock.get_city_market_summary = Mock(return_value={})
    return mock


@pytest.fixture  
def sample_opportunity_data():
    """Sample opportunity data for testing"""
    return {
        "property_id": "test_prop_1",
        "city": "Krakow",
        "address": "Test Street 1, Krakow", 
        "total_property_value": 180000.0,
        "predicted_fair_value": 198000.0,
        "undervaluation_percentage": 10.0,
        "min_investment_amount": 100.0,
        "max_investment_amount": 500.0,
        "is_undervalued": True,
        "is_affordable": True,
        "attractiveness_score": 85.0,
        "confidence_score": 75.0,
        "recommendation": InvestmentRecommendation.BUY,
        "analysis_summary": "Test property analysis",
        "risk_factors": ["Market volatility"],
        "upside_factors": ["Undervalued market"],
        "property_type": "APARTMENT",
        "size_sqm": 70.0,
        "rooms": 3,
        "year_built": 2010
    }
