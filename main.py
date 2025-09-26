#!/usr/bin/env python3
"""
main.py - Team-Realized AI Real Estate Tokenization Platform

Entry point for the FastAPI application that enables young Europeans
to buy €100-1000 micro-investments in real estate properties.

Key Features:
- 50+ European cities real estate scanning
- AI-powered undervaluation detection (Fast.ai + PyTorch + DeepSeek)  
- One-tap fractional ownership purchase
- Instant Solana SPL token certificates
- Progressive Web App for mobile-first experience
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any

# FastAPI and related imports
from fastapi import FastAPI, HTTPException, Request, Query, Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

# Pydantic models
from pydantic import BaseModel, Field
from typing import List, Optional

# Standard library
import uvicorn
from datetime import datetime
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Import our AI models (with fallback handling)
try:
    from src.adapters.ai_models import AIModelsService
    AI_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: AI models not available - {e}")
    AI_MODELS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("team_realized.log")
    ]
)
logger = logging.getLogger(__name__)

# Pydantic models for API
class PropertyOpportunity(BaseModel):
    """Property investment opportunity model"""
    property_id: str
    city: str
    address: str
    total_property_value: float
    predicted_fair_value: float
    undervaluation_percentage: float
    min_investment_amount: float = Field(default=100.0, description="Minimum investment in EUR")
    max_investment_amount: float
    is_undervalued: bool
    is_affordable: bool
    attractiveness_score: float
    analysis_summary: str
    risk_factors: List[str]
    recommendation: str

class AffordablePropertiesResponse(BaseModel):
    """Response model for affordable properties endpoint"""
    opportunities: List[PropertyOpportunity]
    total_count: int
    user_budget: float
    cities_scanned: int
    analysis_time_seconds: float

class OneTabPurchaseRequest(BaseModel):
    """Request model for one-tap purchase"""
    property_id: str
    investment_amount: float = Field(..., ge=100.0, le=1000.0, description="Investment amount in EUR (100-1000)")
    user_id: str

class OneTabPurchaseResponse(BaseModel):
    """Response model for one-tap purchase"""
    success: bool
    transaction_id: Optional[str] = None
    certificate_token: Optional[str] = None
    ownership_percentage: Optional[float] = None
    message: str

class CityOpportunitiesResponse(BaseModel):
    """Response model for city-specific opportunities"""
    city: str
    opportunities: List[PropertyOpportunity]
    market_summary: Dict[str, Any]
    analysis_timestamp: datetime

# Global AI service instance
ai_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting Team-Realized platform...")

    global ai_service
    if AI_MODELS_AVAILABLE:
        try:
            ai_service = AIModelsService()
            logger.info("AI Models Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AI service: {e}")
            ai_service = None
    else:
        logger.warning("AI Models Service not available - using mock data")

    yield

    # Shutdown
    logger.info("Shutting down Team-Realized platform...")

# Initialize FastAPI app
app = FastAPI(
    title="Team-Realized API",
    description="AI-Powered Real Estate Tokenization Platform - Enabling 100-1000 EUR micro-investments for young Europeans",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Mount static files for PWA
static_path = Path(__file__).parent / "src" / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Templates for PWA
templates_path = Path(__file__).parent / "src" / "templates"
if templates_path.exists():
    templates = Jinja2Templates(directory=str(templates_path))

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "ai_models_available": AI_MODELS_AVAILABLE,
        "services": {
            "ai_service": ai_service is not None,
            "api": True,
            "database": True  # TODO: Add real DB health check
        }
    }

# Root endpoint - PWA interface
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the Progressive Web App"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="theme-color" content="#2196F3">
    <title>Team-Realized - Own Property for 100 EUR</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            line-height: 1.6; color: #333; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            min-height: 100vh; 
        }
        .container { max-width: 400px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; color: white; margin-bottom: 30px; }
        .header h1 { font-size: 2em; margin-bottom: 10px; }
        .header p { opacity: 0.9; }
        .card { 
            background: white; border-radius: 15px; padding: 25px; margin-bottom: 20px; 
            box-shadow: 0 4px 20px rgba(0,0,0,0.1); 
        }
        .btn { 
            background: #4CAF50; color: white; border: none; padding: 15px 25px; 
            border-radius: 8px; font-size: 1.1em; font-weight: bold; width: 100%; 
            margin: 10px 0; cursor: pointer; transition: all 0.3s ease; 
        }
        .btn:hover { background: #45a049; transform: translateY(-2px); }
        .btn-secondary { background: #2196F3; }
        .btn-secondary:hover { background: #1976D2; }
        .property { border-left: 4px solid #4CAF50; padding-left: 15px; margin: 15px 0; }
        .property h3 { color: #2196F3; margin-bottom: 5px; }
        .property .details { font-size: 0.9em; color: #666; }
        .property .price { color: #4CAF50; font-weight: bold; font-size: 1.1em; margin-top: 5px; }
        .loading { display: none; text-align: center; }
        .spinner { 
            border: 3px solid #f3f3f3; border-top: 3px solid #3498db; border-radius: 50%; 
            width: 30px; height: 30px; animation: spin 1s linear infinite; margin: 20px auto; 
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .success { background: #d4edda; color: #155724; padding: 15px; border-radius: 8px; margin: 10px 0; }
        .error { background: #f8d7da; color: #721c24; padding: 15px; border-radius: 8px; margin: 10px 0; }
        input { 
            width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; 
            border-radius: 5px; font-size: 1em; 
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏠 Team-Realized</h1>
            <p>Own property for 100 EUR. Escape rent slavery.</p>
        </div>

        <div class="card">
            <h2>Find Your Property Investment</h2>
            <label for="budget">Monthly Budget (EUR):</label>
            <input type="number" id="budget" value="500" min="100" max="2000">

            <button class="btn" onclick="findOpportunities()">
                🔍 Find Properties I Can Afford
            </button>
        </div>

        <div id="loading" class="loading">
            <div class="spinner"></div>
            <p>AI scanning 50+ European cities...</p>
        </div>

        <div id="results"></div>

        <div class="card" style="text-align: center; color: #666;">
            <p><strong>How it works:</strong><br>
            1. AI finds undervalued properties<br>
            2. Buy 100-1000 EUR pieces<br>
            3. Get instant ownership certificate<br>
            4. Build real estate portfolio</p>
        </div>
    </div>

    <script>
        async function findOpportunities() {
            const budget = document.getElementById('budget').value;
            const loading = document.getElementById('loading');
            const results = document.getElementById('results');

            loading.style.display = 'block';
            results.innerHTML = '';

            try {
                const response = await fetch(`/properties/affordable?budget=${budget}`);
                const data = await response.json();

                loading.style.display = 'none';

                if (data.opportunities && data.opportunities.length > 0) {
                    let html = `<div class="card">
                        <h3>🎯 Found ${data.total_count} Investment Opportunities</h3>
                        <p>Scanned ${data.cities_scanned} cities in ${data.analysis_time_seconds.toFixed(1)}s</p>
                    </div>`;

                    data.opportunities.slice(0, 5).forEach(opp => {
                        html += `<div class="card">
                            <div class="property">
                                <h3>${opp.city} Property</h3>
                                <div class="details">${opp.address}</div>
                                <div class="details">
                                    ${opp.is_undervalued ? '📈 ' + opp.undervaluation_percentage.toFixed(1) + '% undervalued' : '📊 Fair priced'}
                                </div>
                                <div class="price">
                                    Invest: ${opp.min_investment_amount} EUR - ${opp.max_investment_amount} EUR
                                </div>
                                <button class="btn btn-secondary" onclick="buyProperty('${opp.property_id}', ${opp.min_investment_amount})">
                                    💰 Buy ${opp.min_investment_amount} EUR Piece
                                </button>
                            </div>
                        </div>`;
                    });

                    results.innerHTML = html;
                } else {
                    results.innerHTML = `<div class="error">
                        No opportunities found for ${budget} EUR budget. Try increasing your budget or check back later.
                    </div>`;
                }
            } catch (error) {
                loading.style.display = 'none';
                results.innerHTML = `<div class="error">
                    Service temporarily unavailable. Please try again later.
                </div>`;
                console.error('Error:', error);
            }
        }

        async function buyProperty(propertyId, amount) {
            try {
                const response = await fetch('/purchase/one-tap', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        property_id: propertyId,
                        investment_amount: amount,
                        user_id: 'demo_user_' + Date.now()
                    })
                });

                const result = await response.json();

                if (result.success) {
                    alert(`🎉 Success! You own ${result.ownership_percentage.toFixed(3)}% of this property. Certificate: ${result.certificate_token}`);
                } else {
                    alert('Purchase failed: ' + result.message);
                }
            } catch (error) {
                alert('Purchase failed. Please try again.');
                console.error('Purchase error:', error);
            }
        }

        // Auto-load opportunities on page load
        document.addEventListener('DOMContentLoaded', function() {
            // Uncomment to auto-load on startup
            // findOpportunities();
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

# Core API Endpoints

@app.get("/properties/affordable", response_model=AffordablePropertiesResponse)
async def get_affordable_properties(
    budget: float = Query(500.0, ge=100.0, le=5000.0, description="Monthly budget in EUR"),
    location: str = Query("Europe", description="Geographic preference"),
    max_results: int = Query(10, ge=1, le=50, description="Maximum number of results")
):
    """
    Get properties affordable with user's monthly budget.
    Uses AI to scan 50+ European cities and return top opportunities.
    """
    start_time = datetime.utcnow()

    try:
        logger.info(f"Finding affordable properties for budget: {budget} EUR")

        if ai_service:
            # Use real AI service
            opportunities = await ai_service.find_affordable_opportunities(
                user_budget=budget,
                user_location=location
            )
            cities_scanned = len(ai_service.city_analyzer.target_cities)
        else:
            # Mock data fallback
            logger.warning("Using mock data - AI service not available")
            opportunities = _generate_mock_opportunities(budget, max_results)
            cities_scanned = 25  # Mock value

        # Convert to response format
        property_opportunities = [
            PropertyOpportunity(
                property_id=opp.property_id,
                city=opp.city,
                address=opp.address,
                total_property_value=opp.total_property_value,
                predicted_fair_value=opp.predicted_fair_value,
                undervaluation_percentage=opp.undervaluation_percentage,
                min_investment_amount=opp.min_investment_amount,
                max_investment_amount=opp.max_investment_amount,
                is_undervalued=opp.is_undervalued,
                is_affordable=opp.is_affordable,
                attractiveness_score=opp.attractiveness_score,
                analysis_summary=opp.analysis_summary,
                risk_factors=opp.risk_factors,
                recommendation=opp.recommendation
            ) for opp in opportunities[:max_results]
        ]

        analysis_time = (datetime.utcnow() - start_time).total_seconds()

        return AffordablePropertiesResponse(
            opportunities=property_opportunities,
            total_count=len(property_opportunities),
            user_budget=budget,
            cities_scanned=cities_scanned,
            analysis_time_seconds=analysis_time
        )

    except Exception as e:
        logger.error(f"Error finding affordable properties: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Property analysis temporarily unavailable. Please try again later."
        )

@app.get("/cities/{city}/opportunities", response_model=CityOpportunitiesResponse)
async def get_city_opportunities(
    city: str = PathParam(..., description="City name (e.g., 'Krakow', 'Berlin')"),
    budget: float = Query(500.0, ge=100.0, le=5000.0, description="Investment budget in EUR")
):
    """Get investment opportunities for a specific city"""

    try:
        logger.info(f"Getting opportunities for city: {city}")

        if ai_service:
            market_summary = await ai_service.get_city_market_summary(city)
            # For now, filter general opportunities by city
            all_opportunities = await ai_service.find_affordable_opportunities(user_budget=budget)
            city_opportunities = [opp for opp in all_opportunities if opp.city.lower() == city.lower()]
        else:
            # Mock data
            market_summary = {
                "city": city,
                "average_price_per_sqm": 2500.0,
                "price_trend_6m": 5.2,
                "inventory_count": 150,
                "market_sentiment": "Positive",
                "investment_attractiveness": 7.5
            }
            city_opportunities = _generate_mock_opportunities(budget, 5, city)

        property_opportunities = [
            PropertyOpportunity(
                property_id=opp.property_id,
                city=opp.city,
                address=opp.address,
                total_property_value=opp.total_property_value,
                predicted_fair_value=opp.predicted_fair_value,
                undervaluation_percentage=opp.undervaluation_percentage,
                min_investment_amount=opp.min_investment_amount,
                max_investment_amount=opp.max_investment_amount,
                is_undervalued=opp.is_undervalued,
                is_affordable=opp.is_affordable,
                attractiveness_score=opp.attractiveness_score,
                analysis_summary=opp.analysis_summary,
                risk_factors=opp.risk_factors,
                recommendation=opp.recommendation
            ) for opp in city_opportunities
        ]

        return CityOpportunitiesResponse(
            city=city,
            opportunities=property_opportunities,
            market_summary=market_summary,
            analysis_timestamp=datetime.utcnow()
        )

    except Exception as e:
        logger.error(f"Error getting city opportunities: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis for {city} temporarily unavailable")

@app.post("/analyze/property/{property_id}")
async def analyze_property_detailed(
    property_id: str = PathParam(..., description="Unique property identifier")
):
    """Get detailed AI analysis for a specific property"""

    try:
        logger.info(f"Analyzing property: {property_id}")

        if ai_service:
            # Mock property data for analysis
            property_data = {
                'id': property_id,
                'city': 'Krakow',
                'address': 'Sample Street 1, Krakow',
                'price': 180000,
                'size_sqm': 70,
                'property_age': 12,
                'rooms': 3
            }

            opportunity = await ai_service.analyze_specific_property(property_id, property_data)

            if opportunity:
                return {
                    "property_id": property_id,
                    "detailed_analysis": opportunity.analysis_summary,
                    "undervaluation_percentage": opportunity.undervaluation_percentage,
                    "investment_recommendation": opportunity.recommendation,
                    "risk_assessment": opportunity.risk_factors,
                    "confidence_score": opportunity.attractiveness_score
                }
            else:
                raise HTTPException(status_code=404, detail="Property not found")
        else:
            # Mock analysis
            return {
                "property_id": property_id,
                "detailed_analysis": "Mock analysis: This property shows potential for growth based on location and market trends.",
                "undervaluation_percentage": 12.5,
                "investment_recommendation": "BUY",
                "risk_assessment": ["Market volatility", "Liquidity risk"],
                "confidence_score": 75.0
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing property {property_id}: {e}")
        raise HTTPException(status_code=500, detail="Property analysis failed")

@app.post("/purchase/one-tap", response_model=OneTabPurchaseResponse)
async def one_tap_purchase(request: OneTabPurchaseRequest):
    """
    Execute one-tap purchase of fractional property ownership.
    Issues instant Solana SPL token certificate.
    """

    try:
        logger.info(f"Processing one-tap purchase: {request.property_id} for {request.investment_amount} EUR")

        # Basic validation
        if request.investment_amount < 100 or request.investment_amount > 1000:
            return OneTabPurchaseResponse(
                success=False,
                message="Investment amount must be between 100 EUR and 1000 EUR"
            )

        # Mock purchase processing (would integrate with real payment & blockchain)
        import uuid
        transaction_id = str(uuid.uuid4())
        certificate_token = f"PROP-{request.property_id[:8].upper()}-{transaction_id[:8].upper()}"

        # Calculate ownership percentage (assuming 200k EUR property)
        property_value = 200000  # Mock total property value
        ownership_percentage = (request.investment_amount / property_value) * 100

        logger.info(f"Purchase successful: {transaction_id}")

        return OneTabPurchaseResponse(
            success=True,
            transaction_id=transaction_id,
            certificate_token=certificate_token,
            ownership_percentage=ownership_percentage,
            message=f"Successfully purchased {request.investment_amount} EUR ownership! Certificate: {certificate_token}"
        )

    except Exception as e:
        logger.error(f"Purchase failed: {e}")
        return OneTabPurchaseResponse(
            success=False,
            message="Purchase failed. Please try again later."
        )

# Mock data generator for fallback
def _generate_mock_opportunities(budget: float, count: int = 10, specific_city: str = None):
    """Generate mock opportunities when AI service is unavailable"""
    import random
    from dataclasses import dataclass

    @dataclass
    class MockOpportunity:
        property_id: str
        city: str  
        address: str
        total_property_value: float
        predicted_fair_value: float
        undervaluation_percentage: float
        min_investment_amount: float
        max_investment_amount: float
        is_undervalued: bool
        is_affordable: bool
        attractiveness_score: float
        analysis_summary: str
        risk_factors: list
        recommendation: str

    cities = [specific_city] if specific_city else ['Krakow', 'Berlin', 'Prague', 'Barcelona', 'Warsaw']

    opportunities = []
    for i in range(count):
        city = random.choice(cities)
        property_value = random.randint(120000, 300000)
        predicted_value = property_value * random.uniform(0.9, 1.2)
        undervaluation = ((predicted_value - property_value) / property_value) * 100

        opportunity = MockOpportunity(
            property_id=f"{city.lower()}_prop_{i+1}",
            city=city,
            address=f"Sample Street {i+1}, {city}",
            total_property_value=property_value,
            predicted_fair_value=predicted_value,
            undervaluation_percentage=max(0, undervaluation),
            min_investment_amount=100.0,
            max_investment_amount=min(budget, 1000.0),
            is_undervalued=undervaluation > 5,
            is_affordable=True,
            attractiveness_score=random.uniform(60, 95),
            analysis_summary=f"Mock analysis: {city} property shows {undervaluation:.1f}% value opportunity",
            risk_factors=["Market volatility", "Liquidity risk"],
            recommendation="BUY" if undervaluation > 10 else "HOLD"
        )
        opportunities.append(opportunity)

    return opportunities

# API documentation endpoints
@app.get("/api/info")
async def api_info():
    """Get API information and available endpoints"""
    return {
        "name": "Team-Realized API",
        "version": "1.0.0",
        "description": "AI-Powered Real Estate Tokenization Platform",
        "endpoints": {
            "GET /": "Progressive Web App interface",
            "GET /health": "Health check",
            "GET /properties/affordable": "Find affordable properties",
            "GET /cities/{city}/opportunities": "City-specific opportunities",
            "POST /analyze/property/{id}": "Detailed property analysis", 
            "POST /purchase/one-tap": "One-tap purchase",
            "GET /docs": "Interactive API documentation",
            "GET /redoc": "Alternative API documentation"
        },
        "features": [
            "50+ European cities scanning",
            "AI undervaluation detection",
            "100-1000 EUR micro-investments", 
            "Instant blockchain certificates",
            "Mobile-first PWA experience"
        ]
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found. Check /api/info for available endpoints."}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."}
    )

# Main execution
if __name__ == "__main__":
    # Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    logger.info(f"Starting Team-Realized server on {HOST}:{PORT}")
    logger.info(f"PWA available at: http://{HOST}:{PORT}")
    logger.info(f"API docs at: http://{HOST}:{PORT}/docs")

    # Run server
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        access_log=True,
        log_level="info" if not DEBUG else "debug"
    )
