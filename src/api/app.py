"""
FastAPI application setup with all routes
Main API layer for Team-Realized platform
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from typing import List, Optional
import logging

from .schemas import (
    AffordablePropertiesRequest, AffordablePropertiesResponse,
    CityOpportunitiesResponse, PropertyAnalysisResponse,
    OneTapPurchaseRequest, OneTapPurchaseResponse,
    HealthCheckResponse
)
from ..core.usecases import (
    FindAffordablePropertiesUseCase,
    DetectUndervaluationUseCase,
    OneTapPurchaseUseCase,
    ScanCitiesForOpportunitiesUseCase
)
from ..core.entities import UserBudget, InvestmentRecommendation
from ..core.config import settings
from datetime import datetime

logger = logging.getLogger(__name__)

# Create main API router
router = APIRouter()


# Dependency injection - will be implemented with proper DI container
async def get_find_affordable_use_case():
    """Get FindAffordablePropertiesUseCase instance"""
    # TODO: Implement proper dependency injection
    return None  # Placeholder

async def get_detect_undervaluation_use_case():
    """Get DetectUndervaluationUseCase instance"""
    # TODO: Implement proper dependency injection
    return None  # Placeholder

async def get_one_tap_purchase_use_case():
    """Get OneTapPurchaseUseCase instance"""
    # TODO: Implement proper dependency injection
    return None  # Placeholder


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=settings.app_version,
        environment=settings.environment
    )


@router.get("/properties/affordable", response_model=AffordablePropertiesResponse)
async def get_affordable_properties(
    budget: float = Query(500.0, ge=100.0, le=5000.0, description="Monthly budget in EUR"),
    location: Optional[str] = Query(None, description="Geographic preference"),
    max_results: int = Query(10, ge=1, le=50, description="Maximum number of results"),
    risk_tolerance: str = Query("MEDIUM", description="Risk tolerance: LOW, MEDIUM, HIGH")
):
    """
    Find properties affordable with user's monthly budget.
    Uses AI to scan 50+ European cities and return top opportunities.
    """
    start_time = datetime.utcnow()

    try:
        logger.info(f"Finding affordable properties for budget: {budget} EUR")

        # Create user budget
        user_budget = UserBudget(
            user_id=f"api_user_{int(start_time.timestamp())}",
            monthly_budget=budget,
            max_single_investment=min(budget, settings.max_investment_amount),
            preferred_cities=[location] if location else settings.target_cities[:10],
            risk_tolerance=risk_tolerance,
            investment_horizon="MEDIUM",
            created_at=start_time
        )

        # Mock opportunities for hackathon (replace with real use case)
        opportunities = await _generate_mock_opportunities(user_budget, max_results)

        analysis_time = (datetime.utcnow() - start_time).total_seconds()

        return AffordablePropertiesResponse(
            opportunities=opportunities,
            total_count=len(opportunities),
            user_budget=budget,
            cities_scanned=len(settings.target_cities[:25]),
            analysis_time_seconds=analysis_time,
            filters_applied={
                "location": location,
                "risk_tolerance": risk_tolerance,
                "budget_range": f"{budget} EUR/month"
            }
        )

    except Exception as e:
        logger.error(f"Error finding affordable properties: {e}")
        raise HTTPException(
            status_code=500,
            detail="Property analysis temporarily unavailable. Please try again later."
        )


@router.get("/cities/{city}/opportunities", response_model=CityOpportunitiesResponse)
async def get_city_opportunities(
    city: str = Path(..., description="City name (e.g., 'Krakow', 'Berlin')"),
    budget: float = Query(500.0, ge=100.0, le=5000.0, description="Investment budget in EUR")
):
    """Get investment opportunities for a specific city"""

    try:
        logger.info(f"Getting opportunities for city: {city}")

        # Create user budget
        user_budget = UserBudget(
            user_id=f"city_user_{int(datetime.utcnow().timestamp())}",
            monthly_budget=budget,
            max_single_investment=min(budget, settings.max_investment_amount),
            preferred_cities=[city],
            risk_tolerance="MEDIUM",
            investment_horizon="MEDIUM",
            created_at=datetime.utcnow()
        )

        # Mock city opportunities
        opportunities = await _generate_mock_opportunities(user_budget, 5, city)

        # Mock market summary
        market_summary = {
            "city": city,
            "average_price_per_sqm": 2500.0 + hash(city) % 1000,
            "price_trend_6m": 2.0 + (hash(city) % 60) / 10,
            "inventory_count": 100 + hash(city) % 200,
            "market_sentiment": "Positive" if hash(city) % 2 else "Neutral",
            "investment_attractiveness": 6.0 + (hash(city) % 40) / 10,
            "rental_yield": 4.0 + (hash(city) % 30) / 10
        }

        return CityOpportunitiesResponse(
            city=city,
            opportunities=opportunities,
            market_summary=market_summary,
            analysis_timestamp=datetime.utcnow()
        )

    except Exception as e:
        logger.error(f"Error getting city opportunities: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis for {city} temporarily unavailable"
        )


@router.post("/analyze/property/{property_id}", response_model=PropertyAnalysisResponse)
async def analyze_property_detailed(
    property_id: str = Path(..., description="Unique property identifier")
):
    """Get detailed AI analysis for a specific property"""

    try:
        logger.info(f"Analyzing property: {property_id}")

        # Mock detailed analysis
        property_hash = hash(property_id)
        undervaluation = 5.0 + (property_hash % 200) / 10  # 5-25% undervaluation
        confidence = 65 + property_hash % 30  # 65-95% confidence

        risk_factors = [
            "Market volatility",
            "Liquidity risk", 
            "Economic uncertainty"
        ]

        if property_hash % 3 == 0:
            risk_factors.append("High property taxes")
        if property_hash % 5 == 0:
            risk_factors.append("Location gentrification risk")

        upside_factors = [
            "Undervalued asset",
            "Growing neighborhood",
            "Good transport links"
        ]

        if property_hash % 4 == 0:
            upside_factors.append("Upcoming infrastructure development")

        recommendation = InvestmentRecommendation.BUY if undervaluation > 15 else (
            InvestmentRecommendation.HOLD if undervaluation > 8 else InvestmentRecommendation.REVIEW
        )

        return PropertyAnalysisResponse(
            property_id=property_id,
            detailed_analysis=f"Comprehensive analysis shows this property is {undervaluation:.1f}% undervalued based on AI pricing models, market comparables, and economic indicators. The asset demonstrates strong fundamentals with moderate risk exposure.",
            undervaluation_percentage=undervaluation,
            confidence_score=confidence,
            investment_recommendation=recommendation,
            risk_assessment=risk_factors,
            upside_factors=upside_factors,
            estimated_holding_period="2-3 years",
            expected_roi_range="8-15% annually"
        )

    except Exception as e:
        logger.error(f"Error analyzing property {property_id}: {e}")
        raise HTTPException(status_code=500, detail="Property analysis failed")


@router.post("/purchase/one-tap", response_model=OneTapPurchaseResponse)
async def one_tap_purchase(request: OneTapPurchaseRequest):
    """
    Execute one-tap purchase of fractional property ownership.
    Issues instant Solana SPL token certificate.
    """

    try:
        logger.info(f"Processing one-tap purchase: {request.property_id} for {request.investment_amount} EUR")

        # Validation
        if request.investment_amount < settings.min_investment_amount:
            raise HTTPException(
                status_code=400,
                detail=f"Investment amount must be at least {settings.min_investment_amount} EUR"
            )

        if request.investment_amount > settings.max_investment_amount:
            raise HTTPException(
                status_code=400,
                detail=f"Investment amount cannot exceed {settings.max_investment_amount} EUR"
            )

        # Mock purchase processing
        import uuid
        transaction_id = str(uuid.uuid4())
        certificate_token = f"PROP-{request.property_id[:8].upper()}-{transaction_id[:8].upper()}"

        # Mock property value and ownership calculation
        property_value = 180000 + (hash(request.property_id) % 100000)  # 180k-280k EUR
        ownership_percentage = (request.investment_amount / property_value) * 100

        # Calculate fees
        fees = request.investment_amount * (settings.transaction_fee_percentage / 100)
        total_cost = request.investment_amount + fees

        logger.info(f"Purchase successful: {transaction_id}")

        return OneTapPurchaseResponse(
            success=True,
            transaction_id=transaction_id,
            certificate_token=certificate_token,
            ownership_percentage=ownership_percentage,
            property_value=property_value,
            total_cost=total_cost,
            fees_paid=fees,
            blockchain_network="SOLANA",
            estimated_annual_return=request.investment_amount * 0.08,  # 8% estimated return
            message=f"Successfully purchased {request.investment_amount} EUR ownership! You now own {ownership_percentage:.4f}% of this property."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Purchase failed: {e}")
        return OneTapPurchaseResponse(
            success=False,
            message=f"Purchase failed: {str(e)}. Please try again later."
        )


@router.get("/api/stats")
async def get_api_stats():
    """Get API statistics and information"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "features": {
            "cities_supported": len(settings.target_cities),
            "min_investment": f"{settings.min_investment_amount} EUR",
            "max_investment": f"{settings.max_investment_amount} EUR",
            "transaction_fee": f"{settings.transaction_fee_percentage}%",
            "ai_analysis": settings.ai_models_enabled
        },
        "endpoints": {
            "properties": "/properties/affordable",
            "city_analysis": "/cities/{city}/opportunities",
            "property_details": "/analyze/property/{property_id}",
            "purchase": "/purchase/one-tap",
            "health": "/health"
        }
    }


# Mock data helper
async def _generate_mock_opportunities(user_budget: UserBudget, count: int = 10, specific_city: str = None):
    """Generate mock opportunities for demonstration"""
    import random
    from ..api.schemas import PropertyOpportunitySchema

    cities = [specific_city] if specific_city else random.sample(settings.target_cities, min(5, len(settings.target_cities)))

    opportunities = []
    for i in range(count):
        city = random.choice(cities)
        property_id = f"{city.lower()}_prop_{i+1}"

        # Mock property data with some variance
        base_value = 150000 + random.randint(-50000, 150000)
        predicted_value = base_value * random.uniform(0.85, 1.25)
        undervaluation = max(0, ((predicted_value - base_value) / base_value) * 100)

        # Calculate investment range
        min_inv, max_inv = user_budget.get_affordable_range()

        opportunity = PropertyOpportunitySchema(
            property_id=property_id,
            city=city,
            address=f"Sample Street {i+1}, {city}",
            total_property_value=base_value,
            predicted_fair_value=predicted_value,
            undervaluation_percentage=undervaluation,
            min_investment_amount=min_inv,
            max_investment_amount=max_inv,
            is_undervalued=undervaluation > 5,
            is_affordable=True,
            attractiveness_score=50 + undervaluation * 2 + random.uniform(-10, 20),
            confidence_score=70 + random.uniform(-15, 25),
            recommendation=InvestmentRecommendation.BUY if undervaluation > 15 else InvestmentRecommendation.HOLD,
            analysis_summary=f"AI analysis indicates {city} property shows {undervaluation:.1f}% undervaluation opportunity with moderate risk profile.",
            risk_factors=["Market volatility", "Liquidity risk"],
            upside_factors=["Undervalued market", "Growth potential"],
            property_type="APARTMENT",
            size_sqm=50 + random.randint(10, 100),
            rooms=2 + random.randint(0, 3),
            year_built=1980 + random.randint(0, 40)
        )
        opportunities.append(opportunity)

    # Sort by attractiveness score
    opportunities.sort(key=lambda x: x.attractiveness_score, reverse=True)

    return opportunities


# Create app factory
def create_app():
    """Create FastAPI application with all routes"""
    from fastapi import FastAPI

    app = FastAPI(
        title=settings.app_name,
        description="AI-Powered Real Estate Tokenization Platform",
        version=settings.app_version
    )

    app.include_router(router, prefix="/api/v1")

    return app
