"""
Pydantic schemas for Team-Realized API
Request and response models
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

from ..core.entities import InvestmentRecommendation, PropertyType


class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: datetime
    version: str
    environment: str
    services: Optional[Dict[str, bool]] = None


class PropertyOpportunitySchema(BaseModel):
    """Property investment opportunity schema"""
    property_id: str
    city: str
    address: str
    total_property_value: float = Field(..., gt=0, description="Total property value in EUR")
    predicted_fair_value: float = Field(..., gt=0, description="AI predicted fair value in EUR")
    undervaluation_percentage: float = Field(..., ge=0, description="Undervaluation percentage")

    # Investment amounts
    min_investment_amount: float = Field(default=100.0, ge=100, description="Minimum investment in EUR")
    max_investment_amount: float = Field(..., ge=100, le=1000, description="Maximum investment in EUR")

    # Flags
    is_undervalued: bool
    is_affordable: bool

    # Scoring
    attractiveness_score: float = Field(..., ge=0, le=100, description="Investment attractiveness (0-100)")
    confidence_score: float = Field(..., ge=0, le=100, description="AI confidence score (0-100)")
    recommendation: InvestmentRecommendation

    # Analysis
    analysis_summary: str
    risk_factors: List[str]
    upside_factors: List[str]

    # Property details
    property_type: Optional[str] = "APARTMENT"
    size_sqm: Optional[float] = None
    rooms: Optional[int] = None
    year_built: Optional[int] = None

    @validator('attractiveness_score', 'confidence_score')
    def validate_scores(cls, v):
        """Ensure scores are within valid range"""
        return max(0, min(100, v))


class AffordablePropertiesRequest(BaseModel):
    """Request model for affordable properties search"""
    budget: float = Field(..., ge=100, le=5000, description="Monthly budget in EUR")
    location: Optional[str] = Field(None, description="Geographic preference")
    max_results: int = Field(10, ge=1, le=50, description="Maximum results to return")
    risk_tolerance: str = Field("MEDIUM", description="Risk tolerance: LOW, MEDIUM, HIGH")


class AffordablePropertiesResponse(BaseModel):
    """Response model for affordable properties endpoint"""
    opportunities: List[PropertyOpportunitySchema]
    total_count: int
    user_budget: float
    cities_scanned: int
    analysis_time_seconds: float
    filters_applied: Optional[Dict[str, Any]] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CityOpportunitiesResponse(BaseModel):
    """Response model for city-specific opportunities"""
    city: str
    opportunities: List[PropertyOpportunitySchema]
    market_summary: Dict[str, Any]
    analysis_timestamp: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PropertyAnalysisResponse(BaseModel):
    """Response model for detailed property analysis"""
    property_id: str
    detailed_analysis: str
    undervaluation_percentage: float = Field(..., ge=0)
    confidence_score: float = Field(..., ge=0, le=100)
    investment_recommendation: InvestmentRecommendation
    risk_assessment: List[str]
    upside_factors: List[str]
    estimated_holding_period: Optional[str] = None
    expected_roi_range: Optional[str] = None


class OneTapPurchaseRequest(BaseModel):
    """Request model for one-tap purchase"""
    property_id: str = Field(..., description="Property identifier")
    investment_amount: float = Field(
        ..., 
        ge=100.0, 
        le=1000.0, 
        description="Investment amount in EUR (100-1000)"
    )
    user_id: str = Field(..., description="User identifier")
    payment_method: Optional[str] = Field("CARD", description="Payment method")

    @validator('investment_amount')
    def validate_investment_amount(cls, v):
        """Validate investment amount is in acceptable range"""
        if v < 100:
            raise ValueError("Investment amount must be at least 100 EUR")
        if v > 1000:
            raise ValueError("Investment amount cannot exceed 1000 EUR")
        return v


class OneTapPurchaseResponse(BaseModel):
    """Response model for one-tap purchase"""
    success: bool
    transaction_id: Optional[str] = None
    certificate_token: Optional[str] = None
    ownership_percentage: Optional[float] = None
    property_value: Optional[float] = None
    total_cost: Optional[float] = None
    fees_paid: Optional[float] = None
    blockchain_network: Optional[str] = None
    estimated_annual_return: Optional[float] = None
    message: str

    @validator('ownership_percentage')
    def validate_ownership_percentage(cls, v):
        """Ensure ownership percentage is reasonable"""
        if v is not None and v > 100:
            raise ValueError("Ownership percentage cannot exceed 100%")
        return v


class UserPreferencesSchema(BaseModel):
    """User preferences for property recommendations"""
    preferred_cities: List[str] = []
    property_types: List[PropertyType] = [PropertyType.APARTMENT]
    min_size_sqm: Optional[float] = None
    max_size_sqm: Optional[float] = None
    risk_tolerance: str = "MEDIUM"
    investment_horizon: str = "MEDIUM"


class MarketSummarySchema(BaseModel):
    """Market summary for a city"""
    city: str
    average_price_per_sqm: float
    price_trend_6m: float
    inventory_count: int
    market_sentiment: str
    investment_attractiveness: float
    rental_yield: float


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Export all schemas
__all__ = [
    'HealthCheckResponse',
    'PropertyOpportunitySchema', 
    'AffordablePropertiesRequest',
    'AffordablePropertiesResponse',
    'CityOpportunitiesResponse',
    'PropertyAnalysisResponse',
    'OneTapPurchaseRequest',
    'OneTapPurchaseResponse',
    'UserPreferencesSchema',
    'MarketSummarySchema',
    'ErrorResponse'
]
