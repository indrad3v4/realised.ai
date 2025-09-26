"""
Core business entities for Team-Realized platform
All entities in one file for hackathon MVP
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid


class InvestmentRecommendation(str, Enum):
    """Investment recommendation types"""
    BUY = "BUY"
    HOLD = "HOLD" 
    SELL = "SELL"
    PASS = "PASS"
    REVIEW = "REVIEW"


class PropertyType(str, Enum):
    """Property types supported"""
    APARTMENT = "APARTMENT"
    HOUSE = "HOUSE"
    CONDO = "CONDO"
    STUDIO = "STUDIO"


class TransactionStatus(str, Enum):
    """Transaction status types"""
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class UserBudget:
    """User's investment budget and preferences"""
    user_id: str
    monthly_budget: float  # EUR per month
    max_single_investment: float  # Maximum EUR for single property
    preferred_cities: List[str]
    risk_tolerance: str  # LOW, MEDIUM, HIGH
    investment_horizon: str  # SHORT, MEDIUM, LONG
    created_at: datetime

    def can_afford(self, investment_amount: float) -> bool:
        """Check if user can afford investment amount"""
        return investment_amount <= self.max_single_investment

    def get_affordable_range(self) -> tuple[float, float]:
        """Get affordable investment range"""
        min_investment = 100.0  # Minimum EUR 100
        max_investment = min(self.max_single_investment, self.monthly_budget)
        return (min_investment, max_investment)


@dataclass
class AffordableProperty:
    """Property that user can afford to invest in"""
    property_id: str
    city: str
    address: str
    property_type: PropertyType
    total_value: float  # Total property value in EUR
    size_sqm: float
    rooms: int
    year_built: int
    price_per_sqm: float

    # Investment metrics
    min_investment: float = 100.0  # EUR
    max_investment: float = 1000.0  # EUR
    is_affordable: bool = True

    # Location data
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    neighborhood: Optional[str] = None

    def get_investment_percentage(self, investment_amount: float) -> float:
        """Calculate ownership percentage for investment amount"""
        if self.total_value <= 0:
            return 0.0
        return (investment_amount / self.total_value) * 100

    def get_monthly_rental_yield(self) -> float:
        """Estimate monthly rental yield (mock calculation)"""
        # Simple heuristic: 0.5% of property value per month
        return self.total_value * 0.005


@dataclass  
class MarketAnalysis:
    """Market analysis data for a city/region"""
    city: str
    country: str
    analysis_date: datetime

    # Market metrics
    average_price_per_sqm: float
    price_trend_6m: float  # Percentage change in 6 months
    price_trend_1y: float  # Percentage change in 1 year
    inventory_levels: int  # Number of properties available
    demand_supply_ratio: float  # >1.0 means high demand

    # Economic indicators
    unemployment_rate: float
    gdp_growth: float
    population_growth: float

    # Investment metrics
    rental_yield_average: float  # Annual rental yield %
    liquidity_score: float  # 0-10 how easy to sell
    market_sentiment: float  # 0-10 market sentiment score

    def is_hot_market(self) -> bool:
        """Determine if market is 'hot' for investment"""
        return (
            self.price_trend_6m > 3.0 and  # >3% growth in 6m
            self.demand_supply_ratio > 1.2 and  # High demand
            self.market_sentiment > 7.0  # Positive sentiment
        )


@dataclass
class Opportunity:
    """Investment opportunity combining property and analysis"""
    # Identifiers
    opportunity_id: str
    property_id: str

    # Property details
    city: str
    address: str
    total_property_value: float
    predicted_fair_value: float

    # Investment analysis
    undervaluation_percentage: float
    min_investment_amount: float
    max_investment_amount: float

    # Flags
    is_undervalued: bool
    is_affordable: bool

    # Scoring and recommendations
    attractiveness_score: float  # 0-100
    confidence_score: float  # 0-100
    recommendation: InvestmentRecommendation

    # Analysis details
    analysis_summary: str
    risk_factors: List[str]
    upside_factors: List[str]

    # Metadata
    created_at: datetime
    expires_at: datetime

    @classmethod
    def create_from_property(
        cls, 
        property: AffordableProperty,
        analysis: MarketAnalysis,
        ai_prediction: float,
        user_budget: UserBudget
    ) -> 'Opportunity':
        """Create opportunity from property and analysis"""

        undervaluation = ((ai_prediction - property.total_value) / property.total_value) * 100
        is_undervalued = undervaluation > 5.0

        # Calculate attractiveness score
        attractiveness = 50.0  # Base score
        if is_undervalued:
            attractiveness += min(undervaluation * 2, 40)  # Up to +40 for undervaluation
        if analysis.is_hot_market():
            attractiveness += 10  # +10 for hot market

        attractiveness = max(0, min(100, attractiveness))  # Clamp 0-100

        # Investment amounts
        min_inv, max_inv = user_budget.get_affordable_range()

        return cls(
            opportunity_id=str(uuid.uuid4()),
            property_id=property.property_id,
            city=property.city,
            address=property.address,
            total_property_value=property.total_value,
            predicted_fair_value=ai_prediction,
            undervaluation_percentage=max(0, undervaluation),
            min_investment_amount=min_inv,
            max_investment_amount=max_inv,
            is_undervalued=is_undervalued,
            is_affordable=user_budget.can_afford(min_inv),
            attractiveness_score=attractiveness,
            confidence_score=75.0,  # Default confidence
            recommendation=InvestmentRecommendation.BUY if attractiveness > 70 else InvestmentRecommendation.HOLD,
            analysis_summary=f"Property in {property.city} showing {undervaluation:.1f}% value opportunity",
            risk_factors=["Market volatility", "Liquidity risk"],
            upside_factors=["Undervalued asset", "Growing market"],
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow().replace(hour=23, minute=59, second=59)  # Expires end of day
        )


@dataclass
class OwnershipPiece:
    """Represents fractional ownership of a property"""
    certificate_id: str
    user_id: str
    property_id: str

    # Ownership details
    investment_amount: float  # EUR invested
    ownership_percentage: float  # % of property owned
    purchase_price_per_sqm: float

    # Blockchain data
    blockchain_token_id: str
    transaction_hash: str
    smart_contract_address: str

    # Metadata
    purchase_date: datetime
    current_value: float  # Current estimated value

    def calculate_current_value(self, current_property_value: float) -> float:
        """Calculate current value of ownership piece"""
        self.current_value = (self.ownership_percentage / 100) * current_property_value
        return self.current_value

    def calculate_roi(self) -> float:
        """Calculate return on investment percentage"""
        if self.investment_amount <= 0:
            return 0.0
        return ((self.current_value - self.investment_amount) / self.investment_amount) * 100


@dataclass
class Transaction:
    """Investment transaction record"""
    transaction_id: str
    user_id: str
    opportunity_id: str
    property_id: str

    # Transaction details
    investment_amount: float
    ownership_percentage: float
    status: TransactionStatus

    # Payment details
    payment_method: str  # CARD, CRYPTO, BANK_TRANSFER
    payment_reference: str

    # Blockchain details
    blockchain_network: str = "SOLANA"
    token_mint_address: Optional[str] = None
    certificate_token_id: Optional[str] = None

    # Timestamps
    initiated_at: datetime
    completed_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None

    # Metadata
    fees_paid: float = 0.0
    exchange_rate_eur_usd: Optional[float] = None

    def mark_completed(self, certificate_token: str, blockchain_hash: str):
        """Mark transaction as completed"""
        self.status = TransactionStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.certificate_token_id = certificate_token
        self.token_mint_address = blockchain_hash

    def mark_failed(self, reason: str):
        """Mark transaction as failed"""
        self.status = TransactionStatus.FAILED
        self.failed_at = datetime.utcnow()


@dataclass
class Portfolio:
    """User's real estate investment portfolio"""
    user_id: str
    total_invested: float = 0.0
    current_value: float = 0.0
    total_properties: int = 0

    # Holdings
    ownership_pieces: List[OwnershipPiece] = None

    def __post_init__(self):
        if self.ownership_pieces is None:
            self.ownership_pieces = []

    def add_ownership(self, ownership: OwnershipPiece):
        """Add new ownership piece to portfolio"""
        self.ownership_pieces.append(ownership)
        self.total_invested += ownership.investment_amount
        self.current_value += ownership.current_value
        self.total_properties = len(set(op.property_id for op in self.ownership_pieces))

    def calculate_total_roi(self) -> float:
        """Calculate total portfolio ROI"""
        if self.total_invested <= 0:
            return 0.0
        return ((self.current_value - self.total_invested) / self.total_invested) * 100

    def get_diversification_score(self) -> float:
        """Calculate portfolio diversification (0-100)"""
        if not self.ownership_pieces:
            return 0.0

        unique_cities = len(set(op.property_id.split('_')[0] for op in self.ownership_pieces))
        max_cities = min(10, len(self.ownership_pieces))  # Cap at 10 for scoring

        return (unique_cities / max_cities) * 100 if max_cities > 0 else 0.0


# Export all entities
__all__ = [
    'UserBudget', 'AffordableProperty', 'MarketAnalysis', 'Opportunity',
    'OwnershipPiece', 'Transaction', 'Portfolio',
    'InvestmentRecommendation', 'PropertyType', 'TransactionStatus'
]
