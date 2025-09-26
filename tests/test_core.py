"""
Tests for Team-Realized core business logic
Essential tests for hackathon demo
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from ..core.entities import (
    UserBudget, AffordableProperty, Opportunity, Transaction,
    InvestmentRecommendation, PropertyType, TransactionStatus
)
from ..core.usecases import (
    FindAffordablePropertiesUseCase,
    DetectUndervaluationUseCase,
    OneTapPurchaseUseCase
)


class TestUserBudget:
    """Test UserBudget entity"""

    def test_create_user_budget(self):
        """Test creating a user budget"""
        budget = UserBudget(
            user_id="test_user",
            monthly_budget=500.0,
            max_single_investment=1000.0,
            preferred_cities=["Krakow", "Berlin"],
            risk_tolerance="MEDIUM",
            investment_horizon="MEDIUM",
            created_at=datetime.utcnow()
        )

        assert budget.user_id == "test_user"
        assert budget.monthly_budget == 500.0
        assert budget.max_single_investment == 1000.0
        assert "Krakow" in budget.preferred_cities

    def test_can_afford(self):
        """Test affordability check"""
        budget = UserBudget(
            user_id="test_user",
            monthly_budget=500.0,
            max_single_investment=800.0,
            preferred_cities=[],
            risk_tolerance="MEDIUM",
            investment_horizon="MEDIUM",
            created_at=datetime.utcnow()
        )

        assert budget.can_afford(600.0) is True
        assert budget.can_afford(900.0) is False

    def test_get_affordable_range(self):
        """Test getting affordable investment range"""
        budget = UserBudget(
            user_id="test_user",
            monthly_budget=500.0,
            max_single_investment=800.0,
            preferred_cities=[],
            risk_tolerance="MEDIUM", 
            investment_horizon="MEDIUM",
            created_at=datetime.utcnow()
        )

        min_inv, max_inv = budget.get_affordable_range()
        assert min_inv == 100.0  # Minimum EUR 100
        assert max_inv == 500.0  # min(max_single_investment, monthly_budget)


class TestAffordableProperty:
    """Test AffordableProperty entity"""

    def test_create_property(self):
        """Test creating a property"""
        property = AffordableProperty(
            property_id="krakow_prop_1",
            city="Krakow",
            address="Sample Street 1, Krakow",
            property_type=PropertyType.APARTMENT,
            total_value=180000.0,
            size_sqm=70.0,
            rooms=3,
            year_built=2010,
            price_per_sqm=2571.43
        )

        assert property.property_id == "krakow_prop_1"
        assert property.city == "Krakow"
        assert property.property_type == PropertyType.APARTMENT
        assert property.total_value == 180000.0

    def test_get_investment_percentage(self):
        """Test investment percentage calculation"""
        property = AffordableProperty(
            property_id="test_prop",
            city="Test City",
            address="Test Address",
            property_type=PropertyType.APARTMENT,
            total_value=200000.0,
            size_sqm=80.0,
            rooms=3,
            year_built=2015,
            price_per_sqm=2500.0
        )

        # €1000 investment in €200k property = 0.5%
        percentage = property.get_investment_percentage(1000.0)
        assert percentage == 0.5

    def test_get_monthly_rental_yield(self):
        """Test rental yield estimation"""
        property = AffordableProperty(
            property_id="test_prop",
            city="Test City", 
            address="Test Address",
            property_type=PropertyType.APARTMENT,
            total_value=200000.0,
            size_sqm=80.0,
            rooms=3,
            year_built=2015,
            price_per_sqm=2500.0
        )

        # 0.5% of property value per month
        rental_yield = property.get_monthly_rental_yield()
        assert rental_yield == 1000.0  # 200,000 * 0.005


class TestOpportunity:
    """Test Opportunity entity"""

    def test_create_opportunity_from_property(self):
        """Test creating opportunity from property"""
        property = AffordableProperty(
            property_id="test_prop",
            city="Krakow",
            address="Test Street 1",
            property_type=PropertyType.APARTMENT,
            total_value=180000.0,
            size_sqm=70.0,
            rooms=3,
            year_built=2010,
            price_per_sqm=2571.43
        )

        from ..core.entities import MarketAnalysis
        market_analysis = MarketAnalysis(
            city="Krakow",
            country="Poland",
            analysis_date=datetime.utcnow(),
            average_price_per_sqm=2500.0,
            price_trend_6m=5.0,
            price_trend_1y=8.0,
            inventory_levels=150,
            demand_supply_ratio=1.2,
            unemployment_rate=3.5,
            gdp_growth=3.0,
            population_growth=1.0,
            rental_yield_average=5.0,
            liquidity_score=8.0,
            market_sentiment=8.5
        )

        user_budget = UserBudget(
            user_id="test_user",
            monthly_budget=500.0,
            max_single_investment=1000.0,
            preferred_cities=["Krakow"],
            risk_tolerance="MEDIUM",
            investment_horizon="MEDIUM",
            created_at=datetime.utcnow()
        )

        # AI predicts 10% undervaluation
        ai_prediction = 198000.0  # 10% higher than market price

        opportunity = Opportunity.create_from_property(
            property, market_analysis, ai_prediction, user_budget
        )

        assert opportunity.property_id == "test_prop"
        assert opportunity.city == "Krakow"
        assert opportunity.undervaluation_percentage == 10.0
        assert opportunity.is_undervalued is True
        assert opportunity.recommendation == InvestmentRecommendation.BUY


class TestFindAffordablePropertiesUseCase:
    """Test FindAffordablePropertiesUseCase"""

    @pytest.mark.asyncio
    async def test_execute_find_properties(self):
        """Test finding affordable properties"""
        # Mock repository
        mock_property_repo = Mock()
        mock_market_repo = Mock()

        # Mock properties
        mock_properties = [
            AffordableProperty(
                property_id="krakow_prop_1",
                city="Krakow",
                address="Sample Street 1",
                property_type=PropertyType.APARTMENT,
                total_value=180000.0,
                size_sqm=70.0,
                rooms=3,
                year_built=2010,
                price_per_sqm=2571.43
            )
        ]

        mock_property_repo.find_affordable = AsyncMock(return_value=mock_properties)

        # Create use case
        use_case = FindAffordablePropertiesUseCase(mock_property_repo, mock_market_repo)

        # Create user budget
        user_budget = UserBudget(
            user_id="test_user",
            monthly_budget=500.0,
            max_single_investment=1000.0,
            preferred_cities=["Krakow"],
            risk_tolerance="MEDIUM",
            investment_horizon="MEDIUM",
            created_at=datetime.utcnow()
        )

        # Execute use case
        result = await use_case.execute(user_budget)

        # Verify results
        assert len(result) == 1
        assert result[0].property_id == "krakow_prop_1"
        assert result[0].is_affordable is True


class TestOneTapPurchaseUseCase:
    """Test OneTapPurchaseUseCase"""

    @pytest.mark.asyncio
    async def test_successful_purchase(self):
        """Test successful one-tap purchase"""
        # Mock services
        mock_payment_service = Mock()
        mock_blockchain_service = Mock()
        mock_transaction_repo = Mock()

        # Mock successful payment
        from ..adapters.blockchain import PaymentResult
        mock_payment_service.process_payment = AsyncMock(
            return_value=PaymentResult(success=True, transaction_id="pay_123")
        )

        # Mock successful certificate
        from ..adapters.blockchain import CertificateResult  
        mock_blockchain_service.issue_certificate = AsyncMock(
            return_value=CertificateResult(
                success=True,
                token_id="CERT_123",
                blockchain_hash="hash_123"
            )
        )

        mock_transaction_repo.save = AsyncMock(return_value=True)

        # Create use case
        use_case = OneTapPurchaseUseCase(
            mock_payment_service,
            mock_blockchain_service, 
            mock_transaction_repo
        )

        # Execute purchase
        result = await use_case.execute(
            user_id="test_user",
            opportunity_id="test_opp_1",
            investment_amount=500.0
        )

        # Verify results
        assert result.status == TransactionStatus.COMPLETED
        assert result.investment_amount == 500.0
        assert result.certificate_token_id == "CERT_123"

    @pytest.mark.asyncio
    async def test_invalid_investment_amount(self):
        """Test invalid investment amount"""
        use_case = OneTapPurchaseUseCase(Mock(), Mock(), Mock())

        with pytest.raises(ValueError, match="Investment amount must be between"):
            await use_case.execute("test_user", "test_opp", 50.0)  # Too low

        with pytest.raises(ValueError, match="Investment amount must be between"):
            await use_case.execute("test_user", "test_opp", 2000.0)  # Too high


class TestTransaction:
    """Test Transaction entity"""

    def test_create_transaction(self):
        """Test creating a transaction"""
        transaction = Transaction(
            transaction_id="txn_123",
            user_id="user_123",
            opportunity_id="opp_123",
            property_id="prop_123",
            investment_amount=500.0,
            ownership_percentage=0.25,
            status=TransactionStatus.PENDING,
            payment_method="CARD",
            payment_reference="pay_ref_123",
            initiated_at=datetime.utcnow()
        )

        assert transaction.transaction_id == "txn_123"
        assert transaction.status == TransactionStatus.PENDING
        assert transaction.investment_amount == 500.0

    def test_mark_completed(self):
        """Test marking transaction as completed"""
        transaction = Transaction(
            transaction_id="txn_123",
            user_id="user_123",
            opportunity_id="opp_123",
            property_id="prop_123",
            investment_amount=500.0,
            ownership_percentage=0.25,
            status=TransactionStatus.PENDING,
            payment_method="CARD",
            payment_reference="pay_ref_123",
            initiated_at=datetime.utcnow()
        )

        transaction.mark_completed("cert_123", "hash_123")

        assert transaction.status == TransactionStatus.COMPLETED
        assert transaction.certificate_token_id == "cert_123"
        assert transaction.token_mint_address == "hash_123"
        assert transaction.completed_at is not None

    def test_mark_failed(self):
        """Test marking transaction as failed"""
        transaction = Transaction(
            transaction_id="txn_123",
            user_id="user_123",
            opportunity_id="opp_123", 
            property_id="prop_123",
            investment_amount=500.0,
            ownership_percentage=0.25,
            status=TransactionStatus.PENDING,
            payment_method="CARD",
            payment_reference="pay_ref_123",
            initiated_at=datetime.utcnow()
        )

        transaction.mark_failed("Payment declined")

        assert transaction.status == TransactionStatus.FAILED
        assert transaction.failed_at is not None


# Test fixtures
@pytest.fixture
def sample_user_budget():
    """Sample user budget for testing"""
    return UserBudget(
        user_id="test_user",
        monthly_budget=500.0,
        max_single_investment=1000.0,
        preferred_cities=["Krakow", "Berlin"],
        risk_tolerance="MEDIUM",
        investment_horizon="MEDIUM",
        created_at=datetime.utcnow()
    )


@pytest.fixture
def sample_property():
    """Sample property for testing"""
    return AffordableProperty(
        property_id="test_prop_1",
        city="Krakow",
        address="Sample Street 1, Krakow",
        property_type=PropertyType.APARTMENT,
        total_value=180000.0,
        size_sqm=70.0,
        rooms=3,
        year_built=2010,
        price_per_sqm=2571.43
    )
