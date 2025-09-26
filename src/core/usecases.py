"""
Core use cases for Team-Realized platform
Business logic and application services
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from .entities import (
    UserBudget, AffordableProperty, MarketAnalysis, Opportunity,
    OwnershipPiece, Transaction, Portfolio, TransactionStatus,
    InvestmentRecommendation, PropertyType
)

logger = logging.getLogger(__name__)


class FindAffordablePropertiesUseCase:
    """Find properties user can afford based on budget"""

    def __init__(self, property_repository, market_repository):
        self.property_repository = property_repository
        self.market_repository = market_repository

    async def execute(
        self, 
        user_budget: UserBudget,
        location_filter: Optional[str] = None,
        max_results: int = 10
    ) -> List[AffordableProperty]:
        """Find affordable properties for user"""

        logger.info(f"Finding properties for budget: {user_budget.monthly_budget} EUR")

        try:
            # Get investment range
            min_investment, max_investment = user_budget.get_affordable_range()

            # Apply filters
            filters = {
                'min_total_value': min_investment * 100,  # Min property value = 100x investment
                'max_total_value': max_investment * 500,  # Max property value = 500x investment
                'cities': user_budget.preferred_cities if user_budget.preferred_cities else None,
                'location_filter': location_filter
            }

            # Fetch properties from repository
            properties = await self.property_repository.find_affordable(
                filters=filters,
                limit=max_results * 2  # Get extra for filtering
            )

            # Filter by affordability
            affordable_properties = []
            for prop in properties:
                prop.min_investment = min_investment
                prop.max_investment = min(max_investment, prop.total_value * 0.1)  # Max 10% of property

                if user_budget.can_afford(prop.min_investment):
                    prop.is_affordable = True
                    affordable_properties.append(prop)

                if len(affordable_properties) >= max_results:
                    break

            logger.info(f"Found {len(affordable_properties)} affordable properties")
            return affordable_properties

        except Exception as e:
            logger.error(f"Error finding affordable properties: {e}")
            raise


class DetectUndervaluationUseCase:
    """Detect undervalued properties using AI analysis"""

    def __init__(self, ai_service, market_repository):
        self.ai_service = ai_service
        self.market_repository = market_repository

    async def execute(
        self,
        properties: List[AffordableProperty],
        user_budget: UserBudget
    ) -> List[Opportunity]:
        """Detect undervalued properties and create opportunities"""

        logger.info(f"Analyzing {len(properties)} properties for undervaluation")

        opportunities = []

        try:
            for prop in properties:
                # Get market analysis for city
                market_analysis = await self.market_repository.get_city_analysis(prop.city)
                if not market_analysis:
                    # Create default market analysis
                    market_analysis = MarketAnalysis(
                        city=prop.city,
                        country="Europe", 
                        analysis_date=datetime.utcnow(),
                        average_price_per_sqm=prop.price_per_sqm * 1.1,
                        price_trend_6m=2.5,
                        price_trend_1y=5.0,
                        inventory_levels=100,
                        demand_supply_ratio=1.1,
                        unemployment_rate=4.0,
                        gdp_growth=2.0,
                        population_growth=1.0,
                        rental_yield_average=4.5,
                        liquidity_score=7.0,
                        market_sentiment=7.5
                    )

                # Get AI prediction for fair value
                if self.ai_service:
                    predicted_value = await self.ai_service.predict_property_value(prop)
                else:
                    # Mock prediction - add some variance
                    variance = 0.9 + (hash(prop.property_id) % 40) / 100  # 0.9 to 1.3
                    predicted_value = prop.total_value * variance

                # Create opportunity
                opportunity = Opportunity.create_from_property(
                    property=prop,
                    analysis=market_analysis,
                    ai_prediction=predicted_value,
                    user_budget=user_budget
                )

                opportunities.append(opportunity)

            # Sort by attractiveness score
            opportunities.sort(key=lambda x: x.attractiveness_score, reverse=True)

            logger.info(f"Created {len(opportunities)} investment opportunities")
            return opportunities

        except Exception as e:
            logger.error(f"Error detecting undervaluation: {e}")
            raise


class OneTapPurchaseUseCase:
    """Execute one-tap purchase of property ownership"""

    def __init__(self, payment_service, blockchain_service, transaction_repository):
        self.payment_service = payment_service
        self.blockchain_service = blockchain_service
        self.transaction_repository = transaction_repository

    async def execute(
        self,
        user_id: str,
        opportunity_id: str,
        investment_amount: float
    ) -> Transaction:
        """Execute one-tap purchase flow"""

        logger.info(f"Processing purchase: {opportunity_id} for {investment_amount} EUR")

        # Validate investment amount
        if investment_amount < 100 or investment_amount > 1000:
            raise ValueError("Investment amount must be between 100 EUR and 1000 EUR")

        # Create transaction record
        transaction = Transaction(
            transaction_id=f"txn_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{user_id[:8]}",
            user_id=user_id,
            opportunity_id=opportunity_id,
            property_id=opportunity_id.split('_')[0] + "_prop",  # Extract property ID
            investment_amount=investment_amount,
            ownership_percentage=(investment_amount / 200000) * 100,  # Assume 200k property
            status=TransactionStatus.PENDING,
            payment_method="CARD",
            payment_reference=f"pay_{transaction.transaction_id}",
            initiated_at=datetime.utcnow(),
            fees_paid=investment_amount * 0.03  # 3% fee
        )

        try:
            # Save pending transaction
            await self.transaction_repository.save(transaction)

            # Process payment
            payment_result = await self.payment_service.process_payment(
                amount=investment_amount + transaction.fees_paid,
                user_id=user_id,
                reference=transaction.payment_reference
            )

            if not payment_result.success:
                transaction.mark_failed("Payment failed")
                await self.transaction_repository.save(transaction)
                raise Exception(f"Payment failed: {payment_result.error}")

            # Issue blockchain certificate
            certificate_result = await self.blockchain_service.issue_certificate(
                user_id=user_id,
                property_id=transaction.property_id,
                ownership_percentage=transaction.ownership_percentage,
                investment_amount=investment_amount
            )

            if not certificate_result.success:
                # TODO: Refund payment
                transaction.mark_failed("Certificate issuance failed")
                await self.transaction_repository.save(transaction)
                raise Exception(f"Certificate issuance failed: {certificate_result.error}")

            # Mark transaction completed
            transaction.mark_completed(
                certificate_result.token_id,
                certificate_result.blockchain_hash
            )

            await self.transaction_repository.save(transaction)

            logger.info(f"Purchase completed: {transaction.transaction_id}")
            return transaction

        except Exception as e:
            logger.error(f"Purchase failed: {e}")
            transaction.mark_failed(str(e))
            await self.transaction_repository.save(transaction)
            raise


class ScanCitiesForOpportunitiesUseCase:
    """Scan multiple cities for investment opportunities"""

    def __init__(self, property_repository, ai_service):
        self.property_repository = property_repository
        self.ai_service = ai_service

    async def execute(
        self,
        cities: List[str],
        user_budget: UserBudget,
        max_per_city: int = 5
    ) -> Dict[str, List[Opportunity]]:
        """Scan cities for opportunities"""

        logger.info(f"Scanning {len(cities)} cities for opportunities")

        city_opportunities = {}

        try:
            for city in cities:
                # Find properties in city
                city_properties = await self.property_repository.find_by_city(
                    city=city,
                    limit=max_per_city * 2
                )

                if not city_properties:
                    continue

                # Filter affordable properties
                find_affordable = FindAffordablePropertiesUseCase(
                    self.property_repository, 
                    None  # Market repository not needed here
                )

                affordable = []
                for prop in city_properties:
                    min_inv, max_inv = user_budget.get_affordable_range()
                    prop.min_investment = min_inv
                    prop.max_investment = min(max_inv, prop.total_value * 0.1)

                    if user_budget.can_afford(prop.min_investment):
                        prop.is_affordable = True
                        affordable.append(prop)

                # Detect undervaluation
                if affordable and self.ai_service:
                    detect_undervaluation = DetectUndervaluationUseCase(
                        self.ai_service,
                        None  # Market repository not needed here
                    )

                    opportunities = await detect_undervaluation.execute(
                        affordable[:max_per_city],
                        user_budget
                    )

                    if opportunities:
                        city_opportunities[city] = opportunities

            logger.info(f"Found opportunities in {len(city_opportunities)} cities")
            return city_opportunities

        except Exception as e:
            logger.error(f"Error scanning cities: {e}")
            raise


class InstantCertificateUseCase:
    """Issue instant blockchain ownership certificates"""

    def __init__(self, blockchain_service):
        self.blockchain_service = blockchain_service

    async def execute(
        self,
        transaction: Transaction
    ) -> OwnershipPiece:
        """Create ownership piece from completed transaction"""

        logger.info(f"Creating ownership certificate for {transaction.transaction_id}")

        try:
            ownership = OwnershipPiece(
                certificate_id=transaction.certificate_token_id,
                user_id=transaction.user_id,
                property_id=transaction.property_id,
                investment_amount=transaction.investment_amount,
                ownership_percentage=transaction.ownership_percentage,
                purchase_price_per_sqm=0.0,  # TODO: Calculate from property data
                blockchain_token_id=transaction.certificate_token_id,
                transaction_hash=transaction.token_mint_address,
                smart_contract_address="SOLANA_PROGRAM_ADDRESS",  # TODO: Real address
                purchase_date=transaction.completed_at,
                current_value=transaction.investment_amount  # Initial value = investment
            )

            logger.info(f"Ownership certificate created: {ownership.certificate_id}")
            return ownership

        except Exception as e:
            logger.error(f"Error creating certificate: {e}")
            raise


# Export all use cases
__all__ = [
    'FindAffordablePropertiesUseCase',
    'DetectUndervaluationUseCase', 
    'OneTapPurchaseUseCase',
    'ScanCitiesForOpportunitiesUseCase',
    'InstantCertificateUseCase'
]
