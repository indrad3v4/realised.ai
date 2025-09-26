"""
Real Estate API adapter for Team-Realized platform
Integration with property data APIs across 50+ European cities
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import aiohttp
import random

from ..core.entities import AffordableProperty, MarketAnalysis, PropertyType
from ..core.config import settings

logger = logging.getLogger(__name__)


class PropertyDataProvider:
    """Base class for property data providers"""

    def __init__(self, name: str, base_url: str, api_key: str = None):
        self.name = name
        self.base_url = base_url
        self.api_key = api_key
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": "Team-Realized/1.0"}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()


class MockPropertyProvider(PropertyDataProvider):
    """Mock property data provider for hackathon demonstration"""

    def __init__(self):
        super().__init__("MockProvider", "https://mock-api.com")

    async def search_properties(
        self,
        city: str,
        min_price: float = None,
        max_price: float = None,
        property_type: PropertyType = None,
        limit: int = 10
    ) -> List[AffordableProperty]:
        """Search for properties in a city"""

        logger.info(f"Searching properties in {city} (mock data)")

        properties = []

        # Generate realistic mock properties
        base_prices = {
            "Krakow": 180000, "Berlin": 450000, "Prague": 220000,
            "Barcelona": 380000, "Warsaw": 200000, "Vienna": 520000,
            "Budapest": 160000, "Amsterdam": 650000, "Munich": 750000,
            "Hamburg": 480000, "Valencia": 280000, "Lisbon": 350000
        }

        base_price = base_prices.get(city, 300000)

        for i in range(min(limit, 8)):
            # Add variance to price
            price_variance = random.uniform(0.7, 1.4)
            property_value = base_price * price_variance

            # Skip properties outside price range
            if min_price and property_value < min_price:
                continue
            if max_price and property_value > max_price:
                continue

            property_id = f"{city.lower()}_prop_{i+1}_{int(datetime.utcnow().timestamp())}"

            property = AffordableProperty(
                property_id=property_id,
                city=city,
                address=f"{random.choice(['Main', 'Central', 'Market', 'Old', 'New'])} Street {i+1}, {city}",
                property_type=property_type or random.choice(list(PropertyType)),
                total_value=property_value,
                size_sqm=45 + random.randint(15, 85),  # 45-130 sqm
                rooms=random.randint(1, 4),
                year_built=1980 + random.randint(0, 40),
                price_per_sqm=property_value / (45 + random.randint(15, 85)),
                latitude=self._get_city_coords(city)[0] + random.uniform(-0.1, 0.1),
                longitude=self._get_city_coords(city)[1] + random.uniform(-0.1, 0.1),
                neighborhood=f"{city} Center"
            )

            properties.append(property)

        logger.info(f"Found {len(properties)} properties in {city}")
        return properties

    async def get_market_analysis(self, city: str) -> Optional[MarketAnalysis]:
        """Get market analysis for a city"""

        logger.info(f"Getting market analysis for {city} (mock data)")

        # Mock market data with some realism
        city_hash = hash(city)

        analysis = MarketAnalysis(
            city=city,
            country="Europe",
            analysis_date=datetime.utcnow(),
            average_price_per_sqm=2000 + (city_hash % 2000),  # 2000-4000 EUR/sqm
            price_trend_6m=1.0 + (city_hash % 80) / 10,  # 1-8% growth
            price_trend_1y=2.0 + (city_hash % 120) / 10,  # 2-14% growth
            inventory_levels=50 + (city_hash % 300),  # 50-350 properties
            demand_supply_ratio=0.8 + (city_hash % 80) / 100,  # 0.8-1.6 ratio
            unemployment_rate=2.0 + (city_hash % 80) / 10,  # 2-10%
            gdp_growth=1.0 + (city_hash % 40) / 10,  # 1-5%
            population_growth=0.5 + (city_hash % 30) / 10,  # 0.5-3.5%
            rental_yield_average=3.0 + (city_hash % 40) / 10,  # 3-7%
            liquidity_score=5.0 + (city_hash % 50) / 10,  # 5-10
            market_sentiment=6.0 + (city_hash % 40) / 10  # 6-10
        )

        return analysis

    def _get_city_coords(self, city: str) -> tuple[float, float]:
        """Get approximate coordinates for major cities"""
        coords = {
            "Krakow": (50.0647, 19.9450),
            "Berlin": (52.5200, 13.4050),
            "Prague": (50.0755, 14.4378),
            "Barcelona": (41.3851, 2.1734),
            "Warsaw": (52.2297, 21.0122),
            "Vienna": (48.2082, 16.3738),
            "Budapest": (47.4979, 19.0402),
            "Amsterdam": (52.3676, 4.9041),
            "Munich": (48.1351, 11.5820),
            "Hamburg": (53.5511, 9.9937),
            "Valencia": (39.4699, -0.3763),
            "Lisbon": (38.7223, -9.1393)
        }
        return coords.get(city, (50.0, 10.0))  # Default to central Europe


class RealEstateAPIService:
    """Main service for real estate data integration"""

    def __init__(self):
        self.providers = {
            "mock": MockPropertyProvider()
        }
        self.cache = {}  # Simple in-memory cache
        self.cache_ttl = 3600  # 1 hour

    async def search_city_properties(
        self,
        city: str,
        budget_range: tuple[float, float] = None,
        property_types: List[PropertyType] = None,
        limit: int = 20
    ) -> List[AffordableProperty]:
        """Search properties across all providers"""

        cache_key = f"properties_{city}_{budget_range}_{limit}"

        # Check cache first
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if datetime.utcnow().timestamp() - cache_entry["timestamp"] < self.cache_ttl:
                logger.info(f"Returning cached properties for {city}")
                return cache_entry["data"]

        all_properties = []

        try:
            # Use mock provider (in production, would query multiple real APIs)
            provider = self.providers["mock"]

            min_price = budget_range[0] * 100 if budget_range else None  # Min property value = 100x investment
            max_price = budget_range[1] * 500 if budget_range else None  # Max property value = 500x investment

            properties = await provider.search_properties(
                city=city,
                min_price=min_price,
                max_price=max_price,
                limit=limit
            )

            all_properties.extend(properties)

            # Cache results
            self.cache[cache_key] = {
                "data": all_properties,
                "timestamp": datetime.utcnow().timestamp()
            }

            logger.info(f"Found {len(all_properties)} properties in {city}")
            return all_properties

        except Exception as e:
            logger.error(f"Failed to search properties in {city}: {e}")
            return []

    async def get_city_market_data(self, city: str) -> Optional[MarketAnalysis]:
        """Get market analysis for a city"""

        cache_key = f"market_{city}"

        # Check cache
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if datetime.utcnow().timestamp() - cache_entry["timestamp"] < self.cache_ttl:
                logger.info(f"Returning cached market data for {city}")
                return cache_entry["data"]

        try:
            provider = self.providers["mock"]
            analysis = await provider.get_market_analysis(city)

            if analysis:
                # Cache results
                self.cache[cache_key] = {
                    "data": analysis,
                    "timestamp": datetime.utcnow().timestamp()
                }

            return analysis

        except Exception as e:
            logger.error(f"Failed to get market data for {city}: {e}")
            return None

    async def scan_multiple_cities(
        self,
        cities: List[str],
        budget_range: tuple[float, float] = None,
        max_per_city: int = 5
    ) -> Dict[str, List[AffordableProperty]]:
        """Scan multiple cities for properties simultaneously"""

        logger.info(f"Scanning {len(cities)} cities for properties")

        # Create concurrent tasks for each city
        tasks = []
        for city in cities:
            task = self.search_city_properties(
                city=city,
                budget_range=budget_range,
                limit=max_per_city
            )
            tasks.append((city, task))

        # Execute all tasks concurrently
        results = {}
        try:
            completed_tasks = await asyncio.gather(
                *[task for _, task in tasks],
                return_exceptions=True
            )

            for i, (city, _) in enumerate(tasks):
                result = completed_tasks[i]
                if isinstance(result, Exception):
                    logger.error(f"Failed to scan {city}: {result}")
                    results[city] = []
                else:
                    results[city] = result

            total_properties = sum(len(props) for props in results.values())
            logger.info(f"Scanned {len(cities)} cities, found {total_properties} properties")

            return results

        except Exception as e:
            logger.error(f"Failed to scan cities: {e}")
            return {}

    async def get_property_details(self, property_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific property"""

        try:
            # Parse city from property ID
            city = property_id.split('_')[0].title()

            # Mock detailed property info
            details = {
                "property_id": property_id,
                "city": city,
                "detailed_description": f"Beautiful property in {city} with modern amenities and great location.",
                "amenities": [
                    "Central heating",
                    "Balcony", 
                    "Parking space",
                    "Elevator access",
                    "Near public transport"
                ],
                "nearby_facilities": [
                    "Supermarket (200m)",
                    "Metro station (500m)",
                    "School (800m)",
                    "Hospital (1.2km)",
                    "Park (300m)"
                ],
                "energy_rating": random.choice(["A", "B", "C"]),
                "last_sold": "2023-03-15",
                "price_history": [
                    {"date": "2023-01-01", "price": 180000},
                    {"date": "2023-06-01", "price": 185000},
                    {"date": "2024-01-01", "price": 190000}
                ],
                "images": [
                    f"https://mock-api.com/images/{property_id}_1.jpg",
                    f"https://mock-api.com/images/{property_id}_2.jpg",
                    f"https://mock-api.com/images/{property_id}_3.jpg"
                ]
            }

            return details

        except Exception as e:
            logger.error(f"Failed to get property details for {property_id}: {e}")
            return None

    async def estimate_rental_yield(self, property_id: str) -> Optional[float]:
        """Estimate rental yield for a property"""

        try:
            # Mock rental yield estimation
            property_hash = hash(property_id)
            yield_percentage = 3.0 + (property_hash % 50) / 10  # 3-8% yield

            logger.info(f"Estimated rental yield for {property_id}: {yield_percentage:.1f}%")
            return yield_percentage

        except Exception as e:
            logger.error(f"Failed to estimate rental yield for {property_id}: {e}")
            return None

    def clear_cache(self):
        """Clear the property cache"""
        self.cache.clear()
        logger.info("Property cache cleared")


# Global service instance
real_estate_service = RealEstateAPIService()

# Export service
__all__ = ['RealEstateAPIService', 'PropertyDataProvider', 'MockPropertyProvider', 'real_estate_service']
