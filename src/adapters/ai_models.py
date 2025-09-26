#AI Models Integration for Team-Realized Real Estate Tokenization Platform
#Combines Fast.ai, PyTorch, OpenAI Agents SDK, and DeepSeek v3.1 for:
#- 50+ cities real estate market analysis
#- Property undervaluation detection
#- Market sentiment analysis
#- Real-time property opportunity scoring

import os
import asyncio
import logging
import pickle
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Fast.ai imports
from fastai.vision.all import *
from fastai.tabular.all import *
from fastai.text.all import *

# OpenAI Agents SDK
from openai import AsyncOpenAI
import openai

# DeepSeek integration
import httpx

# Core entities
from ..core.entities import AffordableProperty, Opportunity, MarketAnalysis

logger = logging.getLogger(__name__)


@dataclass
class PropertyFeatures:
    """Features extracted from property data for ML models"""
    location_embedding: np.ndarray
    price_per_sqm: float
    property_age: int
    nearby_amenities_count: int
    transport_score: float
    market_sentiment: float
    legal_risk_score: float
    historical_price_trend: List[float]


@dataclass
class CityMarketData:
    """Market data structure for a single city"""
    city_name: str
    average_price_per_sqm: float
    price_trend_6m: float
    inventory_levels: int
    demand_supply_ratio: float
    economic_indicators: Dict[str, float]
    sentiment_score: float


class PropertyImageDataset(Dataset):
    """Fast.ai compatible dataset for property images"""

    def __init__(self, df: pd.DataFrame, transforms=None):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']

        # Load image using Fast.ai
        image = PILImage.create(image_path)

        if self.transforms:
            image = self.transforms(image)

        features = {
            'price': torch.tensor(row['price'], dtype=torch.float32),
            'location_score': torch.tensor(row['location_score'], dtype=torch.float32),
            'property_age': torch.tensor(row['property_age'], dtype=torch.float32),
        }

        return image, features


class PropertyPricePredictor(nn.Module):
    """PyTorch model for property price prediction"""

    def __init__(self, input_dim: int = 256, hidden_dims: List[int] = [512, 256, 128]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim

        # Final layer for price prediction
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class FastAIPropertyFeatureExtractor:
    """Fast.ai based feature extractor for property images and data"""

    def __init__(self, model_path: str = 'models/property_feature_extractor.pkl'):
        self.model_path = Path(model_path)
        self.vision_learner = None
        self.tabular_learner = None
        self._load_models()

    def _load_models(self):
        """Load pre-trained Fast.ai models"""
        try:
            if self.model_path.exists():
                self.vision_learner = load_learner(self.model_path / 'vision_model.pkl')
                self.tabular_learner = load_learner(self.model_path / 'tabular_model.pkl')
                logger.info("Fast.ai models loaded successfully")
            else:
                logger.warning("Model files not found, will create new models")
                self._create_default_models()
        except Exception as e:
            logger.error(f"Error loading Fast.ai models: {e}")
            self._create_default_models()

    def _create_default_models(self):
        """Create default models if pre-trained ones don't exist"""
        # This would typically be trained on property data
        logger.info("Creating default Fast.ai models - should be replaced with trained models")

    async def extract_image_features(self, image_path: str) -> np.ndarray:
        """Extract features from property image using Fast.ai vision model"""
        try:
            if not self.vision_learner:
                return np.random.random(512)  # Fallback for demo

            # Load and predict using Fast.ai
            img = PILImage.create(image_path)
            features = self.vision_learner.model[:-1](img.unsqueeze(0))
            return features.detach().numpy().flatten()

        except Exception as e:
            logger.error(f"Error extracting image features: {e}")
            return np.random.random(512)  # Fallback

    async def extract_tabular_features(self, property_data: Dict) -> np.ndarray:
        """Extract features from tabular property data"""
        try:
            if not self.tabular_learner:
                # Manual feature engineering as fallback
                features = [
                    property_data.get('price_per_sqm', 0),
                    property_data.get('property_age', 0),
                    property_data.get('rooms', 0),
                    property_data.get('floor', 0),
                    property_data.get('total_floors', 0),
                ]
                return np.array(features, dtype=np.float32)

            # Use Fast.ai tabular model
            df = pd.DataFrame([property_data])
            dl = self.tabular_learner.dls.test_dl(df)
            features = self.tabular_learner.model.predict(dl)
            return features.detach().numpy().flatten()

        except Exception as e:
            logger.error(f"Error extracting tabular features: {e}")
            return np.array([0.0] * 10)


class PyTorchPricePredictor:
    """PyTorch-based property price prediction model"""

    def __init__(self, model_path: str = 'models/price_predictor.pt'):
        self.model_path = model_path
        self.model = PropertyPricePredictor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()

    def _load_model(self):
        """Load pre-trained PyTorch model"""
        try:
            if Path(self.model_path).exists():
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                logger.info("PyTorch price predictor loaded successfully")
            else:
                logger.warning("Price predictor model not found, using random initialization")
        except Exception as e:
            logger.error(f"Error loading PyTorch model: {e}")

    async def predict_price(self, features: np.ndarray) -> float:
        """Predict property price from features"""
        try:
            self.model.eval()
            with torch.no_grad():
                features_tensor = torch.tensor(features, dtype=torch.float32)
                if len(features_tensor.shape) == 1:
                    features_tensor = features_tensor.unsqueeze(0)

                prediction = self.model(features_tensor)
                return float(prediction.item())

        except Exception as e:
            logger.error(f"Error predicting price: {e}")
            return 0.0

    async def batch_predict(self, features_batch: List[np.ndarray]) -> List[float]:
        """Batch prediction for multiple properties"""
        try:
            self.model.eval()
            features_tensor = torch.tensor(np.array(features_batch), dtype=torch.float32)

            with torch.no_grad():
                predictions = self.model(features_tensor)
                return [float(p.item()) for p in predictions]

        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            return [0.0] * len(features_batch)


class DeepSeekAnalyzer:
    """DeepSeek v3.1 integration for market reasoning and analysis"""

    def __init__(self):
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        self.base_url = "https://api.deepseek.com/v1"
        self.client = httpx.AsyncClient()

    async def analyze_undervaluation(
        self, 
        property_data: Dict,
        market_data: CityMarketData,
        predicted_price: float,
        market_price: float
    ) -> Dict[str, Any]:
        """Use DeepSeek R1 reasoning for undervaluation analysis"""

        prompt = f"""
        Analyze this property for undervaluation using deep reasoning:

        Property Details:
        - Location: {property_data.get('city', 'Unknown')}, {property_data.get('address', 'N/A')}
        - Current Market Price: €{market_price:,.2f}
        - AI Predicted Fair Value: €{predicted_price:,.2f}
        - Property Age: {property_data.get('property_age', 'Unknown')} years
        - Size: {property_data.get('size_sqm', 'Unknown')} m²

        Market Context:
        - City Average Price/m²: €{market_data.average_price_per_sqm:,.2f}
        - 6-month Price Trend: {market_data.price_trend_6m:+.1%}
        - Demand/Supply Ratio: {market_data.demand_supply_ratio:.2f}
        - Market Sentiment: {market_data.sentiment_score:.2f}

        Provide detailed analysis:
        1. Is this property undervalued? By what specific percentage?
        2. What factors indicate this undervaluation?
        3. Risk assessment (legal, market, liquidity)
        4. Investment recommendation with reasoning
        5. Confidence score (0-100)

        Be specific with percentages and reasoning. Consider all market dynamics.
        """

        try:
            response = await self._make_deepseek_request(prompt)
            return self._parse_undervaluation_response(response, market_price, predicted_price)

        except Exception as e:
            logger.error(f"DeepSeek analysis error: {e}")
            return self._fallback_analysis(market_price, predicted_price)

    async def _make_deepseek_request(self, prompt: str) -> str:
        """Make API request to DeepSeek"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": "deepseek-reasoner",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            }

            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30.0
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"DeepSeek API error: {response.status_code}")
                return ""

        except Exception as e:
            logger.error(f"DeepSeek request error: {e}")
            return ""

    def _parse_undervaluation_response(
        self, 
        response: str, 
        market_price: float, 
        predicted_price: float
    ) -> Dict[str, Any]:
        """Parse DeepSeek response into structured data"""

        # Simple percentage calculation as fallback
        price_difference = ((predicted_price - market_price) / market_price) * 100

        return {
            "is_undervalued": price_difference > 5,
            "undervaluation_percentage": max(0, price_difference),
            "analysis": response,
            "confidence_score": 75,  # Default confidence
            "risk_factors": ["Market volatility", "Liquidity risk"],
            "recommendation": "BUY" if price_difference > 10 else "HOLD"
        }

    def _fallback_analysis(self, market_price: float, predicted_price: float) -> Dict[str, Any]:
        """Fallback analysis when DeepSeek is unavailable"""
        price_difference = ((predicted_price - market_price) / market_price) * 100

        return {
            "is_undervalued": price_difference > 5,
            "undervaluation_percentage": max(0, price_difference),
            "analysis": f"Based on AI price prediction, this property appears {price_difference:+.1f}% relative to market price.",
            "confidence_score": 60,
            "risk_factors": ["Limited analysis due to API unavailability"],
            "recommendation": "REVIEW" if abs(price_difference) < 5 else ("BUY" if price_difference > 0 else "PASS")
        }


class OpenAIAgentOrchestrator:
    """OpenAI Agents SDK integration for property analysis workflow"""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    async def create_property_analysis_agent(self) -> Dict:
        """Create specialized agent for property analysis"""

        agent_config = {
            "name": "PropertyAnalysisAgent",
            "instructions": """
            You are an expert real estate analysis agent for the Team-Realized platform.

            Your role:
            1. Analyze property opportunities for young European renters (22-35)
            2. Focus on micro-ownership investments (€100-1000 pieces)
            3. Identify undervalued properties in 50+ European cities
            4. Provide clear investment recommendations

            Key capabilities:
            - Process property data from multiple cities
            - Combine AI price predictions with market analysis
            - Assess legal and market risks
            - Generate user-friendly investment summaries

            Always provide:
            - Specific undervaluation percentages
            - Clear reasoning for recommendations
            - Risk assessment summary
            - Investment attractiveness score (0-100)
            """,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "analyze_property_opportunity",
                        "description": "Analyze a property for investment opportunity",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "property_id": {"type": "string"},
                                "analysis_depth": {"type": "string", "enum": ["quick", "detailed", "comprehensive"]}
                            }
                        }
                    }
                }
            ]
        }

        return agent_config

    async def run_property_analysis(self, property_data: Dict, market_context: Dict) -> Dict:
        """Run property analysis through OpenAI agent"""
        try:
            messages = [
                {
                    "role": "user", 
                    "content": f"Analyze this property opportunity: {json.dumps(property_data, indent=2)}\n\nMarket Context: {json.dumps(market_context, indent=2)}"
                }
            ]

            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )

            return {
                "analysis": response.choices[0].message.content,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"OpenAI agent error: {e}")
            return {
                "analysis": "Analysis temporarily unavailable",
                "status": "error"
            }


class CityMarketAnalyzer:
    """Main orchestrator for 50+ cities analysis"""

    def __init__(self):
        self.feature_extractor = FastAIPropertyFeatureExtractor()
        self.price_predictor = PyTorchPricePredictor()
        self.deepseek_analyzer = DeepSeekAnalyzer()
        self.openai_agent = OpenAIAgentOrchestrator()

        # List of 50+ European cities for analysis
        self.target_cities = [
            'Krakow', 'Berlin', 'Prague', 'Barcelona', 'Warsaw', 'Vienna',
            'Budapest', 'Amsterdam', 'Munich', 'Hamburg', 'Cologne', 'Dresden',
            'Leipzig', 'Valencia', 'Seville', 'Lisbon', 'Porto', 'Milan', 'Rome',
            'Naples', 'Turin', 'Bologna', 'Florence', 'Brussels', 'Antwerp',
            'Stockholm', 'Gothenburg', 'Copenhagen', 'Aarhus', 'Helsinki',
            'Tallinn', 'Riga', 'Vilnius', 'Ljubljana', 'Zagreb', 'Bratislava',
            'Sofia', 'Bucharest', 'Athens', 'Thessaloniki', 'Dublin', 'Cork',
            'Oslo', 'Bergen', 'Zurich', 'Geneva', 'Basel', 'Lyon', 'Marseille',
            'Toulouse', 'Nice', 'Bordeaux'
        ]

    async def scan_cities_for_opportunities(
        self, 
        user_budget: float = 500.0,
        max_investment: float = 1000.0
    ) -> List[Opportunity]:
        """Scan 50+ cities for property investment opportunities"""

        opportunities = []

        # This would normally fetch real data from property APIs
        # For now, we'll simulate the analysis pipeline

        logger.info(f"Scanning {len(self.target_cities)} cities for opportunities...")

        for city in self.target_cities[:5]:  # Limit for demo
            try:
                city_opportunities = await self._analyze_city(city, user_budget, max_investment)
                opportunities.extend(city_opportunities)

            except Exception as e:
                logger.error(f"Error analyzing {city}: {e}")
                continue

        # Sort by investment attractiveness
        opportunities.sort(key=lambda x: x.attractiveness_score, reverse=True)

        return opportunities[:10]  # Return top 10 opportunities

    async def _analyze_city(
        self, 
        city: str, 
        user_budget: float,
        max_investment: float
    ) -> List[Opportunity]:
        """Analyze opportunities in a specific city"""

        opportunities = []

        # Simulate property data (would come from real estate APIs)
        sample_properties = [
            {
                'id': f'{city}_prop_1',
                'city': city,
                'address': f'Sample Street 1, {city}',
                'price': 150000,
                'size_sqm': 65,
                'property_age': 15,
                'rooms': 3,
                'image_path': 'path/to/property1.jpg'
            },
            {
                'id': f'{city}_prop_2', 
                'city': city,
                'address': f'Another Street 2, {city}',
                'price': 200000,
                'size_sqm': 80,
                'property_age': 8,
                'rooms': 4,
                'image_path': 'path/to/property2.jpg'
            }
        ]

        for prop_data in sample_properties:
            try:
                opportunity = await self._analyze_single_property(prop_data, user_budget, max_investment)
                if opportunity and opportunity.is_affordable:
                    opportunities.append(opportunity)

            except Exception as e:
                logger.error(f"Error analyzing property {prop_data['id']}: {e}")
                continue

        return opportunities

    async def _analyze_single_property(
        self,
        property_data: Dict,
        user_budget: float,
        max_investment: float
    ) -> Optional[Opportunity]:
        """Analyze a single property for investment opportunity"""

        try:
            # 1. Extract features using Fast.ai
            tabular_features = await self.feature_extractor.extract_tabular_features(property_data)

            # 2. Predict fair price using PyTorch
            predicted_price = await self.price_predictor.predict_price(tabular_features)

            # 3. Create mock market data
            market_data = CityMarketData(
                city_name=property_data['city'],
                average_price_per_sqm=2500.0,
                price_trend_6m=0.05,
                inventory_levels=150,
                demand_supply_ratio=1.2,
                economic_indicators={'unemployment': 3.5, 'gdp_growth': 2.1},
                sentiment_score=0.7
            )

            # 4. DeepSeek analysis for undervaluation reasoning
            undervaluation_analysis = await self.deepseek_analyzer.analyze_undervaluation(
                property_data, market_data, predicted_price, property_data['price']
            )

            # 5. Calculate investment metrics
            min_investment = 100  # €100 minimum piece
            max_affordable_investment = min(max_investment, user_budget)

            # Create Opportunity entity
            opportunity = Opportunity(
                property_id=property_data['id'],
                city=property_data['city'],
                address=property_data['address'],
                total_property_value=property_data['price'],
                predicted_fair_value=predicted_price,
                undervaluation_percentage=undervaluation_analysis.get('undervaluation_percentage', 0),
                min_investment_amount=min_investment,
                max_investment_amount=max_affordable_investment,
                is_undervalued=undervaluation_analysis.get('is_undervalued', False),
                is_affordable=min_investment <= max_affordable_investment,
                attractiveness_score=self._calculate_attractiveness_score(undervaluation_analysis),
                analysis_summary=undervaluation_analysis.get('analysis', ''),
                risk_factors=undervaluation_analysis.get('risk_factors', []),
                recommendation=undervaluation_analysis.get('recommendation', 'HOLD')
            )

            return opportunity

        except Exception as e:
            logger.error(f"Error in single property analysis: {e}")
            return None

    def _calculate_attractiveness_score(self, analysis: Dict) -> float:
        """Calculate overall attractiveness score (0-100)"""

        base_score = 50.0

        # Undervaluation bonus
        undervaluation = analysis.get('undervaluation_percentage', 0)
        if undervaluation > 15:
            base_score += 30
        elif undervaluation > 10:
            base_score += 20
        elif undervaluation > 5:
            base_score += 10

        # Confidence bonus
        confidence = analysis.get('confidence_score', 50)
        base_score += (confidence - 50) * 0.3

        # Risk penalty
        risk_count = len(analysis.get('risk_factors', []))
        base_score -= risk_count * 5

        return max(0, min(100, base_score))


# Main integration class
class AIModelsService:
    """Main service for AI models integration"""

    def __init__(self):
        self.city_analyzer = CityMarketAnalyzer()
        logger.info("AI Models Service initialized")

    async def find_affordable_opportunities(
        self, 
        user_budget: float = 500.0,
        user_location: str = "Europe"
    ) -> List[Opportunity]:
        """Find affordable property opportunities using AI analysis"""

        logger.info(f"Finding opportunities for budget: €{user_budget}")

        try:
            opportunities = await self.city_analyzer.scan_cities_for_opportunities(
                user_budget=user_budget,
                max_investment=min(user_budget, 1000.0)
            )

            logger.info(f"Found {len(opportunities)} opportunities")
            return opportunities

        except Exception as e:
            logger.error(f"Error finding opportunities: {e}")
            return []

    async def analyze_specific_property(
        self,
        property_id: str,
        property_data: Dict
    ) -> Optional[Opportunity]:
        """Analyze a specific property for investment potential"""

        try:
            opportunity = await self.city_analyzer._analyze_single_property(
                property_data, 
                user_budget=500.0, 
                max_investment=1000.0
            )
            return opportunity

        except Exception as e:
            logger.error(f"Error analyzing property {property_id}: {e}")
            return None

    async def get_city_market_summary(self, city: str) -> Dict:
        """Get market summary for a specific city"""

        # This would integrate with real market data APIs
        return {
            "city": city,
            "average_price_per_sqm": 2500.0,
            "price_trend_6m": 5.2,
            "inventory_count": 150,
            "market_sentiment": "Positive",
            "investment_attractiveness": 7.5
        }


# Export main service
__all__ = ['AIModelsService', 'CityMarketAnalyzer', 'PropertyFeatures']