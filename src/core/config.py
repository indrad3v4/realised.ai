"""
Configuration for Team-Realized platform
Environment variables and settings management
"""

import os
from typing import List
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings"""

    # Application
    app_name: str = "Team-Realized"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")

    # Server
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")

    # Database
    database_url: str = Field(default="sqlite:///./team_realized.db", env="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")

    # AI Services
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    deepseek_api_key: str = Field(default="", env="DEEPSEEK_API_KEY")

    # Blockchain
    solana_rpc_url: str = Field(
        default="https://api.mainnet-beta.solana.com", 
        env="SOLANA_RPC_URL"
    )
    solana_private_key: str = Field(default="", env="SOLANA_PRIVATE_KEY")

    # Real Estate APIs
    property_api_key: str = Field(default="", env="PROPERTY_API_KEY")
    google_maps_api_key: str = Field(default="", env="GOOGLE_MAPS_API_KEY")

    # Business Logic
    min_investment_amount: float = 100.0
    max_investment_amount: float = 1000.0
    transaction_fee_percentage: float = 3.0  # 3%

    # Target cities for property scanning
    target_cities: List[str] = [
        "Krakow", "Berlin", "Prague", "Barcelona", "Warsaw", "Vienna",
        "Budapest", "Amsterdam", "Munich", "Hamburg", "Cologne", "Dresden",
        "Leipzig", "Valencia", "Seville", "Lisbon", "Porto", "Milan", "Rome",
        "Naples", "Turin", "Bologna", "Florence", "Brussels", "Antwerp",
        "Stockholm", "Gothenburg", "Copenhagen", "Aarhus", "Helsinki",
        "Tallinn", "Riga", "Vilnius", "Ljubljana", "Zagreb", "Bratislava",
        "Sofia", "Bucharest", "Athens", "Thessaloniki", "Dublin", "Cork",
        "Oslo", "Bergen", "Zurich", "Geneva", "Basel", "Lyon", "Marseille",
        "Toulouse", "Nice", "Bordeaux"
    ]

    # AI Model Configuration
    ai_models_enabled: bool = True
    ai_analysis_timeout: int = 30  # seconds

    # Rate Limiting
    api_rate_limit: str = "100/minute"

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = "team_realized.log"

    # Security
    secret_key: str = Field(
        default="team-realized-secret-key-change-in-production",
        env="SECRET_KEY"
    )
    cors_origins: List[str] = ["*"]  # Configure for production

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


# Configuration helpers
def is_development() -> bool:
    """Check if running in development mode"""
    return settings.environment.lower() in ("development", "dev")


def is_production() -> bool:
    """Check if running in production mode"""
    return settings.environment.lower() in ("production", "prod")


def get_database_url() -> str:
    """Get database URL with fallback"""
    if settings.database_url and settings.database_url != "sqlite:///./team_realized.db":
        return settings.database_url

    # Development fallback
    return "sqlite:///./team_realized.db"


def get_ai_config() -> dict:
    """Get AI service configuration"""
    return {
        "openai_api_key": settings.openai_api_key,
        "deepseek_api_key": settings.deepseek_api_key,
        "enabled": settings.ai_models_enabled and bool(settings.openai_api_key),
        "timeout": settings.ai_analysis_timeout
    }


def get_blockchain_config() -> dict:
    """Get blockchain configuration"""
    return {
        "solana_rpc_url": settings.solana_rpc_url,
        "private_key": settings.solana_private_key,
        "enabled": bool(settings.solana_private_key)
    }


# Export settings
__all__ = ["settings", "Settings", "is_development", "is_production", 
           "get_database_url", "get_ai_config", "get_blockchain_config"]
