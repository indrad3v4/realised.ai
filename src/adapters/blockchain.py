"""
Blockchain adapter for Team-Realized platform
Solana integration for instant property ownership certificates
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import json
import hashlib
import base58

from ..core.entities import OwnershipPiece, Transaction
from ..core.config import settings

logger = logging.getLogger(__name__)


class SolanaClient:
    """Mock Solana client for hackathon"""

    def __init__(self, rpc_url: str):
        self.rpc_url = rpc_url
        self.is_connected = False

    async def connect(self):
        """Connect to Solana network"""
        try:
            # Mock connection - would use real Solana client
            self.is_connected = True
            logger.info(f"Connected to Solana network: {self.rpc_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Solana: {e}")
            self.is_connected = False

    async def create_spl_token(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Create SPL token for property ownership"""
        if not self.is_connected:
            await self.connect()

        try:
            # Mock SPL token creation
            token_data = json.dumps(metadata, sort_keys=True)
            token_hash = hashlib.sha256(token_data.encode()).hexdigest()[:32]
            token_address = base58.b58encode(token_hash.encode()).decode()[:44]

            logger.info(f"Created SPL token: {token_address}")
            return token_address

        except Exception as e:
            logger.error(f"Failed to create SPL token: {e}")
            return None

    async def mint_token(self, token_address: str, owner_pubkey: str, amount: int) -> Optional[str]:
        """Mint tokens to owner"""
        try:
            # Mock token minting
            mint_data = f"{token_address}_{owner_pubkey}_{amount}_{datetime.utcnow().timestamp()}"
            mint_hash = hashlib.sha256(mint_data.encode()).hexdigest()

            logger.info(f"Minted {amount} tokens to {owner_pubkey[:8]}...")
            return mint_hash

        except Exception as e:
            logger.error(f"Failed to mint tokens: {e}")
            return None


class CertificateResult:
    """Result of certificate issuance"""

    def __init__(self, success: bool, token_id: str = None, blockchain_hash: str = None, error: str = None):
        self.success = success
        self.token_id = token_id
        self.blockchain_hash = blockchain_hash
        self.error = error


class BlockchainService:
    """Blockchain service for property certificates"""

    def __init__(self):
        self.solana_client = SolanaClient(settings.solana_rpc_url)
        self.program_id = "TEAM_REALIZED_PROGRAM_ID"  # Mock program ID
        self._initialized = False

    async def initialize(self):
        """Initialize blockchain service"""
        if self._initialized:
            return

        try:
            await self.solana_client.connect()
            self._initialized = True
            logger.info("Blockchain service initialized")

        except Exception as e:
            logger.error(f"Failed to initialize blockchain service: {e}")
            # Don't raise - continue with mock functionality

    async def issue_certificate(
        self,
        user_id: str,
        property_id: str,
        ownership_percentage: float,
        investment_amount: float
    ) -> CertificateResult:
        """Issue ownership certificate as Solana SPL token"""

        try:
            logger.info(f"Issuing certificate for {user_id}: {ownership_percentage}% of {property_id}")

            # Create certificate metadata
            certificate_metadata = {
                "user_id": user_id,
                "property_id": property_id,
                "ownership_percentage": ownership_percentage,
                "investment_amount": investment_amount,
                "issued_at": datetime.utcnow().isoformat(),
                "issuer": "Team-Realized",
                "version": "1.0"
            }

            # Create SPL token for certificate
            if self.solana_client.is_connected:
                token_address = await self.solana_client.create_spl_token(certificate_metadata)

                if not token_address:
                    return CertificateResult(
                        success=False,
                        error="Failed to create SPL token"
                    )

                # Mock user public key (would be provided by user's wallet)
                user_pubkey = f"USER_{user_id[:8].upper()}_{hashlib.md5(user_id.encode()).hexdigest()[:16].upper()}"

                # Mint 1 certificate token to user
                mint_hash = await self.solana_client.mint_token(
                    token_address=token_address,
                    owner_pubkey=user_pubkey,
                    amount=1
                )

                if not mint_hash:
                    return CertificateResult(
                        success=False,
                        error="Failed to mint certificate token"
                    )

                certificate_id = f"CERT_{property_id[:8].upper()}_{user_id[:8].upper()}_{int(datetime.utcnow().timestamp())}"

                return CertificateResult(
                    success=True,
                    token_id=certificate_id,
                    blockchain_hash=mint_hash
                )

            else:
                # Fallback mock certificate for demo
                certificate_id = f"MOCK_CERT_{property_id[:8].upper()}_{user_id[:8].upper()}_{int(datetime.utcnow().timestamp())}"
                mock_hash = hashlib.sha256(f"{certificate_id}_{datetime.utcnow().timestamp()}".encode()).hexdigest()

                logger.info(f"Issued mock certificate: {certificate_id}")

                return CertificateResult(
                    success=True,
                    token_id=certificate_id,
                    blockchain_hash=mock_hash
                )

        except Exception as e:
            logger.error(f"Failed to issue certificate: {e}")
            return CertificateResult(
                success=False,
                error=str(e)
            )

    async def verify_certificate(self, certificate_token: str) -> Dict[str, Any]:
        """Verify ownership certificate on blockchain"""

        try:
            logger.info(f"Verifying certificate: {certificate_token}")

            # Mock verification (would query Solana blockchain)
            if certificate_token.startswith("CERT_") or certificate_token.startswith("MOCK_CERT_"):
                return {
                    "valid": True,
                    "certificate_id": certificate_token,
                    "verified_at": datetime.utcnow().isoformat(),
                    "blockchain_network": "SOLANA",
                    "status": "ACTIVE"
                }
            else:
                return {
                    "valid": False,
                    "error": "Invalid certificate format"
                }

        except Exception as e:
            logger.error(f"Failed to verify certificate: {e}")
            return {
                "valid": False,
                "error": str(e)
            }

    async def get_user_certificates(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all certificates owned by user"""

        try:
            logger.info(f"Getting certificates for user: {user_id}")

            # Mock certificates list (would query Solana blockchain)
            mock_certificates = [
                {
                    "certificate_id": f"CERT_KRAKOW01_{user_id[:8].upper()}_1234567890",
                    "property_id": "krakow_prop_1",
                    "ownership_percentage": 0.05,
                    "investment_amount": 100.0,
                    "issued_at": "2025-09-26T20:00:00Z",
                    "status": "ACTIVE"
                }
            ]

            return mock_certificates

        except Exception as e:
            logger.error(f"Failed to get user certificates: {e}")
            return []

    async def transfer_certificate(
        self,
        certificate_token: str,
        from_user_id: str,
        to_user_id: str
    ) -> bool:
        """Transfer certificate between users"""

        try:
            logger.info(f"Transferring certificate {certificate_token} from {from_user_id} to {to_user_id}")

            # Mock transfer (would execute Solana token transfer)
            # In a real implementation, this would:
            # 1. Verify ownership
            # 2. Execute SPL token transfer
            # 3. Update certificate metadata

            logger.info("Certificate transfer successful (mock)")
            return True

        except Exception as e:
            logger.error(f"Failed to transfer certificate: {e}")
            return False

    async def get_certificate_value(self, certificate_token: str) -> Optional[float]:
        """Get current market value of certificate"""

        try:
            # Mock valuation (would use oracle or market data)
            base_value = 100.0
            growth_factor = 1.05  # 5% growth

            current_value = base_value * growth_factor

            logger.info(f"Certificate {certificate_token} valued at {current_value} EUR")
            return current_value

        except Exception as e:
            logger.error(f"Failed to get certificate value: {e}")
            return None


# Payment service integration
class PaymentResult:
    """Result of payment processing"""

    def __init__(self, success: bool, transaction_id: str = None, error: str = None):
        self.success = success
        self.transaction_id = transaction_id
        self.error = error


class PaymentService:
    """Mock payment service for hackathon"""

    def __init__(self):
        self.processing_fee = 0.03  # 3%

    async def process_payment(
        self,
        amount: float,
        user_id: str,
        reference: str
    ) -> PaymentResult:
        """Process payment (mock implementation)"""

        try:
            logger.info(f"Processing payment: {amount} EUR for user {user_id}")

            # Mock payment processing
            # In real implementation, would integrate with:
            # - Stripe for credit cards
            # - Crypto payment processors
            # - Bank transfers

            # Simulate processing delay
            await asyncio.sleep(0.1)

            # Mock success (99% success rate)
            import random
            if random.random() > 0.01:
                payment_id = f"PAY_{int(datetime.utcnow().timestamp())}_{user_id[:8]}"
                logger.info(f"Payment successful: {payment_id}")

                return PaymentResult(
                    success=True,
                    transaction_id=payment_id
                )
            else:
                return PaymentResult(
                    success=False,
                    error="Payment declined by bank"
                )

        except Exception as e:
            logger.error(f"Payment processing failed: {e}")
            return PaymentResult(
                success=False,
                error=str(e)
            )


# Global service instances
blockchain_service = BlockchainService()
payment_service = PaymentService()

# Export services
__all__ = [
    'BlockchainService', 'PaymentService', 'CertificateResult', 'PaymentResult',
    'blockchain_service', 'payment_service'
]
