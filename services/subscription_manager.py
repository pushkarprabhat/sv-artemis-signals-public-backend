"""
Subscription Manager Service — Artemis Signals SaaS
Centralizes all plan, entitlement, and feature gating logic.
"""
from typing import List, Dict, Optional
from enum import Enum

# --- Plan and Feature Registry ---
class PlanId(str, Enum):
    SPARK = 'spark'
    FLAME = 'flame'
    INFERNO = 'inferno'
    PHOENIX = 'phoenix'

class Feature(str, Enum):
    PAIRS = 'PAIRS'
    COINTEGRATION = 'COINTEGRATION'
    GARCH = 'GARCH'
    IV_CRUSH = 'IV_CRUSH'
    API = 'API'
    ALL = 'ALL'

# Central registry: which features are allowed for each plan
PLAN_FEATURES: Dict[PlanId, List[Feature]] = {
    PlanId.SPARK: [Feature.PAIRS],
    PlanId.FLAME: [Feature.PAIRS, Feature.COINTEGRATION, Feature.GARCH],
    PlanId.INFERNO: [Feature.PAIRS, Feature.COINTEGRATION, Feature.GARCH, Feature.IV_CRUSH, Feature.API],
    PlanId.PHOENIX: [Feature.ALL],
}

# --- Subscription Manager Service ---
class SubscriptionManager:
    def __init__(self, user_id: str, plan_id: Optional[str]):
        self.user_id = user_id
        self.plan_id = PlanId(plan_id) if plan_id in PlanId.__members__.values() else None

    def get_allowed_features(self) -> List[Feature]:
        if not self.plan_id:
            return []
        return PLAN_FEATURES.get(self.plan_id, [])

    def can_access_feature(self, feature: Feature) -> bool:
        allowed = self.get_allowed_features()
        return Feature.ALL in allowed or feature in allowed

    def log_access_attempt(self, feature: Feature, result: bool):
        # TODO: Implement audit logging (DB or file)
        pass

    def sync_with_payment_provider(self):
        # TODO: Implement Razorpay sync/check
        pass

# Example usage:
# mgr = SubscriptionManager(user_id, plan_id)
# if mgr.can_access_feature(Feature.PAIRS): ...
# features = mgr.get_allowed_features()
