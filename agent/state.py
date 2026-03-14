from dataclasses import dataclass


@dataclass
class CampaignState:
    """
    Represents a normalized snapshot of a marketing campaign's state.
    """

    campaign_id: str

    # Performance metrics
    CPA: float
    ROAS: float

    # Targets
    target_CPA: float

    # Trends (percentage change, e.g. +0.12 = +12%)
    CPA_trend_7d: float
    ROAS_trend_7d: float

    # Context
    days_active: int

    def to_dict(self) -> dict:
        """
        Convert state to dictionary format for goal evaluation.
        """
        return {
            "campaign_id": self.campaign_id,
            "CPA": self.CPA,
            "ROAS": self.ROAS,
            "target_CPA": self.target_CPA,
            "CPA_trend_7d": self.CPA_trend_7d,
            "ROAS_trend_7d": self.ROAS_trend_7d,
            "days_active": self.days_active,
        }
