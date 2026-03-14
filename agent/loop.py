from typing import Dict
from agent.agent import MarketingAgent


class AgentLoop:
    """
    Orchestrates execution over time.
    In this version, the loop does NOT evaluate goals or decide stance.
    It simply runs the agent on a provided state snapshot.
    """

    def __init__(self, agent: MarketingAgent):
        self.agent = agent

    def run(self, state: Dict) -> Dict:
        decision = self.agent.evaluate(state)
        return decision.to_dict()
