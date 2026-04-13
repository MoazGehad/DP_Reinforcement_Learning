"""
Planner - Orchestrator for running and comparing DP algorithms.

Acts as a high-level interface to run Value Iteration and Policy Iteration,
and produce comparison results.
"""

import numpy as np
from src.environment import TaxiEnvironment
from src.value_iteration import ValueIterationAgent
from src.policy_iteration import PolicyIterationAgent


class Planner:
    """Orchestrator for running and comparing planning algorithms.

    Args:
        env: An environment instance (TaxiEnvironment or compatible).
    """

    def __init__(self, env):
        # TODO: Store environment reference
        pass

    def run_value_iteration(self, gamma=0.99, theta=1e-8):
        """Create and run a ValueIterationAgent.

        Args:
            gamma (float): Discount factor.
            theta (float): Convergence threshold.

        Returns:
            ValueIterationAgent: The trained agent.
        """
        # TODO: Implement
        pass

    def run_policy_iteration(self, gamma=0.99, theta=1e-8):
        """Create and run a PolicyIterationAgent.

        Args:
            gamma (float): Discount factor.
            theta (float): Convergence threshold.

        Returns:
            PolicyIterationAgent: The trained agent.
        """
        # TODO: Implement
        pass

    def compare_agents(self, gamma_values=None, theta=1e-8):
        """Run both algorithms across multiple gamma values and return a summary.

        Args:
            gamma_values (list): List of discount factors to test.
            theta (float): Convergence threshold.

        Returns:
            dict: Comparison results including iterations, policy similarity, etc.
        """
        # TODO: Implement comparison logic
        # TODO: Return structured results for the comparison table
        pass
