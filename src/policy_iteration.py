"""
PolicyIterationAgent - Implements the Policy Iteration algorithm.

Alternates between Policy Evaluation (Bellman expectation equation)
and Policy Improvement (greedy update) until the policy stabilises.
"""

import numpy as np


class PolicyIterationAgent:
    """Policy Iteration agent for the Taxi-v3 environment.

    Args:
        env: An environment instance (TaxiEnvironment or compatible).
        gamma (float): Discount factor.
        theta (float): Convergence threshold for policy evaluation.

    Attributes:
        value_function (np.ndarray): Array of size 500.
        policy (np.ndarray): Array of size 500.
        n_policy_changes (int): Total number of policy updates across all iterations.
    """

    def __init__(self, env, gamma=0.99, theta=1e-8):
        # TODO: Store env, gamma, theta
        # TODO: Initialise value_function, policy, n_policy_changes
        pass

    def policy_evaluation(self):
        """Iteratively evaluate the current policy.

        Applies the Bellman expectation equation until the value function
        converges under the current policy.
        """
        # TODO: Implement iterative policy evaluation
        pass

    def policy_improvement(self):
        """Greedily update the policy w.r.t. the current value function.

        Returns:
            bool: True if the policy is stable (no changes), False otherwise.
        """
        # TODO: Implement greedy policy improvement
        # TODO: Track n_policy_changes
        pass

    def policy_iteration(self):
        """Execute the full Policy Iteration algorithm.

        Alternates policy_evaluation() and policy_improvement()
        until the policy stabilises.
        """
        # TODO: Implement the main PI loop
        pass
