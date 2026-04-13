"""
ValueIterationAgent — Implements the Value Iteration algorithm.

Uses the Bellman optimality update, iterating until the maximum change
across all states falls below the convergence threshold theta.
"""

import numpy as np


class ValueIterationAgent:
    """Value Iteration agent for the Taxi-v3 environment.

    Args:
        env: An environment instance (TaxiEnvironment or compatible).
        gamma (float): Discount factor.
        theta (float): Convergence threshold.

    Attributes:
        value_function (np.ndarray): Array of size 500 — state values.
        policy (np.ndarray): Array of size 500 — optimal actions.
        convergence_history (list): Max-delta values per iteration.
    """

    def __init__(self, env, gamma=0.99, theta=1e-8):
        # TODO: Store env, gamma, theta
        # TODO: Initialise value_function, policy, convergence_history
        pass

    def value_iteration(self):
        """Execute the full Value Iteration algorithm.

        Iterates Bellman optimality update until convergence.
        Stores:
            - self.value_function
            - self.policy (greedy w.r.t. converged value function)
            - self.convergence_history (max delta per iteration)
        """
        # TODO: Implement Bellman optimality update loop
        # TODO: Extract greedy policy after convergence
        pass
