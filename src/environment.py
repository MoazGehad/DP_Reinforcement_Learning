"""
TaxiEnvironment - Wrapper for the Gymnasium Taxi-v3 environment.

This class serves as the sole point of contact with the simulator.
No planning logic should appear inside this class.
"""

import gymnasium as gym
import numpy as np


class TaxiEnvironment:
    """Wraps the Gymnasium Taxi-v3 environment.

    Attributes:
        env: The underlying Gymnasium environment instance.
        n_states (int): Total number of states (500).
        n_actions (int): Total number of actions (6).
        P (dict): The transition model - P[s][a] returns a list of
                  (probability, next_state, reward, done) tuples.
    """

    def __init__(self):
        # TODO: Instantiate the Taxi-v3 environment
        # TODO: Extract n_states, n_actions, and P
        pass

    def reset(self):
        """Reset the environment and return the initial state."""
        # TODO: Implement
        pass

    def step(self, action):
        """Take an action and return (next_state, reward, terminated, truncated, info)."""
        # TODO: Implement
        pass

    def render(self):
        """Render the current state of the environment."""
        # TODO: Implement
        pass
