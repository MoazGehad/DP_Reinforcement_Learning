"""
TransitionModel - Learned approximation of the environment's transition dynamics.

Maintains a model that can be incrementally updated from (s, a, r, s') experience
and exposes P[s][a] in the same format as Gymnasium for seamless integration.
"""

import numpy as np


class TransitionModel:
    """Learned transition model for the Taxi-v3 environment.

    Supports 500 states and 6 actions. Exposes P[s][a] returning a list
    of (probability, next_state, reward, done) tuples, consistent with
    the Gymnasium format.

    Attributes:
        n_states (int): Number of states (500).
        n_actions (int): Number of actions (6).
        P (dict): The learned transition model.
    """

    def __init__(self, n_states=500, n_actions=6):
        # TODO: Initialise counts and model storage
        # TODO: Set up P dictionary structure
        pass

    def update(self, state, action, reward, next_state):
        """Update the model with a new observed transition.

        Args:
            state (int): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (int): Resulting next state.
        """
        # TODO: Increment transition counts
        # TODO: Update estimated probabilities and rewards
        pass

    def learn_from_interaction(self, env, n_episodes=1000, max_steps=200):
        """Interact with the environment to learn the transition model.

        Args:
            env: Environment instance to interact with.
            n_episodes (int): Number of episodes for data collection.
            max_steps (int): Max steps per episode.
        """
        # TODO: Implement online learning loop
        # TODO: Use random or exploratory policy to collect transitions
        pass

    def get_accuracy(self, true_P):
        """Compare learned model against the true transition model.

        Args:
            true_P (dict): The true transition model from the environment.

        Returns:
            dict: Accuracy metrics (probability error, reward error, etc.)
        """
        # TODO: Implement comparison metrics
        pass
