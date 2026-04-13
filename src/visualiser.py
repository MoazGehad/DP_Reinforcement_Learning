"""
Visualiser — Centralised plotting for all assignment visualisations.

All plotting logic MUST be in this class.
No plt.show() calls are permitted outside this class.
"""

import numpy as np
import matplotlib.pyplot as plt


class Visualiser:
    """Handles all visualisation for the DP assignment.

    Every plot must be accompanied by a written explanation
    in the notebook (handled outside this class in markdown cells).
    """

    def __init__(self):
        # TODO: Set default plot styling (font sizes, colour palette, etc.)
        pass

    def plot_value_function(self, value_function, title="Value Function Heatmap"):
        """Plot a heatmap of the value function.

        Args:
            value_function (np.ndarray): Array of state values (size 500).
            title (str): Plot title.
        """
        # TODO: Implement heatmap visualisation
        # Hint: Reshape the 500-element array into a meaningful 2D grid
        pass

    def plot_policy_grid(self, policy, title="Policy Grid"):
        """Plot the policy as a grid with directional arrows.

        Must include pickup (P) and drop-off (D) markers.

        Args:
            policy (np.ndarray): Array of actions (size 500).
            title (str): Plot title.
        """
        # TODO: Implement policy grid with arrows
        # TODO: Add P (pickup) and D (drop-off) markers
        pass

    def plot_convergence(self, convergence_histories, labels, title="Convergence Curves"):
        """Plot convergence curves for Value Iteration.

        Args:
            convergence_histories (list of lists): Max-delta per iteration for each run.
            labels (list of str): Label for each curve (e.g., gamma values).
            title (str): Plot title.
        """
        # TODO: Implement convergence curve plot (one curve per gamma)
        pass

    def plot_gamma_comparison(self, results, title="VI vs PI Iterations"):
        """Plot a bar chart comparing iterations between VI and PI.

        Args:
            results (dict): Comparison results from Planner.compare_agents().
            title (str): Plot title.
        """
        # TODO: Implement grouped bar chart
        pass
