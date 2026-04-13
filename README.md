# Reinforcement Learning — Dynamic Programming Assignment

> **Value Iteration & Policy Iteration on Gymnasium Taxi-v3**

A comprehensive implementation of Dynamic Programming methods for solving the Taxi-v3 environment, including learned transition models and extension to continuous state spaces (MountainCarContinuous-v0).

---

## Overview

This project implements and compares two fundamental Dynamic Programming algorithms:

| Algorithm | Approach |
|---|---|
| **Value Iteration** | Iterative Bellman optimality updates until convergence |
| **Policy Iteration** | Alternating policy evaluation and greedy improvement |

Both algorithms are applied to the [Gymnasium Taxi-v3](https://gymnasium.farama.org/environments/toy_text/taxi/) environment (500 discrete states, 6 actions) and extended to MountainCarContinuous-v0 through state-space discretisation.

## Project Structure

```
RL_Assignment/
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
├── requirements.txt                # Python dependencies
├── Assignment1_DP_RL.pdf           # Assignment specification
├── Assignment1_DP_RL.ipynb         # Main Jupyter notebook (submission)
├── src/
│   ├── __init__.py
│   ├── environment.py              # TaxiEnvironment wrapper
│   ├── value_iteration.py          # Value Iteration agent
│   ├── policy_iteration.py         # Policy Iteration agent
│   ├── planner.py                  # Orchestrator for running & comparing
│   ├── visualiser.py               # Centralised plotting
│   ├── transition_model.py         # Learned transition dynamics
│   └── mountain_car.py             # MountainCar discretisation
└── docs/
    └── conceptual_answers.md       # Conceptual questions (Section 4)
```

## Class Architecture

```
TaxiEnvironment
├── Wraps Gymnasium Taxi-v3
├── Exposes: n_states, n_actions, P
└── Methods: reset(), step(), render()

ValueIterationAgent
├── Bellman optimality update loop
└── Stores: value_function, policy, convergence_history

PolicyIterationAgent
├── policy_evaluation() + policy_improvement()
└── Stores: value_function, policy, n_policy_changes

Planner
├── run_value_iteration()
├── run_policy_iteration()
└── compare_agents()

Visualiser
├── plot_value_function()      → Heatmap
├── plot_policy_grid()         → Directional arrows + P/D markers
├── plot_convergence()         → Convergence curves
└── plot_gamma_comparison()    → VI vs PI bar chart

TransitionModel
├── Online learning from (s, a, r, s') tuples
└── P[s][a] compatible with existing agents
```

## Setup

### Local Environment

```bash
# Clone the repository
git clone <repo-url>
cd RL_Assignment

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

### Google Colab

```python
# Clone the repo in the first cell of your notebook
!git clone <repo-url>
%cd RL_Assignment
!pip install -r requirements.txt

# Add src to the Python path
import sys
sys.path.insert(0, '.')
```

## Usage

```python
from src.environment import TaxiEnvironment
from src.value_iteration import ValueIterationAgent
from src.policy_iteration import PolicyIterationAgent
from src.planner import Planner
from src.visualiser import Visualiser

# Setup
env = TaxiEnvironment()
planner = Planner(env)
vis = Visualiser()

# Run Value Iteration
vi_agent = planner.run_value_iteration(gamma=0.99, theta=1e-8)

# Run Policy Iteration
pi_agent = planner.run_policy_iteration(gamma=0.99, theta=1e-8)

# Compare agents across different gamma values
results = planner.compare_agents(gamma_values=[0.50, 0.90, 0.99])

# Visualise results
vis.plot_convergence([vi_agent.convergence_history], labels=["γ=0.99"])
vis.plot_value_function(vi_agent.value_function)
vis.plot_policy_grid(vi_agent.policy)
```

## Team Members

| Member | Section | Responsibility |
|---|---|---|
| Member 1 | Sections 1 + 2a | `TaxiEnvironment`, `ValueIterationAgent`, `Planner` |
| Member 2 | Sections 2b + 3 | `PolicyIterationAgent`, experiments & comparison table |
| Member 3 | Sections 1c + 3 | `Visualiser` class, all plots & written explanations |
| Member 4 | Section 4 | Conceptual questions (Q5–Q12) |
| Member 5 | Sections 5 + 6 | `TransitionModel`, MountainCar discretisation & analysis |

## Workflow

1. Each member works on a **feature branch** (`feature/<task-name>`)
2. Submit **Pull Requests** to `main` with at least one reviewer
3. Track progress via **GitHub Issues**
4. Final integration into the Jupyter notebook for submission

## Constraints

| Rule | Status |
|---|---|
| RL libraries (Stable-Baselines, RLlib, etc.) | ❌ Not permitted |
| NumPy and pure Python | ✅ Required |
| Object-Oriented Design (OOP) | ✅ Strictly required |
| Written explanation for every plot | ✅ Required |
| `plt.show()` outside `Visualiser` | ❌ Not permitted |
| `gymnasium` (env instantiation + `env.P` only) | ✅ Permitted |

## Dependencies

- Python 3.10+
- NumPy ≥ 1.24
- Matplotlib ≥ 3.7
- Gymnasium ≥ 0.29

## License

This project is for academic purposes only.
