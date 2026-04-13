# Reinforcement Learning - Dynamic Programming Assignment

**AI424 - Reinforcement Learning**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MoazGehad/DP_Reinforcement_Learning/blob/main/Assignment1_DP_RL.ipynb)

## Overview

Implementation and comparison of Value Iteration and Policy Iteration on the Gymnasium Taxi-v3 environment, extended to MountainCarContinuous-v0 through state-space discretisation.

## Team

| Name | ID | Section | Responsibility |
|---|---|---|---|
| Omar Ez-Eldin | 20220228 | Section 1 | OOP Implementation |
| Yussuf Ahmed | 20220385 | Section 2 | Algorithm Details (VI & PI) |
| Moaz Gehad | 20220340 | Sections 3 & 4 | Experiments & Conceptual Questions |
| Mahmoud Ehab | 20220457 | Section 5 | Transition Model Learning |
| Abdelrhman Ebrahim | 20220519 | Section 6 | MountainCar Extension |

## Project Structure

```
RL_Assignment/
├── README.md
├── Assignment1_DP_RL.ipynb         # Main notebook (submission)
├── Assignment1_DP_RL.pdf           # Assignment specification
├── requirements.txt
├── .gitignore
├── src/                            # Dev module skeletons
│   ├── environment.py
│   ├── value_iteration.py
│   ├── policy_iteration.py
│   ├── planner.py
│   ├── visualiser.py
│   ├── transition_model.py
│   └── mountain_car.py
└── docs/
    └── conceptual_answers.md
```

## Setup

### Google Colab (Primary)

1. Click the **Open in Colab** badge above (update `YOUR_USERNAME` with the actual GitHub username)
2. Or: Colab > File > Open Notebook > GitHub > paste the repo URL
3. The first cell installs dependencies automatically

### Local

```bash
git clone https://github.com/YOUR_USERNAME/RL_Assignment.git
cd RL_Assignment
pip install -r requirements.txt
jupyter notebook Assignment1_DP_RL.ipynb
```

## Workflow

1. Each member works on their assigned section in the notebook
2. Use feature branches (`feature/section-1`, `feature/section-2`, etc.)
3. Open Pull Requests to merge into `main`
4. Final review and submission as a single complete notebook

## Constraints

| Rule | Status |
|---|---|
| RL libraries (Stable-Baselines, RLlib, etc.) | Not permitted |
| NumPy and pure Python | Required |
| OOP design | Strictly required |
| Written explanation for every plot | Required |
| `plt.show()` outside `Visualiser` | Not permitted |
| `gymnasium` (env instantiation + `env.P` only) | Permitted |

## Timeline

| Phase | Dates | Milestone |
|---|---|---|
| Foundation | Apr 13-15 | TaxiEnvironment + algorithm skeletons ready |
| Core Implementation | Apr 15-17 | VI, PI, Visualiser, experiments complete |
| Extensions | Apr 17-19 | Transition model + MountainCar done |
| Final Integration | Apr 19-20 | Full notebook review, polish, submit |

**Deadline: April 20, 2026**

## Dependencies

- Python 3.10+
- NumPy >= 1.24
- Matplotlib >= 3.7
- Gymnasium >= 0.29
