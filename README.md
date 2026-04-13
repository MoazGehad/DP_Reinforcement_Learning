# Reinforcement Learning — Dynamic Programming Assignment

> **AI424 — Reinforcement Learning**
> 
> Value Iteration & Policy Iteration on Gymnasium Taxi-v3

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/RL_Assignment/blob/main/Assignment1_DP_RL.ipynb)

---

## Overview

This project implements and compares two fundamental Dynamic Programming algorithms — **Value Iteration** and **Policy Iteration** — applied to the [Gymnasium Taxi-v3](https://gymnasium.farama.org/environments/toy_text/taxi/) environment. It also extends the approach to a continuous state space via the MountainCarContinuous-v0 environment.

## Team Members

| Name | ID | Section | Responsibility |
|---|---|---|---|
| **Omar Ez-Eldin** | 20220228 | Section 1 | OOP Implementation — `TaxiEnvironment`, `Planner`, `Visualiser` class skeletons |
| **Yussuf Ahmed** | 20220385 | Section 2 | Algorithm Details — `ValueIterationAgent`, `PolicyIterationAgent` |
| **Moaz Gehad** | 20220340 | Sections 3 & 4 | Experiments, comparison table, plots & Conceptual Questions |
| **Mahmoud Ehab** | 20220457 | Section 5 | Transition Model Learning & Verification |
| **Abdelrhman Ebrahim** | 20220519 | Section 6 | MountainCarContinuous-v0 Discretisation & Analysis |

## Project Structure

```
RL_Assignment/
├── README.md                       # This file
├── Assignment1_DP_RL.ipynb         # Main notebook (submission)
├── Assignment1_DP_RL.pdf           # Assignment specification
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
├── src/                            # Development module skeletons
│   ├── environment.py              # TaxiEnvironment
│   ├── value_iteration.py          # ValueIterationAgent
│   ├── policy_iteration.py         # PolicyIterationAgent
│   ├── planner.py                  # Planner orchestrator
│   ├── visualiser.py               # Visualiser (all plots)
│   ├── transition_model.py         # Learned transition model
│   └── mountain_car.py             # MountainCar discretisation
└── docs/
    └── conceptual_answers.md       # Conceptual questions draft
```

## How to Use

### Google Colab (Primary)

1. Click the **Open in Colab** badge above (update the URL with your GitHub username first)
2. Or open [Google Colab](https://colab.research.google.com) → File → Open Notebook → GitHub → paste the repo URL
3. The first cell handles all setup automatically

### Local Environment

```bash
git clone https://github.com/YOUR_USERNAME/RL_Assignment.git
cd RL_Assignment
pip install -r requirements.txt
jupyter notebook Assignment1_DP_RL.ipynb
```

## Workflow

1. Each member works on their assigned section in the notebook
2. Use **feature branches** for development (`feature/section-1`, `feature/section-2`, etc.)
3. Open **Pull Requests** to merge into `main`
4. Resolve conflicts in the notebook carefully (coordinate via team chat)
5. Final review and submission as a single complete notebook

## Constraints

| Rule | Status |
|---|---|
| RL libraries (Stable-Baselines, RLlib, etc.) | ❌ Not permitted |
| NumPy and pure Python | ✅ Required |
| Object-Oriented Design (OOP) | ✅ Strictly required |
| Written explanation for every plot | ✅ Required |
| `plt.show()` outside `Visualiser` | ❌ Not permitted |
| `gymnasium` (env instantiation + `env.P` only) | ✅ Permitted |

## Timeline

| Phase | Dates | Milestone |
|---|---|---|
| Foundation | Apr 13–15 | `TaxiEnvironment` + algorithm skeletons ready |
| Core Implementation | Apr 15–17 | VI, PI, Visualiser, experiments complete |
| Extensions | Apr 17–19 | Transition model + MountainCar done |
| Final Integration | Apr 19–20 | Full notebook review, polish, submit |

**Deadline: April 20, 2026**

## Dependencies

- Python 3.10+
- NumPy ≥ 1.24
- Matplotlib ≥ 3.7
- Gymnasium ≥ 0.29
