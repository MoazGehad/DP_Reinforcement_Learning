# Section 4 — Conceptual Questions

> **Instructions**: Answer all of the following questions in full, well-reasoned prose.
> Each answer should demonstrate deep understanding of the concepts.

---

## Q5: How does the discount factor γ influence agent behaviour, and what are the trade-offs between low and high values?

*TODO: Discuss how γ controls the agent's time horizon — low γ makes the agent myopic (prioritises immediate rewards), while high γ makes it far-sighted (considers long-term consequences). Discuss convergence speed trade-offs and numerical stability.*

---

## Q6: What is the theoretical relationship between Value Iteration and Policy Iteration?

*TODO: Explain that VI can be seen as a special case of PI where policy evaluation is truncated to a single sweep. Discuss convergence guarantees and computational trade-offs (per-iteration cost vs number of iterations).*

---

## Q7: Why is exploration unnecessary when using Dynamic Programming methods?

*TODO: Explain that DP methods have access to the full transition model P(s'|s,a) and reward function. They compute values for ALL states simultaneously — no need to "discover" the environment through trial and error.*

---

## Q8: What is the distinction between planning and learning in Reinforcement Learning?

*TODO: Discuss model-based planning (using a known model to compute optimal policy) vs learning (improving policy through direct interaction with an unknown environment). Reference Sutton & Barto's definitions.*

---

## Q9: How does having access to a model differ from learning purely through interaction?

*TODO: Compare model-based methods (DP, planning) with model-free methods (MC, TD). Discuss sample efficiency, computational requirements, and model accuracy assumptions.*

---

## Q10: What would happen if an ε-greedy exploration strategy were introduced into a DP agent?

*TODO: Explain that ε-greedy exploration is unnecessary and potentially harmful in DP. Since DP already evaluates all actions for all states, adding randomness would slow convergence and could prevent finding the true optimal policy.*

---

## Q11: Why is a single greedy policy improvement step insufficient to guarantee optimality?

*TODO: Discuss the policy improvement theorem and why iterative improvement is needed. A single greedy step improves the policy but the value function may not yet reflect the true value of the new policy.*

---

## Q12: Can Dynamic Programming methods such as Value Iteration be applied to complex environments like GTA V? Justify your answer in terms of state space size, model availability, and computational feasibility.

*TODO: Argue that DP is impractical for GTA V due to:*
- *Enormous state space (pixels, physics, NPC states → combinatorial explosion)*
- *No known transition model (game engine is a black box)*
- *Computational infeasibility (storing and updating values for all states)*
*Discuss function approximation and deep RL as alternatives.*

---
