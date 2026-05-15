# Part C: Algorithm Comparison Matrix — PPO vs RLHF vs DPO

> **Lead Author**: Mahmoud Ehab (20220457)  
> **Co-Author**: Moaz Gehad (20220340)  
> **Status**: [ ] Draft → [ ] Review → [✓] Final

---

## Master Comparison Table

| Dimension                     | PPO                                           | RLHF                                          | DPO                                          |
| ----------------------------- | --------------------------------------------- | --------------------------------------------- | -------------------------------------------- |
| **Category**                  | Policy-based (Actor-Critic)                   | Multi-stage pipeline (SFT + RM + PPO)         | Offline RL / Direct preference optimization  |
| **What it learns**            | Policy (token-level generation probabilities) | Full alignment system (reward model + policy) | Policy (directly from preference pairs)      |
| **Action space**              | Discrete (tokens from vocabulary V)           | Discrete (tokens)                             | Response-level (implicit token reweighting)  |
| **Needs reward model**        | Yes (external reward signal required)         | Yes (trained in Stage 2)                      | No                                           |
| **Needs RL loop**             | Yes                                           | Yes (Stage 3 PPO optimization)                | No                                           |
| **Training mode**             | Online (on-policy rollouts)                   | Online (PPO stage)                            | Offline (fixed dataset)                      |
| **Exploration strategy**      | Token sampling + KL penalty                   | Token sampling + KL penalty                   | None (purely offline optimization)           |
| **Training stability**        | ⭐⭐☆ (moderate)                                | ⭐ (low)                                       | ⭐⭐⭐ (high)                                   |
| **Sample efficiency**         | ⭐⭐☆                                           | ⭐⭐☆                                           | ⭐⭐⭐                                          |
| **Data requirements**         | Prompts + reward model                        | SFT data + preference data + prompts          | Preference pairs only                        |
| **Computational cost**        | High                                          | Very high                                     | Low–moderate                                 |
| **Implementation complexity** | Medium–High                                   | Very High                                     | Low–Medium                                   |
| **Real-world safety**         | ⭐⭐☆                                           | ⭐⭐⭐ (best controllability)                    | ⭐⭐⭐ (stable but less controllable)           |
| **Handles reward hacking**    | Vulnerable                                    | Vulnerable (reward model overoptimization)    | Not applicable                               |
| **Can improve beyond data**   | Yes (online exploration)                      | Yes                                           | Limited (dataset-bound)                      |
| **Key hyperparameters**       | Clip ε, KL β, LR, value loss weight           | All PPO params + RM training params           | β (preference strength), LR                  |
| **Main failure mode**         | Reward hacking, instability                   | Reward model misalignment, pipeline mismatch  | Distribution shift, data quality dependence  |
| **Used by**                   | Research + RLHF backbones                     | OpenAI, Google, Anthropic                     | Meta (LLaMA-2-Chat style training), startups |


---

## C.1 — Direct Comparison (Narrative)

### Paragraph 1 — Philosophical Differences

PPO, RLHF, and DPO represent three distinct paradigms for aligning large language models with human preferences. PPO is a general-purpose reinforcement learning algorithm that optimizes a policy using reward signals without assuming anything about language structure. It acts as the underlying optimization engine that updates token-level probabilities using advantage estimates and clipped policy gradients.

RLHF extends PPO into a full alignment pipeline. It introduces a reward model trained from human preference data and then uses PPO to optimize the policy against this learned reward. As a result, RLHF is not a single algorithm but a complete system that connects human feedback to model optimization through intermediate modeling.

DPO takes a fundamentally different approach by eliminating both the reward model and the reinforcement learning loop. It reformulates the RLHF objective into a direct optimization problem over preference pairs, effectively turning alignment into a supervised learning task over relative preferences.

### Paragraph 2 — Practical Implications

In practice, these differences strongly affect complexity, cost, and stability. PPO is flexible but difficult to stabilize due to high variance in policy gradients and sensitivity to reward scaling. RLHF improves expressiveness and control but introduces multiple failure points, including reward model bias and pipeline mismatch between training stages.

DPO significantly simplifies training by removing online sampling and reward modeling, making it far cheaper and more stable. However, it sacrifices some flexibility in reward shaping and fine-grained control compared to RLHF.

### Paragraph 3 — Trade-off Summary

Overall, the three methods form a clear spectrum. PPO provides the most general optimization framework, RLHF offers the strongest controllability through explicit reward modeling, and DPO provides the most efficient and stable training procedure. The choice depends on whether the priority is flexibility (PPO), control (RLHF), or simplicity and efficiency (DPO).

---

## C.2 — Discrete Actions

All three methods operate naturally in discrete action spaces since LLMs generate tokens from a finite vocabulary. PPO explicitly models a stochastic policy over tokens and applies policy gradients at each step, making it directly suited for sequential discrete decision-making.

RLHF inherits this property because its optimization stage is built on PPO. DPO, while operating at the response level, still implicitly optimizes token probabilities through preference-based likelihood shaping. As a result, PPO and RLHF are more naturally aligned with fine-grained discrete control, while DPO operates at a more aggregated level.

---

## C.3 — Continuous Actions

PPO is the most general framework and naturally extends to continuous action spaces using distributions such as Gaussians, making it widely applicable in robotics and control tasks. RLHF and DPO, in contrast, are specialized for language modeling and preference-based alignment, which are inherently discrete.

Therefore, PPO is the most suitable algorithm for continuous action domains, while RLHF and DPO are primarily designed for discrete token-based systems.

---

## C.4 — Training Stability

DPO is the most stable method because it reduces the problem to a supervised learning objective over preference pairs, eliminating reward model errors and RL-induced variance. PPO is moderately stable due to its clipped objective, but it still suffers from gradient variance and sensitivity to hyperparameters.

RLHF is the least stable overall because it combines PPO instability with additional noise introduced by reward model approximation and multi-stage training mismatch.

Ranking: DPO > PPO > RLHF

---

## C.5 — Data Requirements

RLHF requires the most data because it depends on three separate datasets: supervised fine-tuning data, human preference comparisons for reward modeling, and prompt data for PPO optimization. PPO requires fewer datasets but still depends on a reward model and interaction data.

DPO is the most data-efficient because it requires only preference pairs and directly optimizes from them without intermediate modeling stages.

Ranking: RLHF > PPO > DPO

---

## C.6 — Easiest to Implement

DPO is the easiest to implement because it reduces alignment to a single supervised-style loss over preference pairs. PPO is more complex due to the need for reward signals, value networks, and stable policy optimization techniques like clipping and advantage estimation.

RLHF is the most complex because it involves multiple stages (SFT → reward model → PPO) and requires coordination between separate training pipelines.

Ranking (easiest → hardest): DPO < PPO < RLHF

---

## C.7 — Real-World Deployment Safety

RLHF is generally considered the most controllable for real-world deployment because it allows explicit shaping of behavior through a learned reward model, which can incorporate safety rules and human feedback directly.

DPO is highly stable and avoids reward hacking risks but offers less explicit control over safety constraints compared to RLHF. PPO alone is the least safe for deployment because it is highly dependent on reward design and is more prone to reward hacking and unintended optimization behaviors.

---

## C.8 — Final Recommendation

For modern large language model alignment, DPO is the most practical default choice due to its simplicity, stability, and strong empirical performance using preference data alone. It significantly reduces computational cost and engineering complexity while maintaining strong alignment quality.

However, the optimal choice depends on constraints. RLHF remains preferable in high-stakes or regulated environments where fine-grained control and interpretability of the reward system are critical. PPO alone is best suited for research or settings with explicit reward functions and tolerance for instability.

In summary:

DPO → best default (efficiency + stability)
RLHF → best control (safety-critical systems)
PPO → foundational baseline (research & general RL)

---

## Quality Checklist

- [✓] Every question references LLM Alignment specifically
- [✓] Rankings are consistent (no contradictions across questions)
- [✓] Justifications cite specific algorithm mechanisms
- [✓] The final recommendation logically follows from the analysis
- [✓] All three algorithms are discussed in every answer
