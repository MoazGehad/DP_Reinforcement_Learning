# Part C: Algorithm Comparison Matrix — PPO vs RLHF vs DPO

> **Lead Author**: Mahmoud Ehab (20220457)  
> **Co-Author**: Moaz Gehad (20220340)  
> **Status**: [ ] Draft → [ ] Review → [ ] Final

---

## Master Comparison Table

| Dimension | PPO | RLHF | DPO |
|-----------|-----|------|-----|
| **Category** | Policy-based (Actor-Critic) | Multi-stage pipeline (SFT+RM+PPO) | Offline RL / Direct preference optimization |
| **What it learns** | Policy (token-level generation probabilities) | Full alignment system (reward model + policy) | Policy (from preference pairs directly) |
| **Action space** | Discrete (tokens from vocabulary V) | Discrete (tokens) | Response-level (implicit) |
| **Needs reward model** | Yes (provided externally) | Yes (trained in Stage 2) | No |
| **Needs RL loop** | Yes | Yes (Stage 3) | No |
| **Training mode** | Online (generates new data during training) | Online (Stage 3) | Offline (fixed preference dataset) |
| **Exploration strategy** | Token sampling + KL penalty | Token sampling + KL penalty | None (offline) |
| **Training stability** | [⭐–⭐⭐⭐] | [⭐–⭐⭐⭐] | [⭐–⭐⭐⭐] |
| **Sample efficiency** | [⭐–⭐⭐⭐] | [⭐–⭐⭐⭐] | [⭐–⭐⭐⭐] |
| **Data requirements** | Prompts + reward model | Demos + comparisons + prompts | Preference pairs only |
| **Computational cost** | High (policy + value + reward model) | Very high (4 models) | Lower (policy + reference only) |
| **Implementation complexity** | Medium-High | High (3 stages) | Low-Medium |
| **Real-world safety** | [⭐–⭐⭐⭐] | [⭐–⭐⭐⭐] | [⭐–⭐⭐⭐] |
| **Handles reward hacking** | Vulnerable | Vulnerable | Not applicable (no explicit reward) |
| **Can improve beyond data** | Yes (online exploration) | Yes (online exploration) | No (limited by dataset) |
| **Key hyperparameters** | ε (clip), β (KL), learning rate | All PPO params + RM training params | β (KL strength), learning rate |
| **Main failure mode** | Reward hacking, mode collapse | Reward model overoptimization | Distribution shift, data quality |
| **Used by** | OpenAI (InstructGPT/ChatGPT), Anthropic | OpenAI, Google, Meta | Meta (LLaMA-2-Chat), many startups |

---

## C.1 — Direct Comparison (Narrative)

**Paragraph 1 — Philosophical Differences**:

PPO, RLHF, and DPO represent three different philosophies for solving the same problem: making LLMs behave according to human preferences.

PPO is a general-purpose policy optimization algorithm — it doesn't know anything about language or preferences. It simply maximizes whatever reward signal it receives through stable, clipped policy updates. In the LLM context, it's the "engine" that drives alignment when paired with a reward model.

RLHF is not a single algorithm but an end-to-end system. It captures human preferences through a reward model and then uses PPO to optimize against it. It's the full "car" to PPO's "engine" — a complete pipeline from human annotation to aligned model.

DPO takes a radically different approach. It proves mathematically that the reward-model-then-RL approach can be collapsed into a single supervised-learning-style optimization. By reparameterizing the reward function, DPO eliminates both the reward model and the RL loop, directly optimizing the policy from preference data.

[TODO: Continue with Paragraphs 2-3 about practical implications and trade-offs]

---

## C.2 — Discrete Actions

**Best for discrete actions**: [TODO — all three handle discrete tokens, but which is most natural?]

[TODO: 2-3 sentences. PPO explicitly handles the discrete token vocabulary through its policy gradient. RLHF inherits this from PPO. DPO operates at the response level but still internally computes token-level probabilities.]

---

## C.3 — Continuous Actions

**Relevance note**: LLM alignment is inherently discrete (tokens). However:

[TODO: Discuss how PPO is the most versatile — it can handle continuous actions in non-LLM settings (robotics, etc.). RLHF and DPO are specifically designed for the preference-based LLM setting. If the question is about general applicability beyond LLMs, PPO wins.]

---

## C.4 — Training Stability

**Most stable**: DPO

**Why**: [TODO: DPO uses a simple binary cross-entropy loss — equivalent to supervised fine-tuning. No value network estimation errors, no clipping ratio instability, no reward model proxy noise. PPO can oscillate due to the critic's estimation errors and the clipping boundary. RLHF amplifies PPO's instability by adding reward model noise.]

**Ranking**: DPO > PPO > RLHF

---

## C.5 — Data Requirements

**Needs most data**: RLHF

**Why**: [TODO: RLHF needs 3 types of data:
1. Demonstration data for SFT (Stage 1)
2. Human comparison data for RM (Stage 2)
3. Prompt data for PPO training (Stage 3)
DPO needs only preference pairs. PPO needs prompts + a pre-trained reward model.]

**Ranking**: RLHF > PPO > DPO

---

## C.6 — Easiest to Implement

**Easiest**: DPO

**Why**: [TODO: DPO requires implementing one loss function and running standard gradient descent — essentially the same as supervised fine-tuning. PPO requires implementing clipping, GAE, value network, and careful hyperparameter tuning. RLHF requires 3 separate training stages with data pipelines between them.]

**Ranking**: DPO < PPO < RLHF (from easiest to hardest)

---

## C.7 — Real-World Deployment Safety

**Safest for deployment**: [TODO — this is debatable]

Arguments for DPO:
- No online exploration means no risk of generating harmful content during training
- No reward model to hack

Arguments for RLHF:
- Online exploration can discover and avoid harmful edge cases
- Reward model can be used for runtime filtering

[TODO: Make your team's choice and justify]

---

## C.8 — Final Recommendation

**Our choice**: [TODO]

[TODO: Write a well-structured 4-point argument:
1. Problem fit
2. Practical considerations
3. Safety and deployment
4. Trade-offs accepted]

---

## Quality Checklist

- [ ] Every question references LLM Alignment specifically
- [ ] Rankings are consistent (no contradictions across questions)
- [ ] Justifications cite specific algorithm mechanisms
- [ ] The final recommendation logically follows from the analysis
- [ ] All three algorithms are discussed in every answer
