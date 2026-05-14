# Final Assignment: Reinforcement Learning Applications in Real-World Domains

**Course:** AI424 — Reinforcement Learning  
**University:** Cairo University — Faculty of Computer Science and Artificial Intelligence

---

| Name | ID | Assigned Algorithm | Level |
|------|----|--------------------|-------|
| Omar Ez-Eldin | 20220228 | PPO | Medium |
| Yussuf Ahmed | 20220385 | RLHF | Medium |
| Moaz Gehad | 20220340 | DPO | Advanced |
| Mahmoud Ehab | 20220457 | DPO | Advanced |
| Abdelrhman Ebrahim | 20220519 | DPO | Advanced |

**Field:** NLP / LLMs  
**Application Problem:** LLM Alignment  
**Algorithms:** PPO (Medium), RLHF (Medium), DPO (Advanced)

---

## Part A: Problem Formulation

### A.1 — What is the selected field and application problem?

**Field:** Natural Language Processing / Large Language Models (NLP/LLMs)

**Application Problem:** LLM Alignment

LLM Alignment is the challenge of ensuring that a large language model (such as GPT-4, Gemini, or LLaMA) produces outputs that are helpful, honest, and harmless — aligned with human values and intentions. Without alignment, LLMs can generate toxic content, hallucinate facts, refuse reasonable requests, or follow instructions in harmful ways.

The alignment problem is central to deploying LLMs in production: chatbots, code assistants, medical Q&A, and educational tools all require the model to behave according to what humans actually prefer, not just what minimizes a language modeling loss.

[TODO: Expand with your own understanding. Mention specific examples of misalignment you've encountered or read about.]

---

### A.2 — Why is Reinforcement Learning suitable for this problem?

RL is suitable for LLM alignment because:

1. **Sequential decision-making**: Generating a response is a sequence of token-level decisions. Each token choice affects future tokens and the overall quality.

2. **No single "correct" answer**: Unlike supervised learning where there's a ground-truth label, alignment involves subjective human preferences. RL naturally handles optimizing for preference signals rather than exact labels.

3. **Delayed reward**: The quality of a response can only be evaluated after the full response is generated — a classic delayed reward scenario.

4. **Exploration vs. exploitation**: The model must explore diverse response styles while exploiting known good patterns, a core RL trade-off.

5. **Credit assignment**: When a response is rated as poor, RL can help identify which parts (tokens) contributed to the poor rating.

[TODO: Add your own reasoning. Why can't we just use supervised fine-tuning alone?]

---

### A.3 — Who or what is the agent?

The **agent** is the Large Language Model (LLM) itself — specifically, its policy network that generates text token-by-token. At each step, the agent selects the next token from its vocabulary based on the current context (prompt + tokens generated so far).

More precisely, the agent is the language model's policy π_θ, parameterized by weights θ, which maps a sequence of tokens (the current state) to a probability distribution over the next token (the action).

[TODO: Customize further.]

---

### A.4 — What is the environment?

The **environment** consists of:

- **The user prompt / instruction**: The starting input that triggers a response
- **The reward model** (in RLHF) or **preference data** (in DPO): Provides feedback on the quality of generated responses
- **The tokenizer and vocabulary**: Defines the action space
- **The conversation context**: The growing sequence of tokens that forms the current state

The environment is fundamentally a text-generation setting where the agent's actions (token selections) modify the environment state (the growing response), and a reward signal is provided after the complete response is generated.

[TODO: Discuss whether the environment is deterministic or stochastic.]

---

### A.5 — Is the task episodic or continuing?

The task is **episodic**. Each episode consists of:

1. **Start**: The agent receives a user prompt
2. **Interaction**: The agent generates tokens one by one
3. **End**: The agent outputs an end-of-sequence token (or reaches a maximum length)
4. **Reward**: A reward is assigned to the complete response (by the reward model or through preference comparison)

Each prompt-response pair is an independent episode. There is no carry-over of state between different prompts.

[TODO: Discuss nuances — e.g., multi-turn conversations could be seen as longer episodes.]

---

### A.6 — What are the state, action, and reward?

**State (s_t):**
- The concatenation of the user prompt and all tokens generated so far: s_t = [prompt, token_1, token_2, ..., token_t]
- This is a variable-length sequence processed by the transformer's self-attention mechanism
- Dimensionality: sequence of token IDs from a vocabulary of size V (typically 32K–128K)

**Action (a_t):**
- The next token selected from the vocabulary V
- This is a **discrete** action space with |V| possible actions (e.g., 32,000 for LLaMA, 100,000+ for GPT-4)
- The policy outputs a probability distribution over V, and the token is sampled (or selected greedily)

**Reward (r):**
- Typically assigned at the **end of the episode** (after the full response is generated)
- In RLHF: R(response) = reward_model(prompt, response) — a scalar score from a learned reward model
- In DPO: Implicit reward derived from preference pairs (preferred response vs. rejected response)
- Optional: KL-divergence penalty to prevent the aligned model from drifting too far from the base model: r_total = R(response) - β · KL(π_θ || π_ref)

[TODO: Discuss reward shaping — should you give intermediate rewards per token?]

---
---

## Part B: Algorithm Analysis

### Algorithm 1: PPO (Proximal Policy Optimization)

*Assigned to: Omar Ez-Eldin (20220228)*

#### B1.1 — What does PPO do in LLM Alignment?

PPO fine-tunes the LLM's policy to generate responses that receive higher scores from a reward model. It updates the model's parameters to increase the probability of generating high-reward responses while constraining updates to be "proximal" (not too large), preventing catastrophic forgetting or policy collapse.

In LLM alignment, PPO is the optimization engine that takes a reward signal (from a reward model trained on human preferences) and adjusts the LLM to produce more preferred responses.

[TODO: Expand with more detail. How does PPO specifically handle the LLM fine-tuning loop?]

#### B1.2 — What is the state?

[TODO: Define the state representation for PPO in LLM alignment. Consider:
- Prompt + generated tokens so far
- Hidden states of the transformer
- Any additional context (conversation history)]

#### B1.3 — What is the action?

[TODO: Define the action space. PPO works with the discrete token vocabulary. Discuss:
- How PPO handles the large discrete action space
- Whether action masking is used]

#### B1.4 — What is the reward?

[TODO: Define the reward. For PPO in RLHF:
- r = reward_model(prompt, response) - β · KL(π_θ || π_ref)
- Why is the KL penalty important?
- Is the reward per-token or per-episode?]

#### B1.5 — Is it value-based, policy-based, actor-critic, bandit, model-based, or offline RL?

PPO is a **policy-based** method that uses an **actor-critic** architecture:
- **Actor**: The LLM policy π_θ that generates tokens
- **Critic**: A value network V(s) that estimates expected future reward

[TODO: Explain why PPO needs both actor and critic components.]

#### B1.6 — What is the algorithm flow step by step?

[TODO: Write the step-by-step flow. Start with:]
1. Initialize: Pre-trained LLM as π_θ, copy as reference policy π_ref, initialize value network V_φ
2. Sample a batch of prompts from the dataset
3. Generate responses using current policy π_θ
4. Score responses using the reward model
5. Compute advantages using GAE (Generalized Advantage Estimation)
6. For K epochs:
   a. Compute ratio r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)
   b. Compute clipped objective: L = min(r_t·A_t, clip(r_t, 1-ε, 1+ε)·A_t)
   c. Update θ to maximize L
   d. Update φ to minimize value prediction error
7. Repeat from step 2

#### B1.7 — What are the advantages?

[TODO: List at least 3 advantages of PPO for LLM alignment]

#### B1.8 — What are the limitations?

[TODO: List at least 3 limitations]

#### B1.9 — What data, simulator, or feedback is needed?

[TODO: Describe requirements:
- Human preference data (for training the reward model)
- A pre-trained LLM
- Significant GPU compute
- The reward model itself]

#### B1.10 — What can go wrong?

[TODO: Discuss failure modes:
- Reward hacking (gaming the reward model)
- Mode collapse
- KL divergence explosion
- Reward model inaccuracies]

---

### Algorithm 2: RLHF (Reinforcement Learning from Human Feedback)

*Assigned to: Yussuf Ahmed (20220385)*

#### B2.1 — What does RLHF do in LLM Alignment?

RLHF is a full pipeline (not a single algorithm) that uses human preferences to align an LLM. It has three stages:
1. **SFT (Supervised Fine-Tuning)**: Fine-tune the base LLM on high-quality demonstration data
2. **Reward Model Training**: Train a reward model on human preference comparisons (response A vs. response B)
3. **RL Fine-Tuning**: Use PPO (or another RL algorithm) to optimize the LLM against the learned reward model

RLHF is the standard approach used by OpenAI (InstructGPT, ChatGPT), Anthropic (Claude), and Google (Gemini) for aligning their models.

[TODO: Expand on each stage and their interactions.]

#### B2.2 — What is the state?

[TODO: Same framework as PPO, but discuss how RLHF's state includes the reward model's perspective]

#### B2.3 — What is the action?

[TODO: Token-level actions, same vocabulary. Discuss the full action pipeline across all 3 stages]

#### B2.4 — What is the reward?

[TODO: The reward model R_ψ trained on human preferences. Discuss:
- Bradley-Terry model for preference modeling
- How human comparisons become a scalar reward
- KL penalty]

#### B2.5 — Is it value-based, policy-based, actor-critic, bandit, model-based, or offline RL?

[TODO: RLHF as a system uses policy-based RL (specifically PPO, which is actor-critic) in its third stage. Classify and explain.]

#### B2.6 — What is the algorithm flow step by step?

[TODO: Write the full 3-stage pipeline:
Stage 1: SFT
Stage 2: Reward Model Training
Stage 3: PPO Fine-Tuning
Be specific about data flow between stages]

#### B2.7 — What are the advantages?

[TODO: Advantages of RLHF — captures nuanced human preferences, proven at scale, etc.]

#### B2.8 — What are the limitations?

[TODO: Limitations — expensive human annotation, reward model errors, complexity of 3-stage pipeline, etc.]

#### B2.9 — What data, simulator, or feedback is needed?

[TODO: Human annotators, demonstration data, comparison data, compute resources]

#### B2.10 — What can go wrong?

[TODO: Reward model overoptimization, biased human feedback, Goodhart's law, etc.]

---

### Algorithm 3: DPO (Direct Preference Optimization)

*Assigned to: Moaz Gehad (20220340) — Core Analysis*  
*Mahmoud Ehab (20220457) — Comparison Angle*  
*Abdelrhman Ebrahim (20220519) — Evaluation Angle*

#### B3.1 — What does DPO do in LLM Alignment?

DPO is an advanced alternative to RLHF that eliminates the need for a separate reward model and RL training loop. Instead, DPO directly optimizes the LLM from preference data using a clever mathematical reformulation.

The key insight: DPO shows that the optimal policy under the RLHF objective has a closed-form solution that depends only on the preference data and a reference policy. This means you can skip reward model training and PPO entirely, and instead optimize a simple binary cross-entropy loss on preference pairs.

DPO loss: L_DPO(π_θ; π_ref) = -E[(y_w, y_l) ~ D] [log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]

Where y_w is the preferred response, y_l is the rejected response, and β controls the KL constraint strength.

[TODO: Expand with intuition — why does this work? What mathematical insight makes it possible?]

#### B3.2 — What is the state?

[TODO: The prompt x. In DPO, the "state" is simpler because we work with complete responses, not token-by-token]

#### B3.3 — What is the action?

[TODO: The complete response y. DPO operates at the response level, not the token level during optimization. Discuss implications.]

#### B3.4 — What is the reward?

[TODO: DPO has an implicit reward:
r(x, y) = β · log(π_θ(y|x) / π_ref(y|x)) + β · log Z(x)
The reward is never explicitly computed — it's implicitly captured in the policy itself.]

#### B3.5 — Is it value-based, policy-based, actor-critic, bandit, model-based, or offline RL?

[TODO: DPO is technically **offline RL** — it learns from a fixed dataset of preferences without online interaction. Some classify it as policy-based since it directly optimizes the policy. Discuss both perspectives.]

#### B3.6 — What is the algorithm flow step by step?

[TODO: Write the DPO pipeline:
1. Start with a pre-trained LLM (base model)
2. (Optional) SFT on demonstration data → π_ref
3. Collect preference dataset D = {(x, y_w, y_l)}
4. For each training step:
   a. Sample batch of preference pairs
   b. Compute log probabilities under π_θ and π_ref for both y_w and y_l
   c. Compute DPO loss (binary cross-entropy on preference margin)
   d. Update θ using gradient descent
5. No reward model, no PPO, no value network]

#### B3.7 — What are the advantages?

[TODO: At least 3 advantages:
- No reward model needed
- No RL training loop (more stable)
- Simpler implementation
- Lower computational cost
- No reward hacking possible (no explicit reward to hack)]

#### B3.8 — What are the limitations?

[TODO: At least 3 limitations:
- Requires high-quality preference data
- Offline only (can't do online exploration)
- Performance depends heavily on reference policy quality
- May underperform RLHF when online data collection is possible
- Theoretical guarantees assume specific preference models]

#### B3.9 — What data, simulator, or feedback is needed?

[TODO: Preference pairs dataset, reference model, compute for fine-tuning (less than RLHF)]

#### B3.10 — What can go wrong?

[TODO: Distribution shift from reference policy, noisy preferences, out-of-distribution prompts]

---
---

## Part C: Comparison

*Lead author: Mahmoud Ehab (20220457)*  
*Co-author: Moaz Gehad (20220340)*

### C.1 — Direct comparison of the three algorithms

| Criterion | PPO | RLHF | DPO |
|-----------|-----|------|-----|
| **Category** | Policy-based (Actor-Critic) | Pipeline (SFT + RM + PPO) | Offline RL / Policy-based |
| **Action Space** | Discrete (token-level) | Discrete (token-level) | Response-level (implicit) |
| **Needs Reward Model** | Yes (provided externally) | Yes (trained as Stage 2) | No |
| **Needs RL Loop** | Yes | Yes (Stage 3) | No |
| **Training Stability** | [TODO] | [TODO] | [TODO] |
| **Data Requirements** | [TODO] | [TODO] | [TODO] |
| **Implementation Complexity** | [TODO] | [TODO] | [TODO] |
| **Real-world Safety** | [TODO] | [TODO] | [TODO] |
| **Sample Efficiency** | [TODO] | [TODO] | [TODO] |
| **Computational Cost** | [TODO] | [TODO] | [TODO] |

[TODO: Write 2-3 paragraphs synthesizing this table into a narrative comparison]

---

### C.2 — Which algorithm is more suitable for discrete actions?

[TODO: All three handle discrete actions (tokens), but PPO and RLHF explicitly work at the token level during training. Discuss.]

---

### C.3 — Which algorithm is more suitable for continuous actions?

[TODO: None of these are primarily designed for continuous actions in the LLM context. However, discuss how PPO could theoretically handle continuous actions in other domains. Note that LLM alignment is inherently discrete.]

---

### C.4 — Which one is more stable during training?

[TODO: DPO is generally most stable (simple supervised loss). PPO can be unstable. RLHF inherits PPO's instability. Justify with specific mechanisms.]

---

### C.5 — Which one needs more data?

[TODO: RLHF needs the most data (demonstration data + comparison data + RL samples). DPO needs preference pairs only. PPO needs reward model outputs + generated samples. Rank and justify.]

---

### C.6 — Which one is easier to implement?

[TODO: DPO is simplest (one loss function, standard fine-tuning). PPO is moderately complex. RLHF is most complex (3 stages, multiple models). Justify.]

---

### C.7 — Which one is safer for real-world deployment?

[TODO: Consider reward hacking (PPO/RLHF risk), distribution shift (DPO risk), predictability, and constraint satisfaction. Which would you trust more in production?]

---

### C.8 — Which one would you choose and why?

[TODO: Provide a well-reasoned team recommendation. Consider:
- For a startup with limited resources → DPO (simpler, cheaper)
- For a major lab with annotators → RLHF (proven at scale)
- For research purposes → PPO (most flexible)
Make your choice and argue for it.]

---
---

## Part D: Evaluation

*Lead author: Abdelrhman Ebrahim (20220519)*  
*Co-author: Mahmoud Ehab (20220457)*

### D.1 — What metrics would you use to evaluate the algorithms?

[TODO: List and explain metrics such as:
- **Win rate**: How often is the aligned model's response preferred over the base model
- **Reward model score**: Average score from the reward model
- **KL divergence**: How far the aligned model has drifted from the reference
- **Perplexity**: Has the model's language ability degraded?
- **Safety benchmarks**: TruthfulQA, HHH evaluation, toxicity scores
- **Human evaluation**: Helpfulness, harmlessness, honesty ratings
- **MT-Bench / Chatbot Arena**: Standard LLM alignment benchmarks
- **Training cost**: GPU hours, wall-clock time, dollar cost]

### D.2 — Would you use online RL, offline RL, or simulation?

[TODO: Discuss:
- PPO uses online RL (generates new responses during training)
- RLHF uses online RL (Stage 3)
- DPO uses offline RL (fixed preference dataset)
- Simulation in this context means using a reward model as a proxy for human judgment
- Which approach is more practical? More effective?]

### D.3 — Why not use only supervised learning?

[TODO: Explain why SFT alone is insufficient:
- SFT teaches the model to imitate demonstrations, but can't capture nuanced preferences
- No demonstration data shows the full spectrum of "what NOT to do"
- Preferences are comparative (A > B), which SFT can't directly optimize
- SFT suffers from exposure bias and distribution mismatch
- RL/preference optimization can discover novel good behaviors beyond what demonstrations show]

---
---

## Part E: AI Chat Links

| Student Name | Selected Algorithm | Algorithm Level | AI Chat Link(s) |
|-------------|-------------------|----------------|-----------------|
| Omar Ez-Eldin | PPO | Medium | [paste link here] |
| Yussuf Ahmed | RLHF | Medium | [paste link here] |
| Moaz Gehad | DPO | Advanced | [paste link here] |
| Mahmoud Ehab | DPO | Advanced | [paste link here] |
| Abdelrhman Ebrahim | DPO | Advanced | [paste link here] |

---

*End of Report*
