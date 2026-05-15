# Final Assignment: Reinforcement Learning Applications in Real-World Domains

**Course:** AI424 — Reinforcement Learning  
**University:** Cairo University — Faculty of Computer Science and Artificial Intelligence

---

| Name               | ID       | Assigned Algorithm | Level    |
| ------------------ | -------- | ------------------ | -------- |
| Omar Ez-Eldin      | 20220228 | PPO                | Medium   |
| Yussuf Ahmed       | 20220385 | RLHF               | Medium   |
| Moaz Gehad         | 20220340 | DPO                | Advanced |
| Mahmoud Ehab       | 20220457 | DPO                | Advanced |
| Abdelrhman Ebrahim | 20220519 | DPO                | Advanced |

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
- Optional: KL-divergence penalty to prevent the aligned model from drifting too far from the base model: r*total = R(response) - β · KL(π*θ || π_ref)

[TODO: Discuss reward shaping — should you give intermediate rewards per token?]

---

---

## Part B: Algorithm Analysis

### Algorithm 1: PPO (Proximal Policy Optimization)

_Assigned to: Omar Ez-Eldin (20220228)_

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

- r = reward*model(prompt, response) - β · KL(π*θ || π_ref)
- Why is the KL penalty important?
- Is the reward per-token or per-episode?]

#### B1.5 — Is it value-based, policy-based, actor-critic, bandit, model-based, or offline RL?

PPO is a **policy-based** method that uses an **actor-critic** architecture:

- **Actor**: The LLM policy π_θ that generates tokens
- **Critic**: A value network V(s) that estimates expected future reward

[TODO: Explain why PPO needs both actor and critic components.]

#### B1.6 — What is the algorithm flow step by step?

[TODO: Write the step-by-step flow. Start with:]

1. Initialize: Pre-trained LLM as π*θ, copy as reference policy π_ref, initialize value network V*φ
2. Sample a batch of prompts from the dataset
3. Generate responses using current policy π_θ
4. Score responses using the reward model
5. Compute advantages using GAE (Generalized Advantage Estimation)
6. For K epochs:
   a. Compute ratio r*t(θ) = π*θ(a_t|s_t) / π_old(a_t|s_t)
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

_Assigned to: Yussuf Ahmed (20220385)_

#### B2.1 — What does RLHF do in LLM Alignment?

RLHF is a full pipeline (not a single algorithm) that uses human preferences to align an LLM. It has three stages:

1. **SFT (Supervised Fine-Tuning)**: Fine-tune the base LLM on high-quality demonstration data
2. **Reward Model Training**: Train a reward model on human preference comparisons (response A vs. response B)
3. **RL Fine-Tuning**: Use PPO (or another RL algorithm) to optimize the LLM against the learned reward model

In LLM alignment, RLHF is a three‑stage pipeline that uses human feedback to push a pre‑trained language model toward responses that are more helpful, honest, and harmless. In Stage 1 (SFT), the base LLM is fine‑tuned on high‑quality human‑written demonstrations, so it learns a baseline “good behavior” and becomes the SFT model that will later serve as both the starting policy and the reference policy.

In Stage 2 (Reward Model Training), humans compare pairs of responses to the same prompt and choose which they prefer; a separate reward model is trained on these pairwise labels (e.g., with a Bradley–Terry objective) to output a scalar score that predicts human preference for any (prompt, response) pair. This reward model acts as a proxy for human judgment and is frozen once it reaches good accuracy.

In Stage 3 (PPO Fine‑Tuning), the SFT model is further trained with PPO using the reward model as the source of feedback. For each prompt, the current policy generates a response, the reward model assigns it a scalar score plus an implicit KL penalty relative to the frozen reference model, and PPO uses these rewards to adjust the token‑level policy toward responses that the reward model (and therefore humans) prefer, while keeping the policy close to the SFT

#### B2.2 — What is the state?

State in RLHF for LLM alignment is basically:

1.  The prompt + the tokens generated so far (conversation context).

2.  Optionally include how the reward model “sees” this (full response as input).

#### B2.3 — What is the action?

For RLHF in LLM alignment:
The action is the next token selected from the model’s discrete vocabulary at each generation step. The policy outputs a probability distribution over all tokens given the current context (prompt plus previously generated tokens), and one token is then sampled or chosen, so a full response is a sequence of these token-level actions.

#### B2.4 — What is the reward?

1. Bradley–Terry model:
   In RLHF, the reward model is trained with a Bradley–Terry setup, where each response has a latent scalar score, and the probability that one response is preferred over another depends on the difference between their scores.

2. Human comparisons:
   Annotators compare two responses to the same prompt and pick the one they prefer, and the reward model learns to give higher scalar scores to responses that match these preferences. During RL, the model generates a response, the reward model assigns it a scalar R(prompt, response), and this single end-of-episode reward is used to update all the token-level actions in that response.

3. KL penalty term:
   The RL objective also includes a KL-divergence penalty between the aligned policy and the reference SFT policy. Practically, this subtracts a term proportional to KL(pi_aligned || pi_ref) from the effective reward, which keeps the aligned model close to the base model and makes reward hacking less likely.

#### B2.5 — Is it value-based, policy-based, actor-critic, bandit, model-based, or offline RL?

RLHF’s RL part uses PPO, which is a policy-based, actor–critic method that updates the LLM policy with help from a value (critic)
network and stabilizes training via clipping and a KL penalty.

#### B2.6 — What is the algorithm flow step by step?

**Stage 1**: Supervised Fine-Tuning (SFT)

1. Collect a dataset of high‑quality demonstration responses written or selected by humans for many prompts.

2. Fine‑tune the pre‑trained LLM on this dataset so it learns to imitate good behavior; this produces the SFT model,
   which will later serve as both the initial policy and the reference policy.

**Stage 2**: Reward model training

3. For each prompt, collect pairs of model responses and ask annotators to choose which response they prefer.

4. Train a separate reward model on these pairwise labels using a Bradley–Terry style objective so that it outputs a scalar
   score R(prompt, response) that predicts which responses humans prefer.

5. Freeze the reward model once it reaches good accuracy on held‑out preference data.

**Stage 3**: PPO fine‑tuning (RL stage)

6. Initialize the policy with the SFT model and keep a frozen copy as the reference policy.

7. Sample prompts, let the current policy generate full responses, and score each response with the reward model,
   adding a KL penalty term relative to the reference policy.

8. Use these rewards to compute advantages for the token‑level actions, then run several PPO update epochs to adjust
   the policy while keeping updates proximal (clipped ratios and KL control).

9. Repeat steps 7–8 over many batches of prompts until the aligned policy reaches the desired balance of reward, stability, and closeness to the reference model.

#### B2.7 — What are the advantages?

1. **Captures nuanced human preferences**
   RLHF optimizes directly for what humans actually prefer, using pairwise comparisons and a reward model, instead of relying only on fixed ground‑truth labels.

2. **Proven at scale in real systems**
   RLHF‑style pipelines have been successfully used in large production models (e.g., modern chatbots) to make them more helpful, honest, and harmless than pure SFT models.

3. **Better control over behavior**
   By adjusting the reward model and KL penalty, RLHF gives practitioners explicit knobs to trade off alignment, diversity, and closeness to the base model.

4. **Can go beyond demonstration data**
   The PPO stage can discover and reinforce good responses that were not explicitly present in the SFT dataset, potentially improving over pure imitation learning.

#### B2.8 — What are the limitations?

1. High data and annotation cost\*\*:
   RLHF requires curated demonstration data plus many human preference comparisons,
   which is expensive and time‑consuming to collect at scale.

2. **Reward model errors and bias**:
   The reward model only approximates human preference, so it can be noisy or biased (for example, over‑valuing verbose answers),
   and the PPO stage can amplify these mistakes.

3. **Complex, multi‑stage pipeline**:
   RLHF needs three coordinated stages (SFT, reward model training, PPO), multiple large models in memory, and careful engineering of data flow and hyperparameters,
   which makes it hard to implement and tune.

4. **Risk of reward overoptimization and instability**:
   If training is pushed too far, the policy can “hack” the reward model or diverge from the reference model despite KL control,
   leading to degraded quality or unsafe behaviors.

#### B2.9 — What data, simulator, or feedback is needed?

1. **Demonstration data for SFT**
   RLHF needs a dataset of high‑quality prompt–response pairs written or curated by humans to train the initial SFT model
   that serves as both the starting policy and the reference policy.

2. **Human preference comparison data**
   For reward model training, RLHF requires many pairs of responses for the same prompt, each labeled by annotators with which response they prefer, to learn a reward model that maps (prompt, response) to a scalar score.

3. **Prompt distribution and online feedback for PPO**
   In the RL stage, the system needs a realistic distribution of prompts plus the ability to generate new responses and score them with the fixed reward model, effectively using the reward model as a simulator of human feedback.

4. **Substantial compute resources**
   Training the SFT model, reward model, and PPO‑fine‑tuned policy all at large scale requires significant GPU memory and compute, since multiple large models may be active during RLHF.

#### B2.10 — What can go wrong?

1. **Reward model overoptimization (Goodhart’s law)**
   If PPO training focuses too hard on maximizing the reward model’s score, the policy can learn responses that look good to the reward model but are actually worse or less aligned for real users.

2. **Reward hacking**
   The model may discover “shortcuts” that exploit weaknesses in the reward model (for example, always being overly verbose or using certain phrases) to get high reward without truly improving usefulness or safety.

3. **Amplifying biased or low‑quality feedback**
   If human annotators or the comparison data are biased (e.g., preferring a particular style or ignoring safety), RLHF will train the model toward those biases and can reinforce them through repeated updates.

4. **Instability and drift from the base model**
   Poorly tuned KL penalties or PPO hyperparameters can cause the policy to drift too far from the SFT reference model, leading to unstable behavior, mode collapse, or loss of general language ability.

---

### Algorithm 3: DPO (Direct Preference Optimization)

_Assigned to: Moaz Gehad (20220340) — Core Analysis_  
_Mahmoud Ehab (20220457) — Comparison Angle_  
_Abdelrhman Ebrahim (20220519) — Evaluation Angle_

#### B3.1 — What does DPO do in LLM Alignment?

DPO is an advanced alternative to RLHF that eliminates the need for a separate reward model and RL training loop. Instead, DPO directly optimizes the LLM from preference data using a clever mathematical reformulation.

The key insight: DPO shows that the optimal policy under the RLHF objective has a closed-form solution that depends only on the preference data and a reference policy. This means you can skip reward model training and PPO entirely, and instead optimize a simple binary cross-entropy loss on preference pairs.

DPO loss: L*DPO(π*θ; π*ref) = -E[(y_w, y_l) ~ D] [log σ(β · (log π*θ(y*w|x)/π_ref(y_w|x) - log π*θ(y_l|x)/π_ref(y_l|x)))]

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

_Lead author: Mahmoud Ehab (20220457)_  

### C.1 — Direct comparison of the three algorithms

| Criterion                     | PPO                         | RLHF                              | DPO                                    |
| ----------------------------- | --------------------------- | --------------------------------- | -------------------------------------- |
| **Category**                  | Policy-based (Actor-Critic) | Pipeline (SFT + RM + PPO)         | Offline RL / Policy-based              |
| **Action Space**              | Discrete (token-level)      | Discrete (token-level)            | Response-level (implicit)              |
| **Needs Reward Model**        | Yes (provided externally)   | Yes (trained as Stage 2)          | No                                     |
| **Needs RL Loop**             | Yes                         | Yes (Stage 3)                     | No                                     |
| **Training Stability**        | Moderate to Low             | Low                               | High                                   |
| **Data Requirements**         | Moderate (online rollouts)  | High (SFT + preference + RM data) | High (preference pairs only)           |
| **Implementation Complexity** | High                        | Very High                         | Low to Moderate                        |
| **Real-world Safety**         | Moderate                    | High (best alignment control)     | High (but less controllable than RLHF) |
| **Sample Efficiency**         | Low to Moderate             | Low                               | High                                   |
| **Computational Cost**        | High                        | Very High                         | Low to Moderate                        |


PPO, RLHF, and DPO represent three progressively different approaches to aligning large language models with human preferences, differing mainly in how they obtain and use feedback signals. PPO is a classic reinforcement learning method that operates at the token level using an actor–critic setup, requiring a reward model and an online RL loop to optimize policy behavior. While flexible, it tends to be unstable and computationally expensive due to the need for rollouts and careful tuning of reward signals. RLHF extends this idea into a full pipeline: supervised fine-tuning is followed by reward model training and then PPO optimization. This makes RLHF the most structured and widely used alignment framework, but also the most complex and resource-intensive.

DPO, in contrast, removes the need for both a reward model and an explicit RL loop by directly optimizing the policy using preference pairs in an offline manner. This simplification significantly improves training stability and reduces computational cost, while also improving sample efficiency since it avoids expensive environment interactions or rollouts. However, this comes with a trade-off: RLHF with PPO generally provides finer-grained control over optimization dynamics and can incorporate more complex reward shaping, which can be useful in high-stakes safety-critical applications.

Overall, the three methods form a clear spectrum. PPO is the most fundamental but hardest to stabilize, RLHF is the most powerful but operationally heavy, and DPO is the most efficient and stable but slightly less flexible in control. The choice between them depends on the available compute budget, safety requirements, and how much control is needed over the alignment process.
---

### C.2 — Which algorithm is more suitable for discrete actions?

For discrete action spaces, PPO and RLHF (which relies on PPO) are generally the most suitable because they directly optimize a stochastic policy over token-level discrete actions using sampled trajectories and advantage estimates. This allows fine-grained control at each step of generation. DPO also works well in discrete settings, but it optimizes at the response level using preference pairs rather than explicit step-by-step action optimization. As a result, PPO/RLHF are more naturally aligned with discrete sequential decision-making.
---

### C.3 — Which algorithm is more suitable for continuous actions?

For continuous action spaces, PPO is typically the most suitable because it directly supports continuous policies (e.g., Gaussian policy outputs) and provides stable gradient updates through clipping. RLHF (which is built on PPO) can also handle continuous actions in principle, but it is mainly used in discrete settings like language modeling, making it less common in continuous control domains. DPO is the least suitable, since it is designed for preference-based optimization over discrete outputs rather than precise continuous action control. Overall, PPO is the standard choice for continuous control tasks in reinforcement learning.

---

### C.4 — Which one is more stable during training?

In terms of training stability, DPO is generally the most stable because it avoids both reward modeling and online RL optimization, turning the problem into a direct supervised-style objective over preference pairs. This removes common sources of instability like reward hacking and high-variance policy gradients.

PPO is moderately stable due to its clipped objective, but it can still suffer from variance issues, sensitivity to hyperparameters, and instability from reward signals. RLHF (PPO-based pipeline) is the least stable overall because it inherits PPO’s instability and adds extra failure points from reward model training and pipeline mismatch.
---

### C.5 — Which one needs more data?

RLHF typically needs the most data because it requires multiple datasets: supervised fine-tuning data, preference comparison data, and additional samples to train a reward model. This multi-stage setup makes it data-intensive overall.

PPO needs moderate data since it relies on environment interactions or generated rollouts, but it does not require preference datasets or a separate reward model training stage. DPO generally needs less data than RLHF and PPO because it learns directly from preference pairs in an offline setting, though it still requires a reasonably large and high-quality preference dataset to perform well.
---

### C.6 — Which one is easier to implement?

DPO is the easiest to implement because it removes the need for a reward model, policy rollouts, and an explicit RL training loop, reducing the whole process to a supervised learning objective on preference pairs.

PPO is more complex since it requires designing a reward signal, running on-policy rollouts, and carefully tuning stability mechanisms like clipping and advantage estimation. RLHF is the hardest overall because it combines multiple stages—SFT, reward model training, and PPO optimization—making the full pipeline significantly more involved and operationally heavy.
---

### C.7 — Which one is safer for real-world deployment?

RLHF is generally considered the safest for real-world deployment because it allows explicit control over behavior through a learned reward model that can encode human preferences, safety constraints, and red-teaming feedback. This makes it easier to steer the model away from undesirable outputs during training.

DPO is also quite safe in practice since it directly learns from human preference data and is more stable, but it offers less explicit control over fine-grained reward shaping. PPO alone is typically less safe for deployment because it depends heavily on the quality of the reward signal and can be more prone to reward hacking or unintended behaviors if not carefully designed.
---

### C.8 — Which one would you choose and why?

If I had to choose one for most modern LLM alignment setups, I would pick DPO. It offers a strong balance of performance, stability, and simplicity by eliminating the need for a reward model and expensive online RL loops, which significantly reduces engineering complexity and training cost. At the same time, it still aligns models effectively using high-quality preference data, which is often the most reliable signal available.

That said, the choice depends on constraints. If you need maximum control and are working in a safety-critical or highly regulated setting, RLHF (PPO-based) may still be preferable despite its complexity. PPO alone is more suited to research or environments where you have a well-defined reward function and can afford unstable training dynamics.

- For a startup with limited resources → DPO (simpler, cheaper)
- For a major lab with annotators → RLHF (proven at scale)
- For research purposes → PPO (most flexible)
---

## Part D: Evaluation

_Lead author: Abdelrhman Ebrahim (20220519)_  

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

| Student Name       | Selected Algorithm | Algorithm Level | AI Chat Link(s)                                                       |
| ------------------ | ------------------ | --------------- | --------------------------------------------------------------------- |
| Omar Ez-Eldin      | PPO                | Medium          | [paste link here]                                                     |
| Yussuf Ahmed       | RLHF               | Medium          | https://www.perplexity.ai/search/3ef3cb70-29ec-461a-9e13-86ae5554bc53 |
| Moaz Gehad         | DPO                | Advanced        | [paste link here]                                                     |
| Mahmoud Ehab       | DPO                | Advanced        |https://chatgpt.com/share/6a0741ce-89a0-83ea-8322-679faf873678         |
| Abdelrhman Ebrahim | DPO                | Advanced        | [paste link here]                                                     |

---

_End of Report_
