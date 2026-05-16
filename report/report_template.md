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

---

### A.2 — Why is Reinforcement Learning suitable for this problem?

RL is suitable for LLM alignment because:

1. **Sequential decision-making**: Generating a response is a sequence of token-level decisions. Each token choice affects future tokens and the overall quality.

2. **No single "correct" answer**: Unlike supervised learning where there's a ground-truth label, alignment involves subjective human preferences. RL naturally handles optimizing for preference signals rather than exact labels.

3. **Delayed reward**: The quality of a response can only be evaluated after the full response is generated — a classic delayed reward scenario.

4. **Exploration vs. exploitation**: The model must explore diverse response styles while exploiting known good patterns, a core RL trade-off.

5. **Credit assignment**: When a response is rated as poor, RL can help identify which parts (tokens) contributed to the poor rating.

---

### A.3 — Who or what is the agent?

The **agent** is the Large Language Model (LLM) itself — specifically, its policy network that generates text token-by-token. At each step, the agent selects the next token from its vocabulary based on the current context (prompt + tokens generated so far).

More precisely, the agent is the language model's policy π_θ, parameterized by weights θ, which maps a sequence of tokens (the current state) to a probability distribution over the next token (the action).

---

### A.4 — What is the environment?

The **environment** consists of:

- **The State Space ($S$)** consists of all valid sequences of tokens that can fit within the model's context window.
- **The Action Space ($A$)** which is the model's vocabulary any word the model was trained on can be an action (next sequence).
- **The reward model** (in RLHF) or **preference data** (in DPO): Provides feedback on the quality of generated responses.
- **The Transition Space** transitions are deterministic and highly predictable as the next state depends only on the chosen token and the current state.
- ** The Terminal States** The End of Text Token (<|endoftext|>) or Reaching the Max Context Horizon.
  
The environment is fundamentally a text-generation setting where the agent's actions (token selections) modify the environment state (the growing response), and a reward signal is provided after the complete response is generated.

---

### A.5 — Is the task episodic or continuing?

The task is **episodic**. Each episode consists of:

1. **Start**: The agent receives a user prompt
2. **Interaction**: The agent generates tokens one by one
3. **End**: The agent outputs an end-of-sequence token (or reaches a maximum length)
4. **Reward**: A reward is assigned to the complete response (by the reward model or through preference comparison)

Each prompt-response pair is an independent episode. There is no carry-over of state between different prompts unless if the context window can include previous prompts or responses.

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
- In RLHF: R(response) = reward_model(prompt, response).  a scalar score from a learned reward model
- In DPO: Implicit reward derived from preference pairs (preferred response vs. rejected response)
- In PPO: KL-divergence penalty to prevent the aligned model from drifting too far from the base model: r*total = R(response) - β · KL(π*θ || π_ref)

---

---

## Part B: Algorithm Analysis

### Algorithm 1: PPO (Proximal Policy Optimization)

_Assigned to: Omar Ez-Eldin (20220228)_

#### B1.1 — What does PPO do in LLM Alignment?

PPO fine-tunes the LLM's policy to generate responses that receive higher scores from a reward model. It updates the model's parameters to increase the probability of generating high-reward responses while constraining updates to be "proximal" (not too large), preventing catastrophic forgetting or policy collapse.

In LLM alignment, PPO is the optimization engine that takes a reward signal (from a reward model trained on human preferences) and adjusts the LLM to produce more preferred responses.

The Process goes as following:
1. The LLM generates a response, and the Reward Model gives it a score based on human preferences (like helpfulness, clarity, verbostiy, etc.....).
2. PPO takes this score and calculates how to change the LLM's internal weights so the model is more likely to generate high-scoring words in the future and reinforce good subjective qualities (clarity, honesty, reasoning...).

PPO also has safety rails that prevent the model from reaching catastrophic forgetting or reward exploitation that might break the LLM completely.
1. **Clipping:** It ensures the weight changes are small and stable so if we encouter a massive update between old and new probabilities we simply clip (reduce it) the result into our maximimum acceptable value.
2. **KL Penalty:** Kullback-Leibler penalty acts as a permanent anchor. It measures the statistical distance between the token distribution of your current training LLM and the original, unaligned reference LLM.
#### B1.2 — What is the state?

**The State** is the current text sequence (the user's prompt + any other input (PDF) + all tokens generated by the LLM up to this exact moment).

**The state cannot grow indefinitely**. It is limited by the context window and the GPU memory.

#### B1.3 — What is the action?

**The Action is the selection of the single next token from the model's vocabulary.**

The Process goes likes this:
1. The state is passed through the current LLM layers then the final layer outputs a score for every single token in the vocabulary. These scores are converted into percentages using softmax.

2. The model does not pick the absolute highest percentage token all the time (which would make it repetitive), instead it samples from the calculated distribution.

3. then it stores the choosen token and its log probability, because it needs them later to calculate the probability ratio ($r$) during the clipping step.


#### B1.4 — What is the reward?

The Reward is a single value calculated at the very end of the episode (The entire response).

In LLM alignment, rewards are subjective because human preferences apply to the whole thought, not individual words so we need 2 models for an effective evaluation.

1. **The Reward Model** reads the complete response and outputs a base score. The system then subtracts the token-by-token KL penalty to get the final total reward.

2. **The Critic (The Predictor)** is a separate model (usually it is the exact copy of the LLM) its job is to calculate the Advantage ($A$) for every single token action.

**$$A = \text{Actual Reward} - \text{Critic's Predicted Reward}$$**

if we have a positive advantage PPO updates the weights to increase their liklihood and vice versa.

#### B1.5 — Is it value-based, policy-based, actor-critic, bandit, model-based, or offline RL?

PPO is an **actor-critic** reinforcement learning algorithm

- **Actor**: The LLM policy π_θ that generates tokens
- **Critic**: A value network V(s) that estimates expected future reward

The importance of the pipeline is we use the output policy of the actor and reward of the critic which are required to calculate the Advantage used to update the Actor.

#### B1.6 — What is the algorithm flow step by step?

First we need the following as a baseline.
1. a pretrained LLM.
2. a dataset consisting of multiple prompts.

Then the pipeline goes as the following:
1. We fetch a prompt from the dataset.
2. The LLM completes its response based on the prompt.
3. for each token generated we store (the token, its log probability, the critic's final reward from choosing this token)
4. The completed response is passed to the Reward Model to get a base quality score.
5. The system calculates the token-by-token KL Divergence by comparing the current LLM's token probabilities against the original base LLM's probabilities.
6. We use the KL penalty with a scaling factor to calculate the total reward by subtracting the penalty from the model reward.
7. We calculate the Advantage score so we increase the likelihood of values with positive advantage and vice versa
8. We calculates the probability ratio ($r$) between the changing weights and the old weights for each token. 
9. If $r$ exceeds the boundaries of $1 - \epsilon$ or $1 + \epsilon$, the value is clipped to prevent the model from updating too drastically.
10. The LLM updates its weights to make tokens with positive advantages more likely, and tokens with negative advantages less likely.
11. The Critic updates its weights to make its future reward predictions more accurate.

#### B1.7 — What are the advantages?

There are so many advantaged of using **Actor-critic** based algorithms, but the most important ones regarding PPO:

**Training Stability:** The clipping mechanism prevents the model from making massive weight changes that could ruin its pre-existing language capabilities.

**Subjective Alignment:** It allows the model to learn subjective human values (Clarity, honesty, reasoning) that cannot be captured by mathematical functions.

**Built-in Safety Rails:** By working directly with the KL penalty, it keeps the model anchored to a realistic human writing style and prevents it from inventing gibberish to cheat the scoring system.

**Precise Feedback (Credit Assignment):** Through the Actor-Critic framework, PPO can calculate an "Advantage" score for every single token. This allows it to pinpoint exactly which specific words made a sentence good or bad.

#### B1.8 — What are the limitations?

1. **PPO is very sensetive to different hyperparameters** a small difference in the clipping threshold ($\epsilon$), the learning rate, or the KL penalty weight ($\beta$) can cause the entire training run to fail or diverge. Finding the right balance requires extensive trial and error.

2. **PPO Forces the LLM to maximize human preference scores** it often degrades its general capabilities. As the model becomes highly aligned to a specific conversational style, its performance on raw reasoning, math, or technical coding tasks can drop.

3. **Vulnerability to Reward Hacking** because PPO optimizes strictly for the numerical score from the Reward Model, the LLM often finds mathematical loopholes. It learns to output useless text that satisfies the Reward Model's criteria but is unnatural or unhelpful to actual humans.

#### B1.9 — What data, simulator, or feedback is needed?

1. **The Data**
- Prompt Datasets: We need a curated dataset of Instructions and Prompts.
- Diversity: The dataset must cover a wide range of topics so the model doesn't over-optimize for only one type of task.
- Formatiing: All the prompts should be structured in the exact template for production, so the model doesn't suffer from a domain shift.

2. **The Simulator:**
- State Transitions: The simulator takes the prompt data, feeds it to the LLM, receives the token action, appends that token to the history, and passes the new sequence back to the model for the next step.
- Termination: The simulator dictates when an episode ends by stopping the loop when an "End of Text" token is generated or when the maximum context length is hit.

3. **The Feedback:**
- Feedback in PPO is divided into two distinct systems to handle both the overall goal and the step-by-step adjustments.
- External Feedback (The Reward Model): A separate, frozen transformer model trained on human preference data (binary comparisons where a human selected response A over response B) it represents human satisfaction.
- Internal Feedback (The Critic Model): A value prediction network initialized from the same size as the LLM, which updates dynamically alongside the LLM to calculate Value estimates for every single token generated.

#### B1.10 — What can go wrong?

Here we discuss the unexpected downsides of PPO

**Total Reward Hacking:**
The LLM starts generating bizarre strings of text or punctuation that look like total gibberish to a human, but perfectly exploit the Reward Model's code to get a maximum score. The training run finishes, but the resulting model is completely broken and unusable.

**The KL Weight Imbalance**
The scaling factor ($\beta$) for the KL penalty is notoriously hard to set perfectly.
- If $\beta$ is too small: The model ignores the reference anchor.
- If $\beta$ is too large: The penalty completely overpowers the reward model.

**Over-Refusal**
If the prompt dataset or reward model is even slightly unbalanced toward safety over helpfulness, The model learns that the absolute safest way to avoid any negative reward is to refuse to answer anything at all.

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

## D.1 — Metrics for Evaluating Alignment Algorithms

A good evaluation framework should measure not only whether the aligned model produces preferred responses, but also whether it remains useful, safe, and efficient. Common metrics include:

### 1. Win Rate

The **win rate** measures how often the aligned model’s outputs are preferred over a baseline model (usually the original pretrained or SFT model).

For example, human evaluators may compare two responses side by side and choose which is better. If the aligned model wins 70% of comparisons, its win rate is 70%.

This metric directly reflects alignment quality from a user perspective.

---

### 2. Reward Model Score

The **reward model score** is the average score assigned by the learned reward model to generated responses.

Higher reward scores indicate that the policy is producing outputs that better match human preferences according to the reward model.

However, this metric alone can be misleading because the model may exploit weaknesses in the reward model (“reward hacking”).

---

### 3. KL Divergence

**KL divergence** measures how far the aligned policy has drifted from the original reference model.

In PPO-based RLHF, the objective often includes a KL penalty:

\mathrm{KL}(\pi_{\theta} | \pi_{\mathrm{ref}})

A low KL divergence keeps the model close to the pretrained distribution, preserving fluency and general knowledge while still improving alignment.

If KL divergence becomes too large, the model may become unstable or overly optimized for reward.

---

### 4. Perplexity

**Perplexity** measures language modeling quality.

Lower perplexity generally means the model predicts natural language more effectively. Monitoring perplexity helps ensure that alignment training does not degrade the model’s core linguistic abilities.

A model with excellent reward scores but very poor perplexity may become unnatural or repetitive.

---

### 5. Safety Benchmarks

Alignment should improve safety and reliability. Common benchmarks include:

* TruthfulQA for truthfulness
* HHH evaluations (helpful, honest, harmless behavior)
* Toxicity scores
* Jailbreak robustness tests
* Bias and fairness benchmarks

These tests evaluate whether the model avoids harmful, deceptive, or unsafe outputs.

---

### 6. Human Evaluation

Human raters can directly evaluate outputs on dimensions such as:

* Helpfulness
* Harmlessness
* Honesty
* Coherence
* Instruction-following quality

Human evaluation remains one of the most important alignment metrics because automated metrics often fail to capture nuanced preferences.

---

### 7. Standard LLM Benchmarks

Widely used alignment benchmarks include:

* MT-Bench
* Chatbot Arena

These benchmarks compare models across many real-world conversational tasks and provide standardized evaluations.

---

### 8. Training Cost

Practical evaluation should also include efficiency metrics:

* GPU hours
* Wall-clock training time
* Memory usage
* Dollar cost

Methods like DPO are often attractive because they achieve competitive alignment performance with significantly lower computational cost than PPO-based RLHF.

---

## D.2 — Online RL vs Offline RL vs Simulation

### Online RL

**Online RL** generates new responses during training and updates the model continuously based on reward feedback.

Examples:

* PPO
* Stage 3 of RLHF

The model actively explores new behaviors, receives rewards, and improves iteratively.

Advantages:

* Can discover novel high-quality behaviors
* Adapts dynamically during training
* Often achieves strong alignment performance

Disadvantages:

* Computationally expensive
* Requires large-scale generation during training
* Can become unstable
* Sensitive to reward hacking

---

### Offline RL

**Offline RL** trains using a fixed dataset without generating new trajectories during optimization.

Example:

* DPO (Direct Preference Optimization)

DPO learns directly from preference pairs:

* Preferred response
* Rejected response

Advantages:

* Much simpler training pipeline
* More stable optimization
* Lower computational cost
* Easier reproducibility

Disadvantages:

* Limited to behaviors represented in the dataset
* Cannot explore new strategies during training

---

### Simulation

In RLHF systems, “simulation” usually means replacing direct human feedback with a learned **reward model** that simulates human judgment.

Instead of asking humans to score every response, the reward model predicts what humans would prefer.

This approach is necessary because large-scale human evaluation is too expensive to perform continuously during RL training.

---

### Which Approach Is More Practical?

For modern alignment systems:

* **DPO/offline methods** are generally more practical because they are cheaper, simpler, and more stable.
* **Online RL methods** can achieve stronger optimization but require much more engineering and compute.

As a result, many recent systems favor offline preference optimization unless the additional benefits of online exploration justify the cost.

---

## D.3 — Why Not Use Only Supervised Learning?

Supervised fine-tuning (SFT) is important, but it is not sufficient for robust alignment.

### 1. SFT Only Imitates Demonstrations

SFT trains the model to imitate examples written by humans.

However, alignment often depends on subtle preferences rather than exact demonstrations. Human values are difficult to represent fully through labeled examples alone.

---

### 2. Demonstrations Do Not Capture Everything to Avoid

Training data usually contains examples of good behavior, but far fewer examples of undesirable behavior.

As a result, SFT does not strongly teach:

* what *not* to say,
* how to balance competing objectives,
* or how to handle ambiguous situations safely.

Preference learning provides richer signals about undesirable outputs.

---

### 3. Preferences Are Comparative

Human judgment is often comparative rather than absolute.

People can reliably say:

* “Response A is better than Response B”

even when they cannot easily write the perfect response themselves.

SFT cannot directly optimize comparative preferences, whereas RLHF and DPO are explicitly designed for this setting.

---

### 4. Exposure Bias and Distribution Shift

During SFT, the model learns from human-written trajectories.

At inference time, however, the model generates its own outputs step-by-step. Errors can compound because the model encounters states that were not present in training data.

RL-style optimization trains the model under its own generated distribution, reducing this mismatch.

---

### 5. RL Can Discover Novel Good Behaviors

SFT is limited to imitating existing demonstrations.

RL and preference optimization can discover new strategies that humans did not explicitly provide, as long as those strategies receive high reward or strong preference scores.

This allows aligned models to generalize beyond the original demonstration dataset.

---

---

## Part E: AI Chat Links

| Student Name       | Selected Algorithm | Algorithm Level | AI Chat Link(s)                                                       |
| ------------------ | ------------------ | --------------- | --------------------------------------------------------------------- |
| Omar Ez-Eldin      | PPO                | Medium          | https://gemini.google.com/share/5f03e1d67e07                          |
| Yussuf Ahmed       | RLHF               | Medium          | https://www.perplexity.ai/search/3ef3cb70-29ec-461a-9e13-86ae5554bc53 |
| Moaz Gehad         | DPO                | Advanced        | [paste link here]                                                     |
| Mahmoud Ehab       | DPO                | Advanced        |https://chatgpt.com/share/6a0741ce-89a0-83ea-8322-679faf873678         |
| Abdelrhman Ebrahim | DPO                | Advanced        | https://gemini.google.com/share/01ad1439b194                                                     |

---

_End of Report_
