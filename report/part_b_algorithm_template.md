# Algorithm Analysis Worksheet — LLM Alignment

> **Instructions**: Each member fills this out for their assigned algorithm, then merges into the main report under Part B.
> Print this or copy it to your own doc. Fill in all [TODO] sections.

---

## Your Information

- **Your Name**: ___________________
- **Algorithm Name**: ___________________  *(PPO / RLHF / DPO)*
- **Algorithm Level**: Medium / Advanced
- **Application Problem**: LLM Alignment

---

## Pre-Research Checklist

Before writing, ensure you can confidently answer these:

- [ ] I know what type of RL my algorithm belongs to (value-based, policy-based, actor-critic, offline RL, etc.)
- [ ] I understand the loss function or objective my algorithm optimizes
- [ ] I can explain the algorithm's training loop step by step
- [ ] I understand how it handles the state (prompt + tokens), action (next token), and reward (preference signal)
- [ ] I know how my algorithm differs from the other two (PPO vs RLHF vs DPO)
- [ ] I have completed my AI chat session with genuine exploration and critical thinking

---

## Question 1: What does this algorithm do in LLM Alignment?

> *Don't describe the algorithm generically. Explain it IN THE CONTEXT of LLM Alignment.*
> *Example: "In LLM alignment, PPO adjusts the language model's token-generation policy to increase the probability of producing responses that score highly with a reward model trained on human preferences..."*

[TODO — 2-3 paragraphs]

---

## Question 2: What is the state?

| Variable | Description | Type | Example |
|----------|-------------|------|---------|
| Prompt | User instruction/question | Token sequence | "Explain quantum computing simply" |
| Generated tokens | Tokens produced so far | Token sequence | "Quantum computing uses..." |
| Context window | Full sequence fed to transformer | Concatenation | [prompt + generated tokens] |
| [Add more?] | | | |

**State representation**: Sequence of token IDs processed by transformer self-attention

**Dimensionality**: Variable length, up to context window size (e.g., 2048–8192 tokens), each token from vocabulary V (32K–128K)

**How YOUR algorithm uses the state**: [TODO — Does your algorithm process the state token-by-token (PPO) or as complete responses (DPO)?]

---

## Question 3: What is the action?

**Action space type**: Discrete

| For PPO/RLHF | For DPO |
|---------------|---------|
| Next token from vocabulary V | Implicit — operates on complete responses |
| |V| ≈ 32,000–128,000 possible actions | Actions are pairs: (preferred response, rejected response) |
| Sampled from π_θ(·|s_t) at each timestep | Not explicitly selected during optimization |

**How YOUR algorithm selects actions**: [TODO — sampling strategy, temperature, top-k/top-p?]

---

## Question 4: What is the reward?

| Reward Component | Description | When Received |
|-----------------|-------------|---------------|
| **PPO/RLHF**: Reward model score | R_ψ(prompt, response) — scalar | End of episode |
| **PPO/RLHF**: KL penalty | -β · KL(π_θ ‖ π_ref) | Per-token or end |
| **DPO**: Implicit reward | β · log(π_θ(y\|x)/π_ref(y\|x)) | Never computed explicitly |

**For YOUR algorithm, the reward is**: [TODO — be specific]

**Reward type**: Sparse (end-of-episode only) / Dense (per-token)? [TODO]

**Potential issues**: [TODO — reward hacking? Noisy reward model? Implicit reward misspecification?]

---

## Question 5: Algorithm classification

**My algorithm is**: [value-based / policy-based / actor-critic / offline RL / ...]

**Justification** (2-3 sentences):
[TODO — Why does it belong to this category? What does it learn? How does it make decisions?]

| | PPO | RLHF | DPO |
|---|-----|------|-----|
| Classification | Actor-Critic (policy-based with value baseline) | System using Actor-Critic (PPO) in Stage 3 | Offline RL / Direct preference optimization |

---

## Question 6: Algorithm flow — step by step

```
For PPO:
1. Initialize π_θ from SFT model, create frozen copy π_ref
2. Initialize value network V_φ (critic)
3. For each training iteration:
   a. Sample batch of prompts from dataset
   b. Generate full responses using π_θ (with sampling)
   c. Score each response with reward model R_ψ
   d. Compute per-token KL penalties
   e. Compute advantages using GAE
   f. For K minibatch epochs:
      i.  Compute ratio r_t = π_θ(a_t|s_t) / π_old(a_t|s_t)
      ii. Compute clipped surrogate: L = min(r_t·A_t, clip(r_t,1-ε,1+ε)·A_t)
      iii. Update θ to maximize L
      iv. Update φ to minimize (V_φ(s_t) - R_t)²
4. Repeat until convergence
```

```
For RLHF:
Stage 1 — SFT:
1. Collect demonstration dataset D_demo
2. Fine-tune base LLM on D_demo using standard language modeling loss

Stage 2 — Reward Model:
1. Collect comparison dataset D_comp = {(x, y_w, y_l)}
2. Train R_ψ using Bradley-Terry loss: L = -log σ(R_ψ(x,y_w) - R_ψ(x,y_l))
3. Validate R_ψ accuracy on held-out comparisons

Stage 3 — PPO Fine-tuning:
[Same as PPO flow above, using R_ψ from Stage 2]
```

```
For DPO:
1. Start with SFT model as π_ref (frozen)
2. Initialize π_θ = π_ref (trainable copy)
3. Load preference dataset D = {(x, y_w, y_l)}
4. For each training step:
   a. Sample minibatch of preference pairs
   b. Compute log π_θ(y_w|x) and log π_θ(y_l|x)
   c. Compute log π_ref(y_w|x) and log π_ref(y_l|x)
   d. Compute DPO loss:
      L = -E[log σ(β·(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]
   e. Update θ using gradient descent
5. No reward model, no value network, no online generation
```

**YOUR algorithm's flow**: [TODO — customize the relevant template above with additional details]

---

## Question 7: Advantages

| # | Advantage | Explanation in LLM Alignment Context |
|---|-----------|-------------------------------------|
| 1 | [TODO] | [TODO] |
| 2 | [TODO] | [TODO] |
| 3 | [TODO] | [TODO] |
| 4 | [TODO] | [TODO] |

---

## Question 8: Limitations

| # | Limitation | Explanation in LLM Alignment Context |
|---|-----------|-------------------------------------|
| 1 | [TODO] | [TODO] |
| 2 | [TODO] | [TODO] |
| 3 | [TODO] | [TODO] |
| 4 | [TODO] | [TODO] |

---

## Question 9: Data, simulator, or feedback needed

| Requirement | PPO | RLHF | DPO |
|-------------|-----|------|-----|
| Pre-trained LLM | ✅ | ✅ | ✅ |
| SFT demonstration data | Optional (uses SFT model) | ✅ (Stage 1) | ✅ (for π_ref) |
| Human comparison data | ❌ (needs reward model) | ✅ (Stage 2) | ✅ |
| Reward model | ✅ (provided) | ✅ (trained) | ❌ |
| Value network | ✅ | ✅ | ❌ |
| Online generation | ✅ | ✅ | ❌ |
| GPU requirements | High (4 models in memory) | Very high | Moderate (2 models) |

**For YOUR algorithm specifically**: [TODO — detail what you need and what's available]

---

## Question 10: What can go wrong?

| Failure Mode | Description | Severity | Mitigation |
|-------------|-------------|----------|------------|
| **PPO**: Reward hacking | Model finds shortcuts to maximize reward score without actual quality | High | Better reward model, KL penalty |
| **PPO**: Mode collapse | Model converges to repetitive, safe responses | Medium | Entropy bonus, diverse prompts |
| **RLHF**: Reward overoptimization | As PPO trains longer, quality peaks then degrades | High | Early stopping, KL monitoring |
| **RLHF**: Annotation bias | Human annotators inject biases (verbosity, style) | Medium | Annotator guidelines, diverse pool |
| **DPO**: Distribution shift | Model diverges from reference, implicit reward becomes inaccurate | High | β tuning, data quality |
| **DPO**: Data quality dependence | Noisy preferences directly corrupt the policy | High | Data filtering, cross-validation |

**For YOUR algorithm**: [TODO — which of these are most relevant? Are there others?]

---

## Self-Review Checklist

- [ ] All 10 questions answered completely
- [ ] Answers are specific to LLM Alignment (not generic algorithm descriptions)
- [ ] State, action, reward are consistent with Part A
- [ ] Step-by-step flow is detailed enough that someone could implement it
- [ ] At least 3 advantages AND 3 limitations listed
- [ ] Failure modes include concrete LLM-specific examples
- [ ] My AI chat link is ready and shows genuine reasoning
