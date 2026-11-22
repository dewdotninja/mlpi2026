# Chapter 15. Reinforcement Learning for PID Tuning

*Part IV. Learning to act*

> **Case study:** tuning a PID controller for a third-order lag, first for one plant and then for a whole family of plants
> **Companion notebook:** [`extras/PID_tuning_with_RL/RL_PID_tuning_third_order_lag.ipynb`](../../extras/PID_tuning_with_RL/RL_PID_tuning_third_order_lag.ipynb) (about two minutes on a CPU)
> **Built from:** the PID-tuning-with-RL extra (lecture and notebook)

Chapters 13 and 14 used reinforcement learning to *replace* a controller. Most industrial loops are not going to be replaced. They are PID loops, there are thousands of them on a plant, and many are badly tuned. This chapter asks a narrower and more practical question: **can a learning agent tune the PID?** And, just as important, what does that buy us over Ziegler–Nichols, or over an ordinary optimiser?

The answer turns out to depend on how many plants you have. For one fixed plant, a classical optimiser beats RL by more than an order of magnitude in experiments. For a *family* of plants, RL learns something no optimiser produces: a tuning rule, or a tuning procedure, that works on plants it has never seen. That is what Ziegler and Nichols did in 1942, with a slide rule and a great many experiments.

## Learning objectives

After this chapter you should be able to:

1. Cast PID tuning as an MDP in at least three ways, and state the state, action, reward and episode of each.
2. Write a cost that encodes a control specification, and predict what goes wrong when a term is missing.
3. Derive and implement the REINFORCE gradient for a Gaussian policy.
4. Explain why a learned tuner must be trained on a plant family with dimensionless features and rewards.
5. Compare RL tuners fairly against Ziegler–Nichols, a derivative-free optimiser and a per-plant optimum.
6. Verify learned gains with gain margin, phase margin and peak sensitivity before deployment.

---

## 15.1 The plant and the baseline

### 15.1.1 A third-order lag

$$
P(s) = \frac{1}{(s+1)^3},\qquad C(s) = K_p + \frac{K_i}{s} + \frac{K_d\, s}{T_f s + 1}.
$$

Three identical lags each contribute up to −90° of phase, so the open loop reaches −180° at a finite frequency. By hand: −3 arctan ω = −180° gives ω_u = √3 = 1.732 rad/s, where ∣P∣ = 4^(−3/2) = 1/8. The ultimate gain is **K_u = 8** and the ultimate period **T_u = 2π/√3 = 3.628 s**. On a real plant you would measure them with a **relay test** (Åström and Hägglund, 1984): replace the controller with a relay and the loop settles into a limit cycle at ω_u.

The derivative is filtered, with T_f = T_u/36 ≈ 0.10 s, fixed rather than tuned. An ideal K_d s is improper and would send an infinite impulse to the actuator on a set-point step; every real PID filters it.

### 15.1.2 Ziegler–Nichols, and what it was designed for

Ziegler–Nichols (ZN) is a **tuning rule**: a fixed map from two measured numbers (K_u, T_u) to three gains.

| Rule | K_p / K_u | T_i / T_u | T_d / T_u | Designed for | Gains on the nominal plant (K_p, K_i, K_d) |
|---|---|---|---|---|---|
| Classic | 0.60 | 0.50 | 0.125 | Quarter-amplitude decay of a **load** disturbance | 4.80, 2.65, 2.18 |
| Some overshoot | 0.33 | 0.50 | 0.33 | A gentler set-point response | 2.64, 1.46, 3.19 |
| No overshoot | 0.20 | 0.50 | 0.33 | Very gentle | 1.60, 0.88, 1.93 |

ZN classic overshoots a set-point step by **46%** and rings (2% settling time 9.0 s), but it rejects a load step quickly, which is exactly what it was designed for. Every test in this chapter therefore has two segments: a unit set-point step at t = 0, then a load step at the plant input at t = 10 T_u, ending at 20 T_u.

### 15.1.3 The environment is an exact simulator

Every RL method in this chapter interacts with the loop only through **experiments**: apply some gains, run the two-segment test, measure the response. The simulator is the RL environment, and it has to be right. With the plant written as three lag states plus an integrator state and a derivative-filter state, the closed loop is one linear system ż = Az + B[r, d]ᵀ. The inputs are piecewise constant, so a zero-order-hold discretisation through one matrix exponential is *exact*. Checked against an independent transfer-function step response from `scipy.signal`, the two agree to **3.2 × 10⁻¹⁵**.

Stability comes for free: the closed loop is unstable if and only if A has an eigenvalue with non-negative real part, so the simulator checks that before running and charges a fixed cost instead.

## 15.2 Writing the reward: the most important design decision

An RL agent maximises *exactly* the number you hand it. All of your control-engineering knowledge has to go into that number. The cost (reward = −J) has four terms, each dimensionless so that the same formula works across plants:

$$
J = \underbrace{\frac{1}{T_u}\int_{\text{set-point}} |e|\,dt}_{\text{IAE}_r}
  + \underbrace{\frac{1}{T_u}\int_{\text{load}} |e|\,dt}_{\text{IAE}_d}
  + 2\,\underbrace{\max(0,\, y_{\max} - 1)}_{\text{overshoot}}
  + 0.01\,\underbrace{K\Big(K_p + \frac{K_d}{T_f}\Big)}_{\text{noise gain}},
$$

with J_fail = 20 for an unstable loop. The noise gain is the controller's high-frequency gain, ∣C(j∞)∣: how much sensor noise reaches the actuator.

To separate the reward from the learning algorithm, find the *exact* optimum of each partial cost with Nelder–Mead, adding one term at a time:

| Cost | K_p | K_i | K_d | Overshoot | IAE_r | IAE_d | Noise gain |
|---|---|---|---|---|---|---|---|
| Set-point IAE only | 4.22 | 1.70 | 5.45 | 16.6% | 0.262 | 0.090 | 58.3 |
| + load IAE | 5.88 | 4.03 | 7.66 | **34.9%** | 0.279 | 0.047 | **81.9** |
| + overshoot | 2.78 | 1.06 | 3.23 | 1.0% | 0.295 | 0.137 | 34.8 |
| + noise gain (full J) | **2.20** | **0.85** | **1.74** | **0.4%** | 0.342 | 0.164 | **19.5** |
| ZN classic, for reference | 4.80 | 2.65 | 2.18 | 46% | | | 26.4 |

Each missing term is exploited. With set-point IAE alone the "optimal" loop overshoots 17% and has more than twice ZN's noise gain. Adding load rejection makes it *worse* on both counts, because aggressive integral action rejects loads well. Only the overshoot term tames the step, and only the noise term stops the derivative gain from growing. The full-J controller has 0.4% overshoot and a lower noise gain than ZN; its cost is **0.708** against ZN classic's 1.768.

This is the RL version of the old control proverb that you get the closed loop you specify, and the same lesson as the quadrotor reward of Chapter 14. **If a learned controller looks wrong, check the reward first.**

### 15.2.1 The landscape has a cliff

A slice of J over (K_p, K_i) shows two features that shape every algorithm below. The useful region is roughly log-shaped and the scales differ wildly, so every agent acts on **log-gains**, θ = log K, which also rules out negative gains. And there is a **stability cliff**: beyond a boundary J jumps to a flat J_fail, with no gradient pointing back. About **13%** of a log-uniform box around the optimum is unstable, and ZN classic sits close to the edge. A tuner that steps over the cliff is lost; a real plant pushed over it may be damaged.

## 15.3 Three ways to cast PID tuning as RL

"Use RL to tune a PID" can mean very different MDPs. Be explicit before writing any learning code:

| | **A. One plant, one shot** | **B. A learned tuning rule** | **C. Iterative tuning** |
|---|---|---|---|
| State | None (a bandit) | Plant features from a relay and a step test | Gains so far, metrics of the last experiment, plant features, budget used |
| Action | log(K_p, K_i, K_d) | Log-multipliers on the ZN gains | A *change* in the log-multipliers, clipped to ±0.4 |
| Reward | −J | log(J_ZN / J) | log(J_prev / J_new) |
| Episode | One experiment | One experiment on a random plant | Six experiments on one plant |
| Algorithm | REINFORCE in NumPy | Policy gradient with a critic | PPO |
| Analogy | Trial and error | A better ZN table | An engineer at the panel |

A and B are **contextual bandits**: each episode is one decision, so there is no credit assignment through time. C is a genuine multi-step MDP, because early experiments change what the agent knows when it makes later ones. A fourth family, in which RL replaces or schedules the PID at every sample, is continuous control with all the sim-to-real issues of Chapter 14, and is not covered here.

## 15.4 Formulation A: REINFORCE on one plant

**Policy.** A diagonal Gaussian over log-gains, θ ~ 𝒩(μ, diag σ²): six parameters. One episode samples θ, runs one experiment and returns R = −J(e^θ).

**Gradient.** The score-function identity ∇E[R] = E[R ∇ log π(θ)] needs no derivative of R, which is essential here: J is flat on the unstable plateau and has a kink at zero overshoot. Writing θ = μ + σ ⊙ ε,

$$
\nabla_\mu \log\pi = \frac{\varepsilon}{\sigma},\qquad \nabla_{\log\sigma}\log\pi = \varepsilon^2 - 1.
$$

Subtract the batch-mean reward as a baseline and divide by the batch standard deviation, so the advantages A_i are scale-free; then take an Adam step on

$$
\hat g_\mu = \frac{1}{N}\sum_i A_i\, \frac{\varepsilon_i}{\sigma},\qquad \hat g_{\log\sigma} = \frac{1}{N}\sum_i A_i\,(\varepsilon_i^2 - 1).
$$

```python
eps = rng.standard_normal((N, 3))
theta = mu + np.exp(log_sigma) * eps                    # N candidate log-gain vectors
R = -cost_batch(np.exp(theta))                          # N experiments
A = (R - R.mean()) / (R.std() + 1e-8)
g_mu = (A[:, None] * eps / np.exp(log_sigma)).mean(0)
g_ls = (A[:, None] * (eps**2 - 1)).mean(0)
# gradient *ascent* on (mu, log_sigma) with Adam
```

Learning σ lets the policy decide how much to explore. This estimator is also a natural-evolution-strategy update; RL and black-box optimisation meet exactly here (Salimans et al., 2017).

**Start at ZN**: μ₀ = log K_ZN and σ₀ = 0.3 (about ±35%). Starting from a known, stable controller is the most important safety measure when the environment is a real plant.

### 15.4.1 The honest comparison

Give the same plant to Nelder–Mead started at ZN, and to pure random search over a log-uniform box, with the same budget of 2560 experiments:

| Method | Experiments to reach within 1% of the optimum | Unstable trials | Final J |
|---|---|---|---|
| REINFORCE (three seeds) | 1024, 1344, 1120 | 1, 2, 1 | 0.716–0.730 |
| **Nelder–Mead** | **78** | 0 | 0.708 |
| Random search | Never | **20%** of trials | 0.734 |

REINFORCE finds the optimum, and the result is a much better controller than ZN. But **for one fixed plant whose experiments are deterministic, a classical optimiser wins by more than an order of magnitude.** Random search, meanwhile, spends a fifth of its experiments on unstable loops; REINFORCE, exploring locally around a stable start, had one or two in 2560. On a real plant that difference decides whether a method is usable at all.

So why bother with RL? Formulation A is a stepping stone. The payoff comes when one training run produces a tuner for *many* plants.

## 15.5 A family of plants

ZN is valuable because it is a *rule*: measure two numbers, get gains, for any plant. To learn a rule, train on many plants. Keep the third-order-lag structure and randomise it:

$$
P(s) = \frac{K}{(\tau_1 s+1)(\tau_2 s+1)(\tau_3 s+1)},\qquad K \sim \log\mathcal U[0.5, 2],\quad \tau_i \sim \log\mathcal U[0.25, 4]\ \text{s}.
$$

**What the policy may see** is only what an engineer can measure before tuning: (K_u, T_u) from a relay test, and the static gain K and average residence time T_ar from an open-loop step test. The policy gets two **dimensionless** combinations, log(K K_u) (equal to log 8 for three equal lags, larger when one lag dominates) and log(T_u / T_ar). With dimensionless features, a time axis in units of T_u, and a reward as a log ratio, a rule learned on these plants is automatically scale-free, just like ZN. This is the same idea as the domain randomisation of Chapters 14 and 16, applied to a tuning rule.

**The per-plant optimum** for scoring comes from a batched cross-entropy method that tunes every test plant from scratch with about 1440 closed-loop experiments each. It is a reference, not a deployable method.

Two facts frame what follows. On 200 unseen test plants ZN classic costs on average **2.57 ×** the best achievable J, so there is a lot to gain. And the optimal correction to ZN is not constant: the optimal log-multipliers correlate with log(K K_u) at −0.94 for K_p, −1.00 for K_i and −0.76 for K_d. A better table of three constants will help; a rule that *reads the plant* should help more.

## 15.6 Formulation B: learning a tuning rule

A small network maps the two features to the mean log-multipliers on the ZN gains, a = log(K / K_ZN), with a learned state-independent σ. Parameterising the action relative to ZN keeps the output of order one for any plant, makes the policy interpretable ("how much to scale each ZN gain"), and lets the output layer be initialised at multipliers that worked on the nominal plant, so **0%** of early experiments are unstable.

The reward is r = log(J_ZN / J). A raw −J would vary between plants for reasons the policy cannot control (some plants are simply harder), and that variance would swamp the learning signal. Because there is now a context, the right baseline is state-dependent: a small critic V(s) learns E[r ∣ s], and the advantage is r − V(s).

Switching the input off (`context=False`) leaves a policy that can only learn three constants: literally a re-fitted ZN table. Both train on 32 000 simulated episodes in under ten seconds.

| Rule | Learned multipliers on ZN | J / optimum on unseen plants |
|---|---|---|
| Constant (a refitted ZN table) | K_p × 0.45, K_i × 0.29, K_d × 0.81 | 1.149 |
| **Contextual** (reads the plant) | Varies with K K_u | **1.024** |

Every multiplier is below one: ZN classic is too aggressive on every plant in the family. The integral gain needs the deepest cut, from about × 0.3 for three equal lags to about × 0.1 when one lag dominates. Once trained, the contextual rule costs one relay test and one step test per new plant, and no tuning experiments at all.

## 15.7 Formulation C: an iterative tuner trained with PPO

A rule must commit before it sees the closed loop. An engineer does better: try some gains, look at the response, adjust and repeat. Formulation C makes that an MDP:

- **State (12 numbers):** the current log-multipliers (3); the last experiment's log IAE_r, log IAE_d, overshoot, log normalised noise gain, log(J / J_ZN) and a stability flag (6); the two plant features; and the fraction of the experiment budget used.
- **Action:** a change to the three log-multipliers, clipped to ±0.4 (× 0.67 to × 1.49 per step), so the agent can only make moderate moves, as a careful engineer would.
- **Reward:** log(J_prev / J_new). The return telescopes to log(J_ZN / J_final), so maximising it means *finishing* well, while each step gets dense credit for its improvement.
- **Episode:** start at ZN classic and run six experiments.

PPO (Chapter 14) trains it on rollouts of 128 parallel tuning sessions with 8 epochs of clipped updates and GAE (λ = 0.95, γ = 1, because the horizon is short and time is in the state): 92 160 simulated experiments in about 30 s.

| After experiment | 0 (ZN) | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| J / optimum (mean over 200 unseen plants) | 2.575 | 2.046 | 1.638 | 1.371 | 1.194 | 1.070 | **1.004** |

After six experiments the PPO tuner is within half a per cent of tuning every plant from scratch with about 1440 experiments, and none of its experiments on the 200 test plants was unstable. It has learned a *procedure* ("this response overshoots and rings, so back off K_i and K_p"), not a single answer.

Look at what it does on a hard plant (K = 0.57, lags 0.26, 2.39 and 3.8 s, K K_u = 28.3). The K_i log-multiplier changes by −0.4 at **every** step, which is the action limit. The agent wants to move faster than it is allowed, so the final K_i is set by the budget, not by the agent's judgement. That is a design trade-off between safety (small moves) and speed (few experiments), and Exercise 7 explores it.

## 15.8 Head to head on 200 unseen plants

All methods are scored by the same J on the same 200 test plants, as a geometric mean of J / J_optimum (1.00 means as good as tuning each plant from scratch):

| Method | What it needs on a new plant | J / optimum | Worst plant |
|---|---|---|---|
| ZN classic | Relay test | 2.575 | 2.64 |
| ZN some overshoot | Relay test | 1.434 | 1.83 |
| ZN no overshoot | Relay test | 1.483 | 1.79 |
| RL constant rule | Relay test | 1.149 | 1.37 |
| RL contextual rule | Relay and step test | 1.024 | 1.04 |
| **RL PPO iterative tuner** | Relay, step and 6 closed-loop tests | **1.004** | 1.04 |
| Per-plant optimum (CEM) | About 1440 closed-loop tests | 1.000 | 1.00 |

The ranking is the lesson. **More information about the plant buys better gains.** A better table of constants already helps a lot; plant features help more; closed-loop feedback helps most. RL is the tool that turns that information into gains *without anyone deriving the mapping by hand*. The expensive learning happens once, in simulation, across a family of plants; each new loop on the plant floor then costs a relay test and a handful of experiments, roughly what a commissioning engineer would spend anyway.

## 15.9 Verify before you deploy

A learned controller here is still a linear PID, so it can and must be checked with classical tools. Never deploy gains judged only by their reward. Gain margin g_m, phase margin φ_m and the peak sensitivity M_s = max_ω ∣1/(1 + L(jω))∣ come straight from L = C(jω)P(jω); M_s between 1.4 and 2 is the usual robustness target.

| Controller | Gain margin | Phase margin | M_s |
|---|---|---|---|
| ZN classic | 4.51 | 27.6° | **2.43** |
| ZN some overshoot | 7.85 | 66.5° | 1.48 |
| REINFORCE | 11.37 | 69.8° | 1.37 |
| Nelder–Mead optimum | 11.16 | 69.0° | 1.38 |

ZN classic's small phase margin and high M_s are why it rings and why it sat near the cliff. The RL-tuned controller is comfortably robust, even though robustness was never written into the reward; it appeared indirectly through the overshoot and noise-gain terms. Do not count on such luck in general. If robustness matters, put M_s in the reward or reject candidates that violate it before running them (Exercise 6).

This is the great practical advantage of tuning a PID rather than learning a neural controller (Chapter 14): the learned artefact is three numbers that every classical verification tool understands. It also fits the safety architecture of Chapter 17, where the learned component stays out of the actuation path and a verifiable controller owns it.

## 15.10 The decisive experiment

The decisive experiment for a learned tuner is **a fair comparison on unseen plants against both a classical optimiser and a per-plant optimum**. Against ZN alone, every RL method here looks excellent. Against Nelder–Mead on one plant, RL needs about fourteen times as many experiments, which is the result that tells you not to use it for that job. Against the per-plant optimum on 200 unseen plants, the PPO tuner is within 0.4% in six experiments, which is the result that tells you what it *is* for. Without the two stronger baselines, neither conclusion would be available.

## 15.11 A protocol for learned tuning

1. **Write the cost first, and write all of it**: tracking, disturbance rejection, overshoot and actuator or noise effort. A missing term will be exploited.
2. **Start from ZN or any known-stable controller.** Parameterise actions relative to it and in log-gains.
3. **Handle instability explicitly.** Check closed-loop eigenvalues or margins before the experiment and charge a fixed penalty. Expect a cliff, not a slope.
4. **On one known plant, use an optimiser.** Nelder–Mead needed 78 experiments; REINFORCE about 1100.
5. **Make everything dimensionless** (time in T_u, gains relative to ZN, reward as a log ratio) so one policy can generalise across a family.
6. **Give the policy what an engineer would measure**: relay and step-test features, and in the iterative setting the metrics of the last experiment.
7. **Score on unseen plants against a per-plant optimum**, not just against ZN.
8. **Verify the final gains classically**: margins, M_s and a step test, before they touch hardware.
9. **Train in simulation, fine-tune on the plant.** On a real loop, exploration costs product, time and wear. A policy pre-trained on a randomised plant family is what keeps the on-plant budget small (Lawrence et al., 2022, needed about 40 minutes on a real tank system).

## Summary

- "RL for PID tuning" can mean a bandit on one plant, a learned tuning rule, or an iterative tuner; each has a different state, action, reward and episode.
- The cost is the specification. Without overshoot and noise-gain terms, the "optimal" loop overshot 35% with three times ZN's noise gain; with them, 0.4% overshoot and a lower noise gain than ZN.
- On one plant, Nelder–Mead reached the optimum in 78 experiments and REINFORCE in about 1100. Use an optimiser.
- On a family of plants, RL learns what an optimiser cannot: a contextual rule (1.024 × optimum with no tuning experiments) and an iterative PPO tuner (1.004 × in six experiments), against 2.57 × for ZN classic.
- More information about the plant buys better gains.
- The learned artefact is a PID, so verify it classically: the RL-tuned loop had a 70° phase margin and M_s 1.37, against ZN classic's 28° and 2.43.

## Exercises

1. **Saturation and wind-up.** Clip u to [−2, 4] in the simulator (now nonlinear, so apply the ZOH update with the saturated input). Re-run Section 15.4. Does ZN classic look better or worse relative to the optimum? Add back-calculation anti-wind-up and repeat.
2. **Derivative on measurement.** Make the D-term act on −y instead of e. Which optimal gains change, and why does the noise-gain term no longer tell the whole story?
3. **ITAE instead of IAE.** Replace both IAE terms by ∫t∣e∣dt / T_u². Predict how the optimal set-point response changes before you run it.
4. **Out-of-distribution plants.** Add a dead time L = 0.3 T_ar, or a fourth lag, to the test plants only. Which degrades more, the contextual rule or the PPO tuner? Explain using what each policy observes.
5. **Noisy experiments.** Add measurement noise to y and compute the noise term as the standard deviation of Δu. Compare Nelder–Mead and REINFORCE again with a fixed budget. Which suffers more, and why?
6. **A robustness constraint.** Reject any candidate with M_s > 1.6 before running it. How much J does the constraint cost on the nominal plant, and on the test family?
7. **Budget.** Retrain the PPO tuner with 3 and with 10 experiments per episode, and with an action limit of ±0.2 and ±0.8. Plot final J / J_opt against budget. Where do more experiments stop helping?
8. **CEM as the policy optimiser.** Replace REINFORCE in Section 15.4 by the cross-entropy method on a single plant. Compare experiments and unstable trials. When would you prefer each?

## Further reading

- J. G. Ziegler and N. B. Nichols, "Optimum settings for automatic controllers," *Trans. ASME* 64, 759–768, 1942.
- K. J. Åström and T. Hägglund, "Automatic tuning of simple regulators with specifications on phase and amplitude margins," *Automatica* 20(5), 645–651, 1984.
- K. J. Åström and T. Hägglund, *Advanced PID Control*, ISA, 2006.
- R. J. Williams, "Simple statistical gradient-following algorithms for connectionist reinforcement learning," *Machine Learning* 8, 229–256, 1992.
- T. Salimans, J. Ho, X. Chen, S. Sidor and I. Sutskever, "Evolution strategies as a scalable alternative to reinforcement learning," arXiv:1703.03864, 2017.
- R. Y. Rubinstein, "The cross-entropy method for combinatorial and continuous optimization," *Methodology and Computing in Applied Probability* 1, 127–190, 1999.
- N. P. Lawrence et al., "Deep reinforcement learning with shallow controllers: an experimental application to PID tuning," *Control Engineering Practice* 121, 105046, 2022.
- O. Dogru et al., "Reinforcement learning approach to autonomous PID tuning," *Computers & Chemical Engineering* 161, 107760, 2022.
- J. A. van Niekerk, J. D. le Roux and I. K. Craig, "Reinforcement learning based automatic tuning of PID controllers in multivariable grinding mill circuits," *Control Engineering Practice* 165, 106522, 2025.
