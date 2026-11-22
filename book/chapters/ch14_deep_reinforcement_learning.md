# Chapter 14. Deep Reinforcement Learning for Robot Control

*Part IV. Learning to act*

> **Case study:** teaching a planar quadrotor to hover, with DQN, PPO and SAC, and then making it survive a quadrotor that is not the one in the simulator
> **Companion notebook:** [`lectures/lect12/supplement/mlpi26_lect12_nbsup1.ipynb`](../../lectures/lect12/supplement/mlpi26_lect12_nbsup1.ipynb) (Sections 4 to 13)
> **Deep dives:** [`extras/DQN_deep_dive/`](../../extras/DQN_deep_dive/), [`extras/PPO_deep_dive/`](../../extras/PPO_deep_dive/), [`extras/SAC_deep_dive/`](../../extras/SAC_deep_dive/)
> **Built from:** Lecture 12 Supplement 1 and its speaker notes, and the three algorithm deep dives

Chapter 13 ended with a 171 GB Q-table and a cart-pole DQN that plateaued at 143 steps. This chapter takes the same ideas to a plant with six continuous states and two rotors. It solves it three times, with a value method (DQN), an on-policy policy-gradient method (PPO) and an off-policy actor-critic (SAC), and compares them against a cascaded PD controller designed the classical way. Then it does what every real deployment needs: it tests the learned policy on quadrotors with heavier batteries and slower motors, and fixes what breaks.

Three themes run through the chapter. **The action space is a modelling choice**, and choosing it badly shows up as chatter, not as a lower score. **What is expensive decides the algorithm**: samples or compute. And **a policy is only as robust as the simulator it was trained in**, which domain randomisation addresses.

## Learning objectives

After this chapter you should be able to:

1. Formulate a quadrotor hover task with an observation, action, reward, termination and truncation that make RL work.
2. Explain the deadly triad and the two stabilisers that make DQN train, and avoid the time-out bootstrap bug.
3. Derive the policy-gradient idea, explain actor-critic and GAE, and say what PPO's clipped objective protects against.
4. Read the PPO and SAC training diagnostics (clip fraction, KL, entropy, explained variance).
5. Choose between on-policy and off-policy learning from measured sample and compute costs.
6. Report robotics metrics (crash rate, final error, chatter, robustness grid) alongside return.
7. Use domain randomisation to close a sim-to-real gap, and state its limits.

---

## 14.1 The plant and the baseline

### 14.1.1 The planar quadrotor as an MDP

Two rotors at arm length ℓ produce thrusts T₁ and T₂ on a body of mass m and pitch inertia I:

$$
\ddot x = -\frac{T_1+T_2}{m}\sin\theta - c\,\dot x,\qquad
\ddot z = \frac{T_1+T_2}{m}\cos\theta - g - c\,\dot z,\qquad
\ddot\theta = \frac{\ell\,(T_2-T_1)}{I}.
$$

| Design choice | Value | Why |
|---|---|---|
| Parameters | m = 0.80 kg, ℓ = 0.17 m, I = 0.006 kg m², thrust-to-weight 2 | A small 250-class quad |
| Control rate | 50 Hz, 250-step episodes (5 s) | |
| Action | u_i ∈ [−1, 1], T_i = T_hover(1 + u_i), with a first-order motor lag | u = 0 is hover, so the network output is well scaled |
| Observation (9-D) | e_x, e_z, sin θ, cos θ, ẋ, ż, θ̇/4, previous u | sin and cos avoid the wrap at ±π; velocities make it Markov |
| Reward | 1 − ‖e‖ − 0.2∣θ∣ − 0.05‖v‖ − 0.02∣θ̇∣ − 0.05‖Δu‖ | An alive bonus, a tracking term and a smoothness penalty |
| Termination | Ground, ∣x∣ > 2.5 m, ∣θ∣ > 1.4 rad | A crash ends the reward stream: the only "don't crash" signal |
| Time limit | 250 steps | A **truncation**, not a terminal state (Section 14.2.2) |

Each reward term is a gain you are choosing. Change the smoothness weight and the policy trades precision for quieter motors (Exercise 5). **Reward shaping is controller design.**

The simulator steps N quadrotors at once in vectorised NumPy, which is what makes training on a CPU practical: every number in this chapter comes from a notebook that runs in about six to eight minutes on two CPU cores.

### 14.1.2 A classical baseline first

Before training anything, write the controller a controls course would: a **cascaded PD** loop from position error to desired acceleration, to desired tilt, to rotor thrusts, using the nominal model (gains k_p = 4, k_d = 3, k_pθ = 60, k_dθ = 12).

| Controller | Return | Success | Crash | Final error | Chatter |
|---|---|---|---|---|---|
| Do nothing (u = 0) | −20.1 | 0% | 100% | | |
| Cascaded PD | 215.8 | 100% | 0% | 0.04 cm | 0.001 |

Commanding hover thrust is not enough: the random initial tilt and velocity are never corrected, so every episode crashes. The PD controller tells you what "good" looks like, and if RL cannot approach it, the problem is the setup, not the controller. Note the columns. **Return** is what the agent optimises, but crash rate, final error (mean distance to target over the last second; success means within 10 cm) and **chatter** (mean ∣Δu∣ per step) are what a flight-test engineer reports.

## 14.2 Deep Q-networks on the quadrotor

### 14.2.1 The deadly triad and its two stabilisers

DQN (Mnih et al., 2015) replaces the table with a network Q_θ(s, ·) that outputs one value per discrete action, and minimises the Huber TD error. Done naively it diverges, because three things are true at once: function approximation, bootstrapping and off-policy data. Sutton and Barto call this the **deadly triad**. DQN adds the two stabilisers from Section 13.7.1:

- **Experience replay.** Here 8 quads fly ε-greedily and every transition goes into a 100 000-entry buffer; random mini-batches of 128 train the network. Shuffling makes the gradient look like supervised learning and reuses each sample many times.
- **Target network.** The TD target uses a frozen copy refreshed every 500 updates, so the regression target does not move every time θ does.

The DQN deep dive in `extras/` removes each stabiliser on CartPole and shows what goes wrong. With no target network, the Q-values climb to about 2.6 × 10⁷ while the return never leaves about 9: the deadly triad, live. Copying the target only every 1000 updates makes Q crawl up in visible stairs. A 1000-transition buffer holds about four episodes, and the network overfits recent behaviour and forgets.

DQN needs a finite action set. Here it is collective thrust {−0.3, 0, +0.3} × differential {−0.15, 0, +0.15}: nine rotor-command pairs.

### 14.2.2 The time-limit bug almost everyone writes

An episode can end for two reasons, and they must be treated differently in the target:

```python
# WRONG: treats the 5 s time limit as if the quad had crashed
done = crash | timeout
y = r + gamma * (1 - done) * Q_target(s_next).max()

# RIGHT: a time-out is an artefact of the experiment; the future still exists
y = r + gamma * (1 - crash) * Q_target(s_final).max()   # s_final: the state before the auto-reset
```

With the wrong version the agent learns that states near the end of an episode are worth less, for no physical reason. Gymnasium's `step()` returns separate `terminated` and `truncated` flags precisely for this. The same rule applies to PPO's advantage estimate and SAC's critic. Store the final observation *before* the environment auto-resets.

### 14.2.3 Result: DQN flies, and chatters

A 128 × 128 tanh network (18 953 parameters), 240 000 environment steps, about 66 s:

| | Return | Success | Final error | Chatter |
|---|---|---|---|---|
| DQN | 204.0 | 98% | 7.0 cm | **0.131** |

DQN learns to fly. But with only three thrust levels per channel, it can hover only by switching between them, which shows up as a small limit cycle around the target. Section 14.5 shows what that means on hardware.

**Beyond vanilla DQN.** Four extensions, combined in Rainbow (Hessel et al., 2018), address known weaknesses. **Double DQN** chooses the next action with the online network and values it with the target network, which removes the upward bias of a max over noisy estimates (with 15 actions that bias is +1.7σ). **Dueling heads** split Q into a state value and per-action advantages. **n-step returns** carry real rewards back n steps at once. **Prioritised replay** samples transitions with large TD error more often. On CartPole, with three seeds each, all four together solved the task on every seed while vanilla solved none; n-step returns were the largest single contributor, and a dueling head alone made things *worse*. Double DQN did not help by removing overestimation, because on CartPole Q sat *below* the realised return throughout training. The general lesson, which the deep dives repeat: **ablate one change at a time over several seeds before believing it.**

## 14.3 Policy gradients and PPO

### 14.3.1 Learn the policy directly

Instead of learning values and acting greedily, parameterise the policy π_θ(a ∣ s), here a Gaussian whose mean is a network, and do gradient ascent on the expected return. The **policy-gradient theorem** gives

$$
\nabla_\theta J(\theta) = \mathbb E_{\pi_\theta}\big[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\, A_t\big]:
$$

increase the log-probability of actions that turned out better than expected. Using the raw return in place of the **advantage** A_t (REINFORCE) is unbiased but very noisy. An **actor-critic** learns a critic V_φ alongside the actor and estimates the advantage with **generalised advantage estimation** (GAE; Schulman et al., 2016):

$$
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t), \qquad \hat A_t = \sum_{k \ge 0} (\gamma\lambda)^k\, \delta_{t+k}.
$$

λ trades bias (λ = 0: one-step TD) against variance (λ = 1: Monte Carlo). In the PPO deep dive, λ = 0 solved only one of three pendulum seeds, because a young critic's bias went straight into the policy; λ = 0.95 was best.

### 14.3.2 Why PPO clips

A policy gradient is only valid near the policy that collected the data. Take too large a step and the policy can collapse. For a quadrotor, one bad update means every episode crashes, and then there is no good data left to recover with. **Proximal policy optimisation** (PPO; Schulman et al., 2017) reuses each batch for several epochs but removes the incentive to move far:

$$
L^{\mathrm{CLIP}}(\theta) = \mathbb E\Big[\min\big(\rho_t \hat A_t,\; \operatorname{clip}(\rho_t,\, 1-\epsilon,\, 1+\epsilon)\, \hat A_t\big)\Big],\qquad
\rho_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}.
$$

With a positive advantage, the objective rises with ρ until 1 + ε, then goes flat: no incentive to push further. With a negative advantage it goes flat below 1 − ε. Because of the min, the clip only ever removes *optimism*. In code, the whole objective is three lines:

```python
ratio = torch.exp(new_logp - old_logp)
surr = torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv)
loss = -surr.mean() + 0.5 * (returns - value).pow(2).mean()
```

TRPO (Schulman et al., 2015) enforces a hard KL constraint with a second-order method. PPO gets most of the benefit with first-order code. On the pendulum, ten unclipped epochs per batch reached a single-update KL of 9.1 and a final return of −1009; ten clipped epochs reached −175 with the KL never above 0.02. Clipping, an adaptive KL penalty and TRPO all reached the same place at the same speed.

### 14.3.3 PPO on the quadrotor

Everything else in a PPO implementation is plumbing, and the plumbing matters; Huang et al. (2022) catalogue 37 details. The configuration here: 16 parallel quads × 256 steps per batch, 10 epochs of mini-batch 1024, clip 0.2, GAE λ = 0.95, learning rate 3 × 10⁻⁴ decaying linearly, advantage normalisation, gradient-norm clipping at 0.5, a state-independent log standard deviation, and 800 000 steps. Two details bit during development: **initialise the output layer near zero**, so the first policy commands hover instead of a random tilt; and **bootstrap through time-outs** inside the GAE loop.

```python
class ActorCritic(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.mu = mlp(OBS_DIM, ACT_DIM, h)
        self.v = mlp(OBS_DIM, 1, h)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), -0.5))
        with torch.no_grad():                       # first policy commands hover
            self.mu[-1].weight.mul_(0.01); self.mu[-1].bias.zero_()
```

| | Return | Success | Final error | Chatter | Parameters | Training time |
|---|---|---|---|---|---|---|
| Cascaded PD | 215.8 | 100% | 0.04 cm | 0.001 | | |
| PPO | **225.8** | 100% | 0.15 cm | 0.009 | 9797 (actor + critic) | ≈ 60 s |

PPO passes 95% success after about 164 000 steps. Be careful how you say the result. On *this reward*, PPO beats PD, but the PD gains were hand-picked, not optimised for this reward. The honest claim is that **RL matches a competent classical design without being given the model.** The classical controller still wins on analysability and guarantees.

### 14.3.4 Reading the diagnostics

The return curve tells you *that* a run failed, not why. Watch two more signals. The **clip fraction**, the share of samples whose ratio left [0.8, 1.2], averaged 0.030 here: the optimiser pushes against the clip only occasionally, the healthy regime. The **approximate KL** between old and new policy stayed below 0.014.

The PPO deep dive shows why these matter. An "aggressive" quadrotor run (30 epochs, ε = 0.4, learning rate 10⁻³, no annealing), every choice of which looks like "learn faster", reached a return of 190 after 40 000 steps against 245 000 for the healthy run, and peaked at 218. Anyone watching only the return would keep those settings. Around 614 000 steps it fell apart and ended with a 100% crash rate. The other panels had seen it coming from the first update: KL above 0.05 and climbing past 1, a clip fraction of about 0.64, and a policy standard deviation that collapsed to 0.014 (healthy: 0.28).

| Signal | Healthy range (rule of thumb) | Too high means | Too low means |
|---|---|---|---|
| Approximate KL per update | 0.002–0.03 | Steps too big: lower the learning rate, fewer epochs, smaller ε | Learning has stalled |
| Clip fraction | 0.05–0.25 | Same as high KL | The clip is never active |
| Policy std or entropy | Decays slowly | Slow convergence | Premature collapse |
| Explained variance of the critic | Rising toward 0.9+ | | Critic not learning: scale rewards, check bootstrapping |

## 14.4 Off-policy actor-critic: SAC

PPO throws each batch away after ten epochs because its gradient is only valid for the policy that collected the data: it is **on-policy**. **Soft actor-critic** (SAC; Haarnoja et al., 2018) is off-policy, like DQN, but with a continuous actor. It keeps a replay buffer, learns **twin Q-networks** and uses the smaller in the target to fight overestimation (the TD3 remedy, Fujimoto et al., 2018), and maximises reward *plus entropy*,

$$
J(\pi) = \mathbb E\Big[\sum_t \gamma^t \big(r_t + \alpha\, \mathcal H(\pi(\cdot \mid s_t))\big)\Big],
$$

with the temperature α tuned automatically toward a target entropy. The entropy term keeps exploring without a hand-tuned noise schedule. The actor is a tanh-squashed Gaussian, so actions stay in [−1, 1], and its log-probability needs the change-of-variables correction. Forget the correction and, in the SAC deep dive, the entropy estimate sits far above target, the temperature is driven to zero, 96% of motor commands saturate at ±1, and the quad flies bang–bang with a chatter of 101 per second. SAC does one gradient update per environment step, which is the source of both its sample efficiency and its compute cost.

With 4 parallel quads and only 20 000 environment steps (about 2 minutes), SAC reaches a return of **197.3** and does not crash, but it keeps a **14 cm** steady offset, so its strict success score is 0%. It was trained for 40 times fewer steps than PPO and has not yet removed the offset (Exercise 4). The SAC deep dive trains longer and compares SAC with TD3 and DDPG over three seeds: SAC hovered at 5.2 cm against about 13 cm for the other two, with a chatter of 1.2 per second against 4.1 (TD3) and 14.0 (DDPG). Its per-seed results ranged from about 1 cm to 13 cm, so report the spread, not just the mean.

## 14.5 Head to head

### 14.5.1 Precision and chatter

Start the quad 1 m to the left, 0.6 m low and tilted 0.3 rad, and compare:

| Controller | Return | Success | Crash | Final error | Chatter (mean ∣Δu∣ per step) |
|---|---|---|---|---|---|
| Cascaded PD | 215.8 | 100% | 0% | 0.04 cm | 0.0013 |
| DQN (9 actions) | 204.0 | 98% | 0.5% | 7.0 cm | **0.131** |
| PPO | 225.8 | 100% | 0% | 0.15 cm | 0.0088 |
| SAC (20 k steps) | 197.3 | 0% (14 cm offset) | 0% | 14 cm | 0.011 |

All three learned controllers reach the target, and their paths look similar. The rotor command does not. PPO settles to a nearly constant command, as hover should look. DQN never stops switching between its three levels, because hovering exactly needs a thrust it cannot command: about **15 times** PPO's chatter. On hardware that is heat in the ESCs, audible motor noise and excitation of frame vibration modes. A return gap of 22 points does not show it.

The fix is not a better DQN; it is the right action space. Discretise only when the actuator really is discrete: a gripper open or closed, a gear, a choice among motion primitives. (The DQN deep dive confirms that a finer 15-action menu and a slower decision rate trade chatter against precision, but never remove it.)

### 14.5.2 Samples or compute?

| | Environment steps to reach return 190 | CPU µs per environment step |
|---|---|---|
| SAC | **≈ 18 000** | ≈ 6 500 |
| PPO | ≈ 164 000 | **≈ 78** |
| DQN | ≈ 180 000 | ≈ 275 |

SAC needs about nine times fewer samples than PPO and about eighty times more compute per sample. The choice depends on **what is expensive**:

| What is scarce | Prefer | Because |
|---|---|---|
| Real-robot time (learning on hardware) | SAC or another off-policy method | Reuses every sample many times |
| Wall-clock time with a fast, parallel simulator | PPO or another on-policy method | Cheap updates; scales to thousands of environments |
| A discrete actuator | The DQN family | No actor needed |

That is why nearly every large sim-to-real result uses PPO, and why learning directly on a robot, where a sample is a second of flight and a battery, favours SAC.

> **Sidebar: the tools you will actually use**
>
> The notebook writes its own environment so that nothing needs installing and every line can be read. In practice:
>
> - **Gymnasium** (the maintained successor of OpenAI Gym) defines the standard `reset()` and `step()` interface, with `terminated` and `truncated` returned separately, and vectorised environments.
> - **Stable-Baselines3** provides tested PPO, SAC, TD3 and DQN implementations; use them before writing your own, and use your own only to understand them.
> - **MuJoCo** (open source since 2022) is the standard physics engine for contact-rich robot simulation. **Isaac Lab** on NVIDIA Isaac Sim, and MuJoCo's own GPU backend MJX, run thousands of environments in parallel on one GPU, which is what made "a walking policy in minutes" possible (Rudin et al., 2021).
> - Drone-specific simulators such as gym-pybullet-drones and Flightmare wrap quadrotor dynamics in the Gymnasium interface.
>
> Whatever you use, keep the classical baseline and the robotics metrics. A library implementation removes plumbing bugs, not modelling ones.

## 14.6 Case study: the quadrotor that is not the one in the simulator

A policy trained on one simulator meets a different quadrotor: a heavier battery, motors with slower response, ESC latency. Test the nominal PPO policy on a grid of masses (0.8 to 1.45 × nominal) and motor time constants (0 to 100 ms), 200 episodes per cell, and count crashes. Then retrain PPO with **domain randomisation** (DR; Tobin et al., 2017; Molchanov et al., 2019): every episode draws mass and inertia from 0.75 to 1.35 ×, motor strength from 0.80 to 1.10 × and motor lag from 0 to 80 ms. The policy cannot tell which quad it is flying, so it must find one behaviour that works for all of them.

| | Nominal PPO | DR PPO | Cascaded PD |
|---|---|---|---|
| Crash rate at 80 ms lag, nominal mass | **100%** | **0%** | 0% |
| Sag at 1.45 × mass, no lag | 12.1 cm | 6.0 cm | **110 cm** |
| Mean crash rate inside the DR range | 20.6% | 0.1% | 0% |
| Crash rate at 100 ms lag and 1.45 × mass (outside the range) | | 26% | |
| Return on the nominal quad | 225.8 | 224.1 | 215.8 |

The nominal policy is excellent on the quad it was trained on and fragile everywhere else. It learned high-gain feedback that is only stable without actuator delay. **Motor lag is the sim-to-real lever here, not mass**: mass mismatch only causes sag, while lag causes crashes. The randomised policy crashes in 0% of episodes at the same point and halves the sag, at a cost of 1.7 points of return on the nominal quad.

Two things to say out loud:

- **Randomisation covers only what you randomise.** Outside the training box, at 100 ms lag and 1.45 × mass, the DR policy still crashes 26% of the time. Randomise wider than your uncertainty, and evaluate outside the training range.
- **Robustness and precision are different properties.** The fixed-gain PD controller never crashes on this grid because its gains are modest, but it sags 110 cm under the heavy battery because it has no integral term. Measure both.

Chapter 16 returns to the reality gap in general, with residual models learned from real data, system identification and monitoring after deployment.

## 14.7 The decisive experiment

For a learned controller, the decisive experiment is **the robustness grid**: evaluate on dynamics you did not train on, cell by cell, and count crashes. Return on the training simulator ranked the nominal and randomised PPO policies almost equal (225.8 against 224.1). The grid separated them completely: 100% against 0% crashes with 80 ms of motor lag. It also showed where each controller fails (the DR policy outside its box, the PD controller in steady-state error), which no single number can.

## 14.8 Where this is going

| Year | Development | Why it matters for robots |
|---|---|---|
| 2015 | DQN (Mnih et al., *Nature*) | Deep function approximation made stable |
| 2015, 2017 | TRPO, PPO | PPO is the default for simulation-trained locomotion and flight |
| 2017 | Hwangbo et al., *RA-L* | A neural policy stabilises a real quadrotor, even from a hand throw |
| 2018 | SAC | Sample-efficient continuous control; used for learning on hardware |
| 2019 | Molchanov et al. | One randomised policy transfers to several different quadrotors |
| 2020–23 | Dreamer to DreamerV3 | Learn a **world model** and train the policy inside it |
| 2021 | Rudin et al. | Thousands of parallel GPU simulations: a walking policy in minutes |
| 2022 | RLHF (InstructGPT) | PPO fine-tunes language models against a learned model of human preferences |
| 2023 | Swift (Kaufmann et al., *Nature*) | A deep-RL drone racer beats human world champions on a real track |
| 2023 | Song et al., *Science Robotics* | In racing, RL beats optimal control by optimising the true objective (lap time) directly |
| 2025 | DeepSeek-R1 (GRPO) | RL with verifiable rewards trains reasoning in language models; a critic-free descendant of PPO |

The common thread for robotics: **train in simulation at scale, randomise what you cannot model, and correct with residual models learned from real data.** Swift, for example, learned its policy in simulation and corrected the simulator with data-driven residual models of perception and dynamics collected on the real track.

## 14.9 A protocol for robot RL

1. **Write the MDP before the code.** The state must be Markov, the actions must match the actuator, and you must know which endings are terminal and which are truncations.
2. **Build a classical baseline first** (PD, LQR, MPC). It tells you what good is and catches broken environments.
3. **Reward shaping is controller design.** Every term is a gain you are choosing.
4. **Discrete actions on a continuous actuator cause chatter.** Use PPO or SAC for thrust and torque.
5. **Bootstrap through time limits.** `terminated` and `truncated` are different.
6. **Watch the diagnostics**, not only the return: clip fraction and KL for PPO, Q-values and α for SAC, crash rate and command smoothness for the robot.
7. **Pick on-policy or off-policy by what is expensive.** Simulator throughput favours PPO; real-robot time favours SAC.
8. **Randomise the simulator wider than your uncertainty**, and evaluate outside the training range.
9. **Report robotics metrics** (crash rate, steady-state error, chatter, robustness grid) next to return.

## Summary

- DQN, PPO and SAC are all approximations to Bellman's equation. DQN needs replay and a target network to tame the deadly triad, and every method needs time-outs bootstrapped.
- On the planar quadrotor, PPO (return 225.8, 0.15 cm, 100% success) matched a competent cascaded PD controller (215.8) without being given the model. DQN flew (204.0, 98%) but chattered 15 times more because its action set was discrete.
- PPO's clip keeps each update near the data-collecting policy. Clip fraction and KL warn of a collapse long before the return does.
- SAC reached return 190 in about 18 000 steps against PPO's 164 000, at about 80 times the compute per step. Choose by what is expensive.
- The nominal PPO policy crashed 100% of the time with 80 ms of motor lag. Domain randomisation reduced that to 0% at a cost of 1.7 points of nominal return, but only inside the randomised range.
- Robustness and precision are different properties. Report both.

## Exercises

1. **The time-out bug.** Train DQN with `bootstrap_timeout=False`. Compare return and final error, and plot max_a Q against time-in-episode for both agents.
2. **A finer DQN action set.** Use 5 × 5 instead of 3 × 3 actions. Does chatter fall? What happens to training time and success? Where does this approach break down for a four-rotor vehicle?
3. **Watch a collapse.** Train PPO with 30 epochs, ε = 0.4 and a learning rate of 10⁻³ with no annealing. Log clip fraction, KL and policy standard deviation, and find the earliest update at which they predict the collapse.
4. **Finish SAC.** Train SAC for 60 000 steps and compare final error with PPO. Then try two gradient updates per step. Plot return against wall-clock seconds rather than steps. Does the ranking change?
5. **Reward shaping.** Remove the 0.05‖Δu‖ smoothness term and retrain PPO. Measure chatter. Then double it. What does the policy trade away?
6. **Integral action.** Append the clipped running integral of e_z to the observation. Does the DR policy's sag under a 1.45 × mass disappear? Explain in control terms.
7. **Wind gusts.** Add a slowly varying horizontal force (an Ornstein–Uhlenbeck process) to `PlanarQuad.step`. Train with and without gusts and evaluate both in gusty air.
8. **Way-point flight.** Put a random target in the observation instead of a fixed hover point and train PPO to fly to it. What must change in the reward and the termination rules?
9. **Which quad am I flying?** Give the policy the last ten observations and actions (or an LSTM from Chapter 11) and retrain with DR. Does it recover precision lost to randomisation? This is implicit system identification.

## Further reading

- R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018.
- V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature* 518, 529–533, 2015.
- H. van Hasselt, A. Guez and D. Silver, "Deep reinforcement learning with double Q-learning," AAAI 2016.
- M. Hessel et al., "Rainbow: combining improvements in deep reinforcement learning," AAAI 2018.
- J. Schulman, S. Levine, P. Moritz, M. Jordan and P. Abbeel, "Trust region policy optimization," ICML 2015.
- J. Schulman et al., "High-dimensional continuous control using generalized advantage estimation," ICLR 2016.
- J. Schulman, F. Wolski, P. Dhariwal, A. Radford and O. Klimov, "Proximal policy optimization algorithms," arXiv:1707.06347, 2017.
- S. Huang et al., "The 37 implementation details of proximal policy optimization," ICLR Blog Track, 2022.
- T. Haarnoja, A. Zhou, P. Abbeel and S. Levine, "Soft actor-critic," ICML 2018.
- S. Fujimoto, H. van Hoof and D. Meger, "Addressing function approximation error in actor-critic methods" (TD3), ICML 2018.
- J. Hwangbo, I. Sa, R. Siegwart and M. Hutter, "Control of a quadrotor with reinforcement learning," *IEEE RA-L* 2(4), 2017.
- J. Tobin et al., "Domain randomization for transferring deep neural networks from simulation to the real world," IROS 2017.
- A. Molchanov et al., "Sim-to-(multi)-real: transfer of low-level robust control policies to multiple quadrotors," IROS 2019.
- N. Rudin, D. Hoeller, P. Reist and M. Hutter, "Learning to walk in minutes using massively parallel deep reinforcement learning," CoRL 2021.
- E. Kaufmann et al., "Champion-level drone racing using deep reinforcement learning," *Nature* 620, 982–987, 2023.
- Y. Song, A. Romero, M. Müller, V. Koltun and D. Scaramuzza, "Reaching the limit in autonomous racing: optimal control versus reinforcement learning," *Science Robotics* 8(82), 2023.
- Gymnasium <https://gymnasium.farama.org/>, Stable-Baselines3 <https://stable-baselines3.readthedocs.io/>, MuJoCo <https://mujoco.org/>.
