# Chapter 13. Reinforcement Learning Foundations

*Part IV. Learning to act*

> **Case studies:** (A) an AGV finding its way to a loading dock; (B) a quadrotor holding altitude
> **Companion notebook:** [`lectures/lect12/supplement/mlpi26_lect12_nbsup1.ipynb`](../../lectures/lect12/supplement/mlpi26_lect12_nbsup1.ipynb) (Sections 1 to 3)
> **Lab:** [`labs/lab12/mlpi26_Lect12_Lab.ipynb`](../../labs/lab12/mlpi26_Lect12_Lab.ipynb) (Q-learning on the factory floor, then a DQN cart-pole)
> **Built from:** Lecture 12 deck, Lecture 12 Supplement 1 and its speaker notes, Lab 12

Every model in this book so far was handed the right answer. The defect images had labels, the tool had a measured wear value, the part had a gauged diameter. Reinforcement learning (RL) removes the answer key. An agent acts, the world changes, and now and then a scalar **reward** says how well things are going. Nobody says which action was right; the agent only learns that some sequence of actions eventually paid off, and it has to work backwards from there.

Control engineers already design policies. An LQR gives a state-feedback law that minimises a quadratic cost, derived from a model of the plant. RL is optimal control without being handed the model. That is its power and, as this chapter and the next will show repeatedly, the reason it should not be your first choice when you *do* have a model.

This chapter covers the foundations on problems small enough to see completely: the Markov decision process, Bellman's equation, value iteration, Q-learning and exploration. It ends where tables run out. Chapter 14 replaces the table with a network and flies a quadrotor.

## Learning objectives

After this chapter you should be able to:

1. Write a robot task as a Markov decision process: state, action, reward, discount and termination.
2. Explain the action-value function and read a controller off a Q-table.
3. State the Bellman optimality equation and solve a small MDP by value iteration.
4. Implement tabular Q-learning, including the terminal-state case.
5. Explain where exploration comes from (ε-greedy and optimistic initialisation) and diagnose an agent that has stopped exploring.
6. Show how actuation uncertainty changes the optimal policy, and test whether a learned policy is better or merely different.
7. Explain the curse of dimensionality and why it forces function approximation.

---

## 13.1 A different kind of learning

| | Supervised | Unsupervised | Reinforcement |
|---|---|---|---|
| You are given | Inputs and their labels | Inputs only | A world that responds to actions |
| You learn | The mapping | The structure | A policy |

The central difficulty is **credit assignment**. If an AGV reaches the dock after 18 moves, which of the 18 moves deserves the credit? If it hits a forklift on move 9, was move 9 the mistake, or was it move 3, which committed it to a narrow gap? Every algorithm in this chapter is a way of passing the news of a reward backwards to the decisions that caused it.

## 13.2 The Markov decision process

Every RL problem is written down the same way. Getting this framing right is most of the engineering work.

A **Markov decision process** (MDP) is a tuple (S, A, p, r, γ):

| Element | Meaning | AGV on the factory floor | Altitude-holding quadrotor |
|---|---|---|---|
| States S | Where the system can be | The 63 cells of a 7 × 9 floor | Altitude z and climb rate ż |
| Actions A | What the controller may do | Up, down, left, right | Thrust 0.55, 1.0 or 1.45 × mg |
| Dynamics p(s′ ∣ s, a) | How the world responds | Move one cell (or slip) | Newton's law, 0.1 s step |
| Reward r(s, a) | What we actually want | +10 dock, −10 forklift aisle, −0.1 per move | −(z − 1)² − 0.05ż²; −20 for floor or ceiling |
| Discount γ | How much the future counts | 0.95 | 0.97 |

The agent's **policy** π(a ∣ s) maps states to actions. Its goal is to maximise the expected **discounted return**

$$
G_t = \sum_{k \ge 0} \gamma^k\, r_{t+k}.
$$

A useful rule of thumb: 1/(1 − γ) is the effective horizon in steps. With γ = 0.97 and a 0.1 s step, the altitude controller looks about 3 s ahead; with γ = 0.99 at 50 Hz (Chapter 14), about 2 s.

**The reward is the entire specification.** For the AGV, that is all we write: no route, no waypoints, no clearance rule. Whatever behaviour maximises this reward is what we will get, including behaviour we did not intend. Writing the reward badly is the most common way RL projects fail.

**"Markov" means the current state is enough** to predict the future. Position alone is not a Markov state for a quadrotor: two vehicles at the same altitude can be climbing or falling. If your state leaves out something that matters (payload, temperature, wheel wear), no RL algorithm can recover it. Choosing the state is the first engineering decision, and it is yours, not the algorithm's.

**Terminal or not.** An episode can end because something final happened (the AGV reached the dock, the quadrotor hit the floor) or because the experiment ran out of time. The two must be treated differently; Section 13.5 and Chapter 14 return to this.

## 13.3 Value functions and the Bellman equation

Two functions measure how good things are under a policy π:

$$
V^\pi(s) = \mathbb E_\pi[G_t \mid s_t = s], \qquad Q^\pi(s, a) = \mathbb E_\pi[G_t \mid s_t = s, a_t = a].
$$

Q(s, a) is the total future reward you expect if you take action a in state s and then act well afterwards. **If you know Q, you have a controller**: in every state take the action with the largest value, π(s) = argmax_a Q(s, a).

For the AGV, Q is a table of 63 states × 4 actions. You can print it, plot it and argue with it, which is more than can be said of most learned controllers. It is also a trap: 63 states is a toy, and Section 13.7 shows how quickly tables become impossible.

The optimal action values satisfy the **Bellman optimality equation**:

$$
Q^*(s, a) = \mathbb E\Big[\, r + \gamma \max_{a'} Q^*(s', a') \;\Big|\; s, a \Big].
$$

In words: the value of an action is the reward it earns now plus the discounted value of acting optimally from wherever it lands you. Every method in this chapter and the next is an approximation to this one equation.

### 13.3.1 Value iteration: Bellman with a model

If the dynamics are known, apply the Bellman equation as an update to every state at once and repeat until nothing changes. This is **value iteration**, dynamic programming in the sense of Bellman.

The altitude-hold quadrotor is small enough to solve exactly. Discretise altitude (0 to 2 m) and climb rate (±2 m/s) into a 41 × 41 grid: 1681 states × 3 actions. Value iteration converges in **209 sweeps**.

| Policy | Return |
|---|---|
| Always command hover thrust | −14.7 |
| Value-iteration optimum | **−0.90** |

Hover thrust does badly because there is little drag: a vehicle that starts with any velocity keeps it and drifts out of the safe band.

Read the optimal policy map like a control engineer. The boundary between "high thrust" and "low thrust" is not the horizontal line z = 1. It is tilted. A vehicle below the target but already climbing fast is told to cut thrust *before* it arrives. That is the switching curve of a bang–bang, time-optimal controller, and value iteration found it with no control theory, only the model and the reward.

The catch is in the first sentence of this section: value iteration needs p(s′ ∣ s, a). Real robots rarely hand it to us.

## 13.4 Q-learning: Bellman without a model

**Temporal-difference (TD) learning** replaces the expectation in Bellman's equation with a single sampled transition (s, a, r, s′):

$$
Q(s, a) \leftarrow Q(s, a) + \alpha\Big[\underbrace{r + \gamma \max_{a'} Q(s', a')}_{\text{the new estimate}} - \underbrace{Q(s, a)}_{\text{the old estimate}}\Big].
$$

The bracket is the **TD error**: the gap between what you expected and what you got. Move the old estimate part of the way toward the new one, by the learning rate α, and repeat. That is the whole algorithm (Watkins, 1989).

```python
def q_update(Q, s, a, r, s2, done, alpha=0.1, gamma=0.95):
    target = r if done else r + gamma * Q[s2].max()   # a terminal state has no future
    Q[s, a] += alpha * (target - Q[s, a])
```

**The terminal-state rule.** At a terminal state there is no future, so the target is just r. Forget this and the agent keeps crediting a dead state with future value. It is the classic Q-learning bug, and it reappears in every deep RL algorithm in Chapter 14 as the `(1 − done)` factor.

Q-learning is **off-policy**: the target uses the greedy max, whatever action the agent actually takes next. That means the agent can explore freely while still learning about the greedy policy.

### 13.4.1 Q-learning on the altitude hold

Give the altitude problem to Q-learning. The agent never sees the model; it flies the *continuous* simulator and only bins the state to index its table. Over 3000 episodes (about 10 s of CPU), the greedy policy's return climbs from about −16 to **−3.17**, short of value iteration's −0.90. There are two reasons, both general. Some cells are rarely visited. And **binning a continuous state breaks the Markov property**: two states that fall in the same cell behave differently, so the table has to average them.

### 13.4.2 Q-learning on the factory floor

Lab 12 is the AGV case. The 7 × 9 floor has two routes to the dock: a **narrow doorway** straight up the middle, about 10 moves, running between two hazard cells; and a **long detour** around the end of the forklift aisle, about 18 moves. With perfect wheels, Q-learning finds the doorway and earns a return of **9.10** with 100% success.

Plot the greedy arrows and max_a Q for every cell and you can read the controller directly. Before trusting the picture, look away from the route. Out in the corners are arrows pointing nowhere useful. Those cells were barely visited, so their values were updated a handful of times and mean very little. **A learned policy is only valid where it has data.** Push the robot into a state its training never reached and you are reading numbers that were never learned. Chapter 16 is largely about this.

## 13.5 Where does exploration come from?

Act greedily and you keep collecting the reward you already know about. Explore and you pay now, in the hope of finding something better. The standard remedy is **ε-greedy**: act greedily with probability 1 − ε and at random with probability ε.

The textbook claim is that with ε = 0 an agent gets stuck on the first adequate route it finds. Lab 12 tests it over eight seeds, under two initialisations of the Q-table:

| Initial Q | ε | Success |
|---|---|---|
| 0 | 0 | 100% |
| 0 | 0.2 | 100% |
| −5 | 0 | **0%** |
| −5 | 0.2 | 100% |

Read the first two rows first. The greedy agent does as well as the exploring one. That is not a mistake. With zero initial values and a −0.1 cost per move, every action the agent has tried drops below zero, while every action it has not tried still sits at zero. Untried actions look better than tried ones, so the agent explores without being told to. This is **optimistic initialisation** (Sutton and Barto, 2018, Section 2.6), a legitimate exploration strategy.

Now the third row. With pessimistic initial values the optimism is gone, and ε = 0 collapses to 0% success. Where does the agent end up? It navigates the doorway correctly, arrives at the top wall **four cells from the dock**, and then pushes into the wall for the rest of the episode. Bumping a wall costs only −0.1 and leaves the AGV where it is, so that action's value settles at the fixed point of q = −0.1 + 0.95q, which is **−2**. Every action it never tried is still at −5. By its own estimates, pressing uselessly against a wall is better than anything unexplored. It is not stuck because it is confused. It is stuck because it is *confident, and wrong*.

The lesson is not "always use ε-greedy". It is that **exploration must come from somewhere**: ε-greedy, optimistic initial values, an entropy bonus (Chapter 14), or something else. Remove every source and learning stops, sometimes within sight of the goal.

## 13.6 Case study: the route depends on the wheels

Now the experiment that matters for robotics. Train the same agent twice, for 6000 episodes each: once on a floor where the AGV moves exactly where it aims, and once where it slips sideways 20% of the time. Same grid, same rewards, same algorithm.

| Trained with | Route learned | Return |
|---|---|---|
| No slip | Doorway (≈ 10 moves) | 9.10 |
| 20% slip | **Detour (≈ 18 moves)** | 7.72 |

Nobody wrote a clearance rule. The agent inferred one from the reward: once you might not go where you aimed, squeezing through a gap between two hazards stops being worth it.

### 13.6.1 Is the detour better, or merely different?

That is a fair challenge, and it can be answered by measurement. Hand-code a controller that drives straight up the doorway and then right to the dock, and run it 500 times on the slipping floor:

| Policy | Wheel slip | Return | Success |
|---|---|---|---|
| Doorway (forced) | None | 9.10 | 100% |
| Doorway (forced) | 20% | 4.41 | **78%** |
| Learned detour | 20% | **7.72** | 100% |

Forced through the doorway on slipping wheels, the AGV drives into the forklift aisle in **22%** of runs. The detour is not timid; it has the higher expected return.

This is the engineering lesson of the chapter. A path planner that assumes perfect actuation will happily route a real robot into a wall. **Uncertainty in actuation changes the optimal policy**, and an RL agent trained under that uncertainty finds the change by itself. Chapter 14 makes the same point on a quadrotor with motor lag, where the stakes are a 100% crash rate.

## 13.7 When the table runs out

The Q-table worked because the floor had 63 states. Count cells for the same 41 bins per dimension:

| System | State dimensions | Cells | Q-table (float32, 9 actions) |
|---|---|---|---|
| Altitude hold | 2 | 1.68 × 10³ | 60 kB |
| Planar quadrotor | 6 | 4.75 × 10⁹ | **171 GB** |
| Full 3-D quadrotor | 12 | 2.26 × 10¹⁹ | 8 × 10¹¹ GB |

This is Bellman's **curse of dimensionality**. Even a crude 20 bins on each of a cart-pole's four continuous variables gives 160 000 states, most of which the agent will never visit. Worse, Section 13.4.1 showed that binning breaks the Markov property.

What we want instead is something that **generalises between nearby states**: a function approximator. Replace the table with a network Q_θ(s, a) and you have a **Deep Q-Network**. The update rule does not change at all; only how Q is stored.

### 13.7.1 A first DQN: the cart-pole

The last part of Lab 12 builds one on a cart-pole you can read the dynamics of (no external packages): a 4–64–64–2 ReLU network trained on the TD target. On its own a neural Q-function is unstable, and two additions fix it. Both address the same thing: *the target must hold still*.

- A **replay buffer** stores transitions and trains on random past samples. Consecutive steps from one episode are highly correlated, and training on them in order makes the network chase its own tail.
- A **target network**, a frozen copy synced every 50 episodes in the lab, computes y = r + γ(1 − done) max_a′ Q_target(s′, a′). Without it the target moves every time the network updates, and training can diverge. The (1 − done) is the terminal-state rule of Section 13.4, written as arithmetic.

The measured result, on a laptop CPU in **29 s**:

| | Steps survived (mean) |
|---|---|
| First 50 episodes | 21.1 |
| Last 50 episodes | **143.4** |

That is about a sevenfold improvement. Read the curve honestly, though. It is flat until episode 150, exactly where ε finishes annealing (ε = max(0.05, 1 − episode/150)): until then the agent is acting mostly at random, so even a good network cannot show what it knows. Then it climbs and plateaus well short of the 500-step ceiling. Training for 500 episodes instead of 300 gives the same plateau. Plain DQN needs more care to master even this plant (Double DQN, n-step returns, a tuned target-sync rate), which is where Chapter 14 begins.

## 13.8 RL and classical control: an honest comparison

| | Classical control (LQR, MPC) | Reinforcement learning |
|---|---|---|
| Needs a model | Yes; that is the starting point | No; only interaction |
| Needs data | Almost none | A great deal |
| Stability guarantees | Yes, provable | Essentially none |
| Contact, friction, things you cannot model | Handled poorly | Where it earns its keep |
| Cart-pole | An afternoon, with a proof | 29 s of training, still not mastered |

Use RL where you have no model and data is cheap, typically through a simulator. If you can write the dynamics down, write them down: you will get a better controller faster. Chapter 14 makes this comparison again on a quadrotor, where a hand-tuned cascaded PD controller is the baseline every RL method has to beat.

## 13.9 The decisive experiment

For an RL agent the decisive experiment is to **change the world and see whether the policy changes for the right reason**, and then to force the old policy into the new world and measure what it costs. On the factory floor, adding 20% wheel slip turned a 10-move doorway route into an 18-move detour, and forcing the doorway under slip crashed the AGV in 22% of runs. The learned policy was not merely different; it had a higher expected return. The same experiment also guards against the opposite error. Had the forced doorway done as well as the detour, the "clearance rule" would have been a story we told about an arbitrary choice.

## Summary

- RL learns a policy from reward alone. The reward is the entire specification, and the state must be Markov.
- If you know Q, you have a controller. The Bellman equation defines Q*; value iteration solves it with a model (altitude hold: −0.90 in 209 sweeps, against −14.7 for always hovering) and found a time-optimal switching curve unaided.
- Q-learning is one update rule: close the gap between what you expected and what you got. At a terminal state the target is just r.
- Exploration must come from somewhere. Zero initial values were already optimistic; with pessimistic values and ε = 0 the AGV froze, confident and wrong, four cells from the dock.
- Uncertainty changes the optimal policy: with 20% slip the AGV chose the detour, and forcing the doorway crashed it in 22% of runs.
- A learned policy is only valid where it has data.
- Tables cannot scale (171 GB for a planar quadrotor). Replace the table with a network, and stabilise it with replay and a target network.

## Exercises

1. **Discount and horizon.** Re-run value iteration on the altitude hold with γ = 0.8 and γ = 0.995. How does the switching curve change, and why? Relate 1/(1 − γ) to a time horizon at a 0.1 s step.
2. **The terminal-state bug.** Remove the terminal case from the Q-learning update so that the target is always r + γ max Q(s′, ·). Train on the factory floor and plot max_a Q at the dock and at the hazard cells. What happens to the learned route?
3. **How much slip?** Train with slip probabilities 0, 0.05, 0.1, 0.15 and 0.2. At what slip does the learned route switch from doorway to detour? Compute the forced-doorway and detour returns at each slip and check that the switch happens where they cross.
4. **Optimism, measured.** With ε = 0, sweep the initial Q from +2 to −5. Plot success rate and number of episodes to first reach the dock. Where does optimistic initialisation stop working, and why does the step cost set that point?
5. **Coverage map.** Count visits per state during training and overlay them on the greedy-policy map. Mark the states whose greedy action is wrong. How do the two maps relate?
6. **Binning and the Markov property.** Repeat the altitude-hold Q-learning with 11, 21, 41 and 81 bins per dimension. Plot final return against bins and training episodes. Explain the trade-off.
7. **A better cart-pole DQN.** Starting from the lab DQN, add Double DQN and 3-step returns one at a time, with three seeds each. Which one moves the plateau? (The DQN deep dive in `extras/` measures this.)
8. **Write the MDP.** For a pick-and-place arm sorting parts into two bins, write down states, actions, reward, termination and truncation. Name one variable that, if omitted, makes the state non-Markov.

## Further reading

- R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018. Chapters 3, 4 and 6.
- R. Bellman, *Dynamic Programming*, Princeton University Press, 1957.
- C. J. C. H. Watkins and P. Dayan, "Q-learning," *Machine Learning* 8, 279–292, 1992.
- V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature* 518, 529–533, 2015.
- D. P. Bertsekas, *Reinforcement Learning and Optimal Control*, Athena Scientific, 2019. A control engineer's view of the same material.
- `extras/DQN_deep_dive/`: replay, target networks, Double, dueling, n-step and prioritised replay on CartPole and a quadrotor.
