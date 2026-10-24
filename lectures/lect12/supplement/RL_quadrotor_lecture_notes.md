# Reinforcement learning for robotics (quadrotor hover) — deck + notebook (generated 2026-10-23)
 
Basics → DQN → PPO → SAC → domain randomisation, one quadrotor running example.
Audience: **upper-undergraduate engineering** (same as the series). User-chosen extras: **SAC**,
**sim-to-real / domain randomisation**, **recent-advances landscape slide** (explanatory only).
 
## Deliverables
 
- `mlpi26_lect12_supplement1.pptx` — 12 slides (dark title + dark summary, 10 light), speaker notes on every
  slide (930–1190 chars), QA'd via LibreOffice render (959.98 × 540 pt).
- `mlpi26_lect12_nbsup1.ipynb` — 44 cells (19 code), executed with nbclient, **0 errors, 0 stderr**,
  ~5.5 min on 2 CPU cores. Library code is inlined from the session scripts via `ast` block extraction,
  so notebook numbers reproduce `results.json` exactly (timings drift a few %).
## Running example
 
1. **AltitudeHold** (1-D, tabular): state (z, ż) on a 41×41 grid, 3 thrust levels {0.55, 1.0, 1.45}·mg,
   dt 0.1 s, target z=1, reward −(z−1)² − 0.05ż², crash −20 at floor/ceiling, γ 0.97.
2. **PlanarQuad** (vectorised NumPy, 50 Hz, 250-step episodes): m 0.80 kg, ℓ 0.17 m, I 0.006, drag 0.10,
   thrust-to-weight 2; action u∈[−1,1]², T_i = T_hover(1+u_i)·motor; first-order motor lag; obs 9-D
   (e_x, e_z, sinθ, cosθ, ẋ, ż, ω/4, u_prev). Reward 1 − ‖e‖ − 0.2|θ| − 0.05‖v‖ − 0.02|ω| − 0.05‖Δu‖;
   crash = ground / |x|>2.5 / |θ|>1.4; time-out is truncation (bootstrapped everywhere).
   DR: mass & inertia ×U(0.75,1.35), motor ×U(0.80,1.10), lag U(0,80 ms).
- Baseline: cascaded PD (kp 4, kd 3, kp_θ 60, kd_θ 12) using the nominal model.
- DQN: 9 actions (collective ±0.3 × differential ±0.15), MLP 128×2 tanh, 8 envs, 240k steps, replay 100k,
  batch 128, target copy every 500 updates, Huber, lr 5e-4. PPO: 16 envs × 256, 10 epochs, mb 1024, clip
  0.2, GAE 0.95, lr 3e-4 linear decay, near-zero output init, 800k steps. SAC: 4 envs, 20k steps, UTD 1,
  twin Q 128, actor 64 ReLU, auto-α.
## Headline numbers (results.json)
 
| quantity | value |
|---|---|
| value iteration | 209 sweeps, return −0.90 (always-hover −14.7) |
| tabular Q-learning, 3000 episodes | −3.17 |
| 6-D Q-table at 41 bins | 4.75e9 cells, 171 GB |
| PD | return 215.8, 100 % success |
| DQN | return 204.0, 98 % success, final err 7.0 cm, chatter 0.131, ~66 s |
| PPO | return 225.8, 100 %, 1.5 mm, chatter 0.0088 (15× less), ≥95 % at 164k steps, ~60 s, clip frac 0.03 |
| SAC | return 197.3 at 20k steps, 14 cm offset (0 % strict success) |
| steps to return 190 | SAC 18k · PPO 164k · DQN 180k |
| CPU µs / env step | SAC ~6 500 · DQN ~275 · PPO ~78 |
| crash @ 80 ms lag, nominal mass | nominal PPO **100 %** · DR PPO **0 %** · PD 0 % |
| sag @ 1.45× mass | nominal 12.1 cm · DR 6.0 cm · PD 110 cm (no integral term) |
| mean crash inside DR box | nominal 20.6 % → DR 0.1 %; outside (100 ms, 1.45×) DR still 26 % |
| nominal-quad return cost of DR | 225.8 → 224.1 |
 
## Deck outline
 
1 Title · 2 quadrotor as an MDP (loop schematic) · 3 value iteration (V* + switching-curve policy) ·
4 Q-learning + curse of dimensionality · 5 DQN schematic (replay/target) · 6 policy gradient + PPO clip ·
7 PPO training curve + clip/KL + loss code · 8 head-to-head PD/DQN/PPO (path + chatter) ·
9 SAC sample efficiency vs compute · 10 domain-randomisation crash heatmaps · 11 timeline 2013–2025 ·
12 summary (workflow strip + 7 rules)
 
## Notebook sections
 
Objectives · environment · 1 MDP + AltitudeHold · 2 Bellman/value iteration · 3 Q-learning + curse of
dimensionality · 4 planar quad + PD baseline · 5 DQN (+ time-out bug wrong/right) · 6 policy gradients,
GAE, PPO clip · 7 PPO · 8 head-to-head · 9 SAC · 10 domain randomisation · 11 recent-advances table ·
12 9-rule protocol · 13 eight exercises · 14 references.
 
## Load-bearing facts — do not "fix"
 
- **Motor lag is the sim-to-real lever**, not mass: the nominal PPO learns high-gain feedback that goes
  100 % unstable at ≥ 80 ms lag; mass mismatch only causes sag. DR training range must reach 80 ms.
- **DQN ablations (no target net, 2k buffer) made no visible difference** on this easy task — dropped
  rather than presented as a result. The DQN lesson is chatter from the discrete action set.
- **SAC is deliberately short (20k steps, ~2 min)** for runtime; its steady offset is exercise 4.
- PD "never crashes" because its gains are modest — taught as robustness ≠ precision.
- The DR mass grid must include 1.00 so the nominal cell exists (first draft used 0.95 and mislabelled it).
## Reuse notes
 
- Scripts (session workspace only): `quad_common.py`, `rl_agents.py`, `train_all.py` (→ results.json,
  cache.npz, state_dicts), `make_figs.py` (11 figs + aspects.json), `build_notebook.py` (`«key»` template
  substitution from results.json + ast extraction of functions), `build_deck.js`, `run_nb.py`.
- torch 2.14.1 from PyPI installed in ~4 min this time. `float(tensor_requiring_grad)` warns — use `.item()`.
- Matplotlib bar on `41**12` (Python int) overflows — use floats.
- Twin-axis label collides with the next panel unless `wspace≈0.55`; timeline year labels must avoid stem x
  positions (label even years, put events on odd years' stems).
- pptxgenjs: custom 13.333 × 7.5 layout, `figureFit` helper; code box with a 3-line figure slide at yBottomMax 5.75.
## Natural follow-ons
 
Model-based RL / MPC vs RL on the same quad; way-point flight and racing (exercise 8); recurrent or
history-conditioned policies for system identification (the "which quad am I flying?" question);
offline RL from logged flights; RLHF/GRPO as a separate LLM lecture.
 
# Slide Notes

## Slide 1
This slide's job: set expectations. One plant carries the whole lecture — a quadrotor that must hover — and we solve it three times with increasingly capable methods. First a one-dimensional altitude problem small enough for a lookup table, so we can see Bellman's equation work exactly. Then a planar quadrotor with six states and two rotors, which no table can hold, solved with DQN and then PPO. We finish with what every real deployment needs: making the policy survive a quadrotor that is not the one in the simulator.

Everything in the deck is computed in the notebook; every number on these slides comes out of it. Ask the room: who has tuned a PID loop on a drone? Keep those people in mind — they will be our sceptics, and the classical baseline on slide 8 is for them.

A student may ask why not start with a real drone. Answer: crashes are expensive, and almost all successful drone RL in the literature trains in simulation first.



## Slide 2
This slide's job: map control vocabulary onto RL vocabulary. Point at the loop left to right. The agent is a policy — here a small neural network — that sees a state and outputs rotor commands u1 and u2. The environment is the planar quadrotor: total thrust lifts it, the thrust difference pitches it, gravity pulls it down; the three equations are on the figure. After 20 ms it returns the next state and a reward.

The reward is one minus the position error, minus small penalties on tilt, speed, rotation rate and command changes. A crash — ground, tilt past 80 degrees, leaving the box — ends the episode, so the agent loses all future reward. That is the only "don't crash" signal it ever gets.

Question for the room: is position alone a valid state? No — two quads at the same place can be rising or falling. The Markov property means the state must predict the future on its own; that is why velocities and angular rate are in there.

Expect: "isn't the reward just a cost function?" Yes, with the sign flipped. RL is optimal control without being handed the model.



## Slide 3
This slide's job: show Bellman's equation doing real work on a problem small enough to see completely. We keep only the vertical axis: altitude and climb rate, three collective thrust levels, a 0.1 s step. Discretised into 41 by 41 cells, the whole Q-function is 1681 states times 3 actions.

Left panel: the optimal value. It is highest at the target altitude with zero velocity, and falls off towards the floor and ceiling where a crash costs minus 20. Right panel: the greedy policy. Point at the boundary between blue and orange — it is tilted, not horizontal. A vehicle below the target but already climbing fast is told to cut thrust early. That is the switching curve of a time-optimal controller, found with no control theory at all.

Numbers to read: 209 sweeps to converge; optimal return -0.90 versus -14.7 for always commanding hover — with little drag, a vehicle that starts moving keeps moving.

The catch, and the student who spots it gets credit: value iteration needed the model, the transition function. Next slide removes it.



## Slide 4
This slide's job: make the jump from planning to learning, and then show why tables die. Q-learning never sees the model. It flies the continuous simulator, bins the state to index its table, and nudges each entry towards the TD target — reward now plus the discounted best value next.

Left panel: 3000 episodes, greedy-policy return evaluated every 100. It climbs from about -16 to -3.17 but stays short of value iteration's -0.90. Two reasons: some cells are rarely visited, and binning a continuous state breaks the Markov property — two states in one cell behave differently.

Right panel, log scale: 41 bins per dimension. Two states, 1681 cells. Six states, 4.75 billion cells — 171 GB in float32 with 9 actions. Twelve states for a full 3-D quad is hopeless.

Question: what do we want instead of a table? Something that generalises between nearby states — a function approximator. That is the entire motivation for "deep" RL.



## Slide 5
This slide's job: explain the two stabilisers that made deep Q-learning work, using the diagram left to right. Eight simulated quads fly with epsilon-greedy over 9 thrust combinations and every transition goes into a 100 000-entry replay buffer. Random batches of 128 train the online network. The TD target uses a separate target network, copied from the online one every 500 updates.

Why replay: consecutive samples from one flight are nearly identical, so gradients on them are badly correlated; shuffling makes training look like supervised learning and reuses each sample many times. Why a target network: without it, every update moves the very target you are regressing towards. Function approximation, bootstrapping and off-policy data together are Sutton and Barto's "deadly triad".

Results: 240k environment steps, about 66 seconds, return 204.0 and 98 % success.

The bug almost everyone writes: treating the 5-second time limit as a crash. A time-out is an artefact of the experiment; the future still exists, so you bootstrap from the final state. Gymnasium's terminated versus truncated flags exist for exactly this.



## Slide 6
This slide's job: give the one idea behind PPO without a full derivation. Instead of learning Q and acting greedily, parameterise the policy — here a Gaussian whose mean is a network — and do gradient ascent on expected return. The policy-gradient theorem says: weight the gradient of the log-probability of each action by its advantage. Using the raw return works but is very noisy; a learned critic V gives a much better advantage estimate. That is actor-critic, and GAE with lambda 0.95 blends one-step and multi-step estimates.

The danger is step size. The gradient is only valid near the policy that collected the data. For a quadrotor one bad update means every episode crashes, and then there is nothing to learn from.

Point at the plots. Ratio r is new policy probability over old. Left, positive advantage: the objective rises with r until 1.2, then goes flat — no incentive to push further. Right, negative advantage: flat below 0.8. The dashed line is what you would optimise without clipping.

Expect: "why not a hard KL constraint?" That is TRPO; PPO gets most of the benefit with first-order optimisation.



## Slide 7
This slide's job: show PPO working and teach the two diagnostics to watch. Left panel: return of the deterministic policy, evaluated every 20 000 steps, with success rate in orange. It passes 95 percent success after about 164k steps and ends at 225.8, slightly above the cascaded PD controller's 215.8. Be careful how you say this: the PD gains were hand-picked, not optimised for this reward. The honest claim is that RL matches a competent classical design without being given the model.

Right panel: the clip fraction — the share of samples whose ratio left the 0.8 to 1.2 band — and the approximate KL between old and new policy. Clip fraction averages 0.030, KL stays below 0.014. If the clip fraction sits above about 0.3, or KL spikes, the learning rate or epoch count is too aggressive.

The code box is the whole objective in three lines. Everything else in a PPO implementation is plumbing — and the plumbing matters: Huang et al. catalogue 37 details. Two that bit us: initialise the output layer near zero so the first policy commands hover, and bootstrap through time-outs in GAE.



## Slide 8
This slide's job: show that "it reaches the target" is not the whole story for a robot. Left panel: the paths. PD in green is smooth and direct, because it was designed with the model. PPO in blue is close to it. DQN in orange gets there too.

Right panel is the point: rotor-1 command over five seconds. PPO settles to a nearly constant command, as hover should look. DQN never stops switching between its three levels, because hovering exactly requires a thrust it cannot command. Mean command change per step is 0.131 for DQN versus 0.009 for PPO, about 15 times more, and it leaves a small limit cycle with median error 7.0 cm versus 0.1 cm.

On hardware that switching is heat in the ESCs, audible motor noise and excitation of frame vibration modes. The fix is not a better DQN; it is the right action space.

Ask the room: when would discrete actions be right? A gripper open or closed, gear selection, choosing among motion primitives. Exercise 3 tries a 5-by-5 action set.



## Slide 9
This slide's job: one of the most practical decisions in robot RL — on-policy or off-policy — made with numbers. SAC, soft actor-critic, is off-policy like DQN but with a continuous squashed-Gaussian actor. It learns two Q-networks and uses the smaller to fight over-estimation, and it maximises reward plus policy entropy, which keeps exploring without a hand-tuned noise schedule.

Left panel, log x-axis: SAC passes a return of 190 after about 18k environment steps, PPO after about 164k, DQN after about 180k. Right panel: the price. Each SAC step costs about 6599 microseconds of CPU, against 80 for PPO, because of the per-step gradient updates.

So the choice depends on what is expensive. Learning on a real drone, where a sample is a second of flight and a battery — SAC. A fast parallel simulator, where samples are nearly free — PPO, which is why nearly every large sim-to-real result uses it.

Honest caveat: SAC here ran only 20 000 steps; it flies well (return 197.3) but keeps a 14.0 cm offset. Exercise 4 finishes the job.



## Slide 10
This slide's job: the reality gap, demonstrated, and its standard fix. Each cell is 200 episodes on a quad with a different mass and motor time constant; the number is the percentage that crash. The blue square, nominal mass and no lag, is the simulator the nominal policy was trained on.

Left: PPO trained on the nominal quad. Perfect at home, and with slow motors it falls apart — 100.0 % crashes at 80 ms lag. It learned high-gain feedback that is only stable without actuator delay. Middle: the same PPO trained with randomisation, dashed box. Crash rate at the same point 0.0 %; sag under a 1.45-times battery drops from 12.1 to 6.0 cm. On the nominal quad it gives up almost nothing.

Two things to say out loud. First, randomisation only covers what you randomise: at 100 ms lag and 1.45-times mass, outside the box, the DR policy still crashes 27 percent of the time. Second, the PD controller on the right never crashes on this grid because its gains are modest — but it sags 110.2 cm under the heavy battery because it has no integral term. Robustness and precision are different properties.



## Slide 11
This slide's job: place today's methods on a timeline and connect them to quadrotors. Top row, algorithms: DQN on Atari, 2013, in Nature in 2015; TRPO and then PPO in 2017; SAC in 2018; Dreamer in 2020, which learns a world model and trains the policy inside its own imagination, and DreamerV3 in 2023. RLHF in 2022 is PPO fine-tuning a language model against a learned reward model of human preferences — the same algorithm you just saw hover a quadrotor. In 2025, GRPO in DeepSeek-R1 dropped the critic and used verifiable rewards to train reasoning.

Bottom row, robotics: Hwangbo et al. 2017 stabilised a real quadrotor with a neural policy, even recovering from a hand throw. Molchanov 2019 transferred one randomised policy to several different quadrotors — today's slide 10 in miniature. Rudin 2021 trained walking in minutes with thousands of GPU simulations. Swift in 2023 beat human world champions in drone racing on a real track; its simulator was corrected with residual models learned from real flight data.

Discussion question: Song et al. found RL beat optimal control in racing. Why? Because RL can optimise the true objective, lap time, directly instead of a tracking proxy.



## Slide 12
This slide's job: leave the room with a workflow, not just algorithm names. Walk the strip left to right. Write the MDP first: state, action, reward, and which endings are crashes and which are time-outs. Build a classical baseline — today the cascaded PD — so you know what good looks like and catch a broken environment. Pick the algorithm by the action space and by what is expensive. Randomise the simulator. Evaluate on dynamics you did not train on, and report robotics metrics.

The seven rules repeat today's evidence. Value iteration, Q-learning, DQN, PPO and SAC are all approximations to Bellman's equation. The 171 GB table is why we need networks. DQN's chatter is why action spaces matter. PPO's wall-clock advantage and SAC's sample advantage decide which to use. The nominal policy crashing 100.0 % of the time with slow motors is why we randomise.

Point students to the notebook: every number in this deck is produced there, with eight exercises — integral action, gusts, way-point flight, and the time-out bug. Closing question to leave them with: what would you have to add to the observation to let the policy tell which quad it is flying?



