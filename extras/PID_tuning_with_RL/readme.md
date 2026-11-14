# RL for PID tuning on the third-order lag — deck + notebook (generated 2026-11-14)
 
Plant P(s) = 1/(s+1)^3, parallel PID C(s) = Kp + Ki/s + Kd·s (implemented with a filtered derivative
Kd·s/(Tf s+1), Tf = Tu/36 ≈ 0.10 s, fixed). Ziegler–Nichols is the baseline the user previously taught.
Audience assumed upper-undergraduate engineering (same as the series; not asked). Follows the RL
quadrotor lecture (`rl-quadrotor-lecture-notes.md`) and reuses its PPO.
 
## Deliverables
- `RL_PID_tuning_lecture.pptx` — 13 slides (dark title/summary + 11 light), notes 890–1130 chars on every
  slide, QA'd via LibreOffice render (959.98 × 540 pt).
- `RL_PID_tuning_third_order_lag.ipynb` — 54 cells, executed with nbclient, 0 errors / 0 stderr, ~2 min on
  2 CPU cores. Library code inlined from the session scripts via `ast`, so numbers match results.json exactly.
## Running example
- Test = unit set-point step at t=0, input load step d = 0.5/K at 10 Tu, stop at 20 Tu; dt = Tu/60.
- Exact ZOH simulation of the 5-state closed loop (3 lags + integrator + D-filter) via expm of the augmented
  matrix; checked against scipy.signal step of the closed-loop TF (max diff 3e-15). Instability detected by
  eig(A) before simulating → J_fail = 20.
- Cost J = IAE_r/Tu + IAE_d/Tu + 2·overshoot + 0.01·K(Kp + Kd/Tf) (noise gain = |C(j∞)|).
- Plant family: K ~ logU[0.5, 2], τ_i ~ logU[0.25, 4] s; features log(K·Ku), log(Tu/Tar) (relay + step test).
  Train pool 4000 (seed 11), test 200 (seed 209), per-plant optimum by batched CEM (30 iters × 48).
## Headline numbers (results.json)
| quantity | value |
|---|---|
| Ku / Tu | 8 / 3.628 s |
| ZN classic gains / OS / J | 4.80, 2.65, 2.18 / 46 % / 1.77 |
| optimum of full J (Nelder–Mead) | Kp 2.20, Ki 0.85, Kd 1.74; J 0.7075; OS 0.4 %; ts 5.0 s |
| reward variants (OS, noise gain) | IAE only 17 %, 58 · +load 35 %, 82 · +OS 1 %, 35 · full 0.4 %, 19.5 (ZN 26.4) |
| landscape slice unstable | 13 % |
| REINFORCE (32/iter × 80, from ZN, σ0 .3) | J 0.716/0.718/0.730 (3 seeds); within 1 % at ~1160 experiments; 1 unstable trial |
| Nelder–Mead from ZN | within 1 % at 78 experiments |
| random search 2560 | best 0.734, 20 % unstable (522) |
| margins ZN classic → RL | PM 28° → 70°, Ms 2.43 → 1.37, GM 4.5 → 11.4 |
| test J/J_opt (geo mean) | ZN classic 2.57 · ZN some-OS 1.43 · ZN no-OS 1.48 · RL constant 1.15 · RL contextual 1.024 · PPO 1.004 |
| learned constants on ZN | ×0.45, ×0.29, ×0.81 |
| PPO by experiment 0..6 | 2.57, 2.05, 1.64, 1.37, 1.19, 1.07, 1.004; 0 unstable; 92 160 sim experiments, ~31 s |
 
## Deck outline
1 Title · 2 plant + ZN (relay oscillation, 3 ZN variants) · 3 cost J anatomy · 4 reward design (4 variants) ·
5 landscape + stability cliff · 6 three formulations A/B/C schematic · 7 REINFORCE (code box + learning) ·
8 nominal result vs ZN · 9 honest price: NM 78 vs RL ~1160, unstable trials · 10 B learned tuning rule ·
11 C PPO iterative tuner · 12 scorecard on 200 unseen plants · 13 summary (workflow strip + 7 rules)
 
## Notebook sections
Objectives · env · 1 plant/ultimate point · 2 ZN · 3 simulator + TF check · 4 reward + variants ·
5 landscape · 6 formulations table · 7 REINFORCE + NM + random search · 8 plant family + CEM oracle ·
9 learned rule · 10 PPO tuner · 11 scorecard · 12 margins/Ms · 13 protocol · 14 eight exercises · 15 references.
 
## Load-bearing choices — do not "fix"
- PPO step limit must be ±0.4 (±0.25 left it at 1.06× because Ki needs ~−2.4 in log on dominant-lag plants;
  even at 0.4 it saturates on the hardest plant — taught as exercise 7).
- Plant family τ range (0.25, 4): with (0.4, 2.5) the constant rule was nearly as good as the contextual one.
- Reward as log(J_ZN/J), actions as log-multipliers on ZN, output bias init (0.45, 0.30, 0.80) → 0 unstable
  trials at training start.
- NM beating REINFORCE ~15× on one plant is the honest centrepiece of slide 9; keep it.
## Reuse notes
- Scripts (session workspace only): pid_common.py, rl_tuners.py, train_all.py (→ results.json, cache.npz,
  rule_*.pt, ppo_tuner.pt), margins.py (adds margins to results.json), make_figs.py (12 figs + aspects.json),
  build_notebook.py («key» substitution + ast extraction), build_deck.js, run_nb.py.
- Name-clash bug hit: loop variable `w` shadowed the reward-weight argument `w` in cost_batch.
- torch 2.15 from PyPI installed fine in the background (~2 min). pptxgenjs custom 13.333×7.5 layout + figureFit.
- Verified references: Lawrence et al. CEP 121:105046 (2022); Dogru et al. Comput. Chem. Eng. 161:107760
  (2022); Lawrence et al. IFAC 2020 arXiv:2005.04537; van Niekerk et al. CEP 165:106522 (2025).
- Natural follow-ons: saturation + anti-windup in the env; dead-time / OOD plants (rule vs PPO);
  RL as the controller vs PID; safe exploration with an Ms constraint.
 
