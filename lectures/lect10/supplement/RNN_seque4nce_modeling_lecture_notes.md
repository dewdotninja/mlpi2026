# Modeling sequential data with RNNs — deck + notebook (generated 2026-10-09)
 
Source: Raschka, Liu & Mirjalili, *Machine Learning with PyTorch and Scikit-Learn*, **ch. 15**
"Modeling Sequential Data Using Recurrent Neural Networks" (pp. 499–538 of the book, PDF pages
530–569 of the full-book scan). Follows the ch. 14 lecture
(`cnn-image-classification-lecture-notes.md`) in the same series.
 
## Deliverables
 
- `mlpi26_lect10_supplement1.pptx` — 14 slides (dark title + dark summary, 12 light
  content slides), speaker notes on **every** slide (1350–2125 characters each), QA'd via a
  LibreOffice render at 90 dpi. Page size 959.98 × 540 pt.
- `mlpi26_lect10_nbsup1.ipynb` — 53 cells (27 code), executed end to end with nbformat + nbclient,
  all outputs embedded, **zero errors and zero stderr warnings**, ~5 min of training inside a
  ~5 min 45 s total runtime on 2 CPU cores.
Audience level: **upper-undergraduate engineering** (same as the previous four lectures).
Scope decisions by the user: replace both of the chapter's projects (IMDb via `torchtext`,
which PyTorch deprecated and archived in 2024, and the Project Gutenberg text generator) with
one synthetic in-domain dataset; **add** a GRU comparison, a dilated 1-D CNN baseline, and an
attention preview.
 
## Running example
 
**A CNC turning cell.** One sequence per machined part: 4 analogue channels
(`spindle_current` A, `vibration_rms` g, `spindle_temp` °C, `feed_rate` norm) plus a
categorical `op_code` (5 G-code block types) at 50 Hz. Cycles are **110–150 samples**, so
batches are ragged and `pad_sequence` + `pack_padded_sequence` are motivated by the data
rather than by sentence length. Target: the finished-diameter class, `undersize` /
`in tolerance` / `oversize`.
 
### The error budget — this is the load-bearing design decision
 
```python
delta_um = -CHIP_UM * chip + DEFL_UM * heavy + N(0, SIGMA_UM)
CHIP_UM = DEFL_UM = 21.0,  SIGMA_UM = 4.0,  TOL_UM = 12.0
```
 
A **hard inclusion** at t ≈ 25 micro-chips the edge → the part comes out **undersize**.
A **heavy finishing pass** at t ≈ 100 deflects the tool → **oversize**. The two errors have
opposite signs and **cancel**, so the label is the XOR of an event and a load level
~103 samples apart. Marginals: `chip` and `heavy` are each 50 % and each is informative for
the *signed* 3-class target, which is what makes it learnable at all (see below).
 
Phase layout `approach .09 / rough .40 / dwell .05 / finish .34 / retract .12`; nominal levels
in `LEVEL`, with `FINISH_HEAVY = (8.6, 1.15, 0.30)` replacing `finish` when `heavy`.
The inclusion raises **current and vibration together** (`vib += U(2.4, 3.2)`); 1–3 benign
coolant surges raise **current only** (`vib += U(0, 0.22)`), so a vibration threshold
identifies the event and **detection is trivial — only the memory is hard**. Nothing after the
inclusion carries a trace of it; that is stated in the notebook as a deliberate modelling
choice.
 
Models: `nn.Embedding(5, 4)` for the op code concatenated to the 4 standardised channels →
`RNN/LSTM/GRU(8, 48, batch_first=True)` on a packed sequence → `Linear(48,32)-ReLU-Linear(32,3)`.
Adam `lr=3e-3`, `CosineAnnealingLR(T_max=22)`, batch 64, **22 epochs**, `clip_grad_norm_(1.0)`,
`CrossEntropyLoss`. Splits 800 / 300 / 600 (seeds 7 / 108 / 209).
 
## Headline numbers (deck and notebook agree exactly — both read `results.json`)
 
| model | params | train | valid | **test** | secs |
|---|---|---|---|---|---|
| vanilla RNN | 4 471 | 0.527 | 0.477 | **0.4767** | 20 |
| LSTM, default init | 12 823 | 0.527 | 0.477 | **0.4767** | 53 |
| **LSTM, gates open** | 12 823 | 0.994 | 0.997 | **0.9800** | 52 |
| **GRU, gates open** | 10 039 | 0.994 | 0.997 | **0.9817** | 44 |
| bidirectional LSTM, gates open | 25 495 | 0.994 | 0.993 | 0.9783 | 106 |
| dilated 1-D CNN (RF 255) | 20 599 | 0.934 | 0.953 | 0.9233 | 9 |
 
| quantity | value |
|---|---|
| majority class (test) | **0.477** |
| **late-window control** (last 45 % only, same capable model) | **0.5067** — chance |
| manual forward pass vs `nn.RNN` | max abs diff **5.96e-08** (float32 eps) |
| gradient reach, 130 steps, normalised (10 / 50 / 100 back) | RNN **5.6e-03 / 4.4e-13 / 0.0** · LSTM default **2.2e-03 / 2.8e-11 / 4.3e-21** · LSTM gates open **2.1e-02 / 2.2e-02 / 3.8e-02** |
| spectral sweep, ρ(W_hh) → grad norm | 0.33→0.45 · 0.99→0.56 · 1.21→0.65 · 1.43→**13.3** · 1.65→**2.8e+04** |
| clipping demo (W_hh × 2.6) | raw norm max **88.6**, median **12.4**, 100 % of steps rescaled to 1.0 |
| forget-gate bias → f at init → f¹⁰³ | 0 → 0.500 → **9.9e-32** · 1 → 0.731 → 9.7e-15 · **2 → 0.881 → 2.1e-06** · 3 → 0.953 → 6.7e-03 |
| best model per fault cell | neither **100 %** · inclusion only **96.1 %** · heavy only **97.1 %** · both **100 %** |
| best model as a scrap/no-scrap decision | **98.2 %** |
| packing demo (4 real of 10 padded) | `hidden[-1]` differs by **0.356** with vs without packing |
| load-token LM | 2 177 chunks, loss 2.944 → **1.586**, perplexity **4.9 of 48 bins** |
| recurrent-layer params (8→48) | RNN 2 784 · GRU 8 352 · LSTM 11 136 |
 
## Pedagogically load-bearing facts — do not "fix" these
 
- **The 3-class signed target is what makes the task learnable.** An earlier draft used a
  binary scrap label, i.e. a pure XOR of two latent features. XOR has **zero marginal
  correlation** with either factor, so there is no gradient until both are discovered
  simultaneously; every model sat at the majority class for 40 epochs. Switching the target to
  the *signed* error (undersize / in tolerance / oversize) gives each fault a marginal signal
  while keeping the scrap decision an XOR. **Do not revert to a binary label.**
- **The "LSTM with default init also fails" result is the centrepiece and it is robust** —
  verified at seeds 1, 2 and 3 (0.4767, 0.4883, 0.4767). It is what makes the forget-gate-bias
  slide land.
- **lr matters a lot for the gates-open fix.** At `lr=3e-3` the gates-open LSTM reaches 98 %;
  at `lr=1e-2` with the same bias it collapses back to chance. The window is narrow — if a
  future edit changes the optimiser, re-check.
- **The bidirectional LSTM genuinely does not win**, and the reason is worth keeping: each
  direction has one easy end and one hard end (the forward pass must remember the early event,
  the backward pass must remember the late one). It is not a bug in the setup.
- **The vanilla RNN *vanishes*, it does not explode.** Default init puts ρ(W_hh) ≈ 0.55. To
  demonstrate exploding gradients at all you must inflate `W_hh`; the threshold is sharp
  (nothing at ×2.2, 2.8e+04 at ×3.0). An earlier two-run clipping comparison was confounded —
  the unclipped run saturates its tanh and ends up with *smaller* gradients than the clipped
  one. The honest demo is one run: raw norm vs `min(raw, clip)`.
## Deck outline
 
1. Title (dark)
2. Two faults, 100 samples apart, that cancel — three example cycles + error bars
3. Pick the shape before you pick the model — the four categories + code box
4. An RNN is one extra edge — unfolding schematic + the equations
5. The forward pass, written out by hand — manual vs `nn.RNN` bars + code box
6. BPTT multiplies Jacobians, and products misbehave — w^(t−k) + measured spectral sweep
7. How far back does the gradient actually reach? — the three decay curves
8. The LSTM adds a path with no matrix on it — gate schematic + equations
9. Initialise the forget gate open — 47.7 vs 98.0 + the sigmoid + the 4-line fix
10. Ragged batches: pad, then pack — padding schematic + wrong/right code box
11. Six models, one dataset, 22 epochs each — curves + bars
12. Prove the information is where you think — the ablation + per-cell breakdown
13. Many-to-many: generating a cycle — three temperatures + the attention bridge
14. What to take away (dark) — workflow strip + 7 rules
## Notebook sections
 
Environment · what makes data sequential (+ the five shapes table) · the data (inline
generator, the 2×2 fault table) · the RNN dataflow and the manual forward pass · BPTT, w^(t−k),
the measured spectral sweep · **gradient reach, measured** · clipping and TBPTT (with the
"never truncate shorter than the dependency" point) · LSTM gates and GRU · **initialise the
gates open** · padding, packing and embeddings (with a measured packing-vs-not comparison) ·
project one: six models + per-cell breakdown + confusion matrix · the late-window ablation ·
project two: 48-bin load-token LM, Categorical sampling, temperature · attention preview
(untrained weights, mechanism only) · the 10-rule protocol · **8 exercises** · references
(including a note on the chapter's dead `torchtext` dependency).
 
## Notes for reuse
 
- Scripts live only in the session workspace: `cnc_common.py` (generator), `train_all.py` (all
  analyses → `results.json` + `analysis_cache.pt`), `make_figs.py` (13 figures, loads the cache
  so layout iterations never retrain), `build_notebook.py`, `build_deck.js`.
- **Pickling gotcha:** `torch.save` of models defined in a script run as `__main__` writes
  `__main__.SeqClassifier` into the pickle, so a *different* loader script cannot read it. Fix
  used: in the loader, `import __main__; setattr(__main__, name, getattr(train_all, name))`
  before `torch.load`. Saving `state_dict`s instead would avoid it entirely — worth doing next
  time.
- **Figure gotcha, new this time:** `plt.subplots(2, 3, sharex=True)` across rows with
  *incompatible* x-ranges (sample index 0–150 on top, µm −34…34 below) silently sets one xlim
  for all and `bbox_inches="tight"` then produced a **29 468 × 854 px** figure. Never share an
  axis across rows that plot different quantities.
- Only **2 CPU cores**. The six-model suite is ~5 min; the BiLSTM alone is 106 s. The
  gradient-reach measurement is only 1.6 s and is the highest value-per-second cell in the
  notebook.
- `nn.LSTM` gate biases are ordered `[input, forget, cell, output]` and `nn.GRU`'s are
  `[reset, update, new]` — the memory gate is the **same `[h:2h]` slice in both**, which is why
  one `open_gates_` helper covers both.
- pptxgenjs gotcha, fifth lecture running: `LAYOUT_16x9` is 10 × 5.625 in — always
  `defineLayout({name, width: 13.333, height: 7.5})`. `figureFit(s, name, yTop, yBottomMax,
  maxW)` reused again; for a figure + 4-line code box set `yBottomMax ≈ 5.70`, and for a figure
  + 7-line code box ≈ 4.90.
- Palette identical to the previous four decks (blue `#2a78d6`, orange `#eb6834`, aqua
  `#1baf7a`, yellow `#eda100`, violet `#4a3aa7`, red `#e34948`, ink `#0b0b0b`/`#52514e`,
  surface `#fcfcfb`), light content slides, dark title/summary (`#141413`, accent `#3987e5`).
## Natural follow-on lectures
 
**Chapter 16, transformers and attention** — already seeded: slide 13 and notebook §13 show
the recurrent bottleneck and a one-query attention read-out with untrained weights, so the next
lecture can open by learning those scores. Also available from this same dataset: synchronised
many-to-many (per-sample in-cut segmentation, notebook exercise 6), remaining-useful-life
regression, and cost-sensitive decision rules (exercise 8 — an oversize part shipped costs
~50× a scrapped good one).
 
# Slide Notes

## Slide 1
Opening slide. Set up two things before any mathematics.

First, what changes. Every model so far assumed the training examples were independent and identically distributed — shuffle them and nothing is lost. A sequence breaks that by definition: sample 25 of a machining cycle means something different from sample 100. Convolution gave us a model that respects locality in space; recurrence gives us one that respects order in time.

Second, the example. A CNC turning cell logs spindle current, vibration, temperature, feed and the G-code block type for every part it makes. Then the part is gauged. That is real process monitoring and it is the single most common source of industrial sequence data.

Flag the thing that makes this lecture work, on the title slide, because everything hangs off it: the label depends on an event about sample 25 and a load level about sample 100. Roughly 103 samples apart. By slide 7 we will have measured that a vanilla RNN's gradient is exactly zero at that distance, and by slide 9 we will have fixed it with one line.

Question to open with: 'you have a sensor trace and a pass/fail label. Why not just feed the whole trace to a dense network?' Hold the answers — variable length is the obvious one, parameter count is the next, and the one they will miss is that a dense net has no reason to treat sample 25 and sample 26 as related.



## Slide 2
Spend real time here. Every later slide refers back to this picture.

Walk the top row. Blue is spindle current, orange is vibration scaled by four. The blue band is the finishing pass; the orange band is the inclusion event. In the left panel the inclusion is obvious — current and vibration jump together around sample 30. In the middle panel there is no inclusion, but look at the finishing pass: it sits at about 9 amps instead of 5. In the right panel both are present.

Now the bottom row, which is the payoff. Inclusion only: minus 23 microns, undersize, scrap. Heavy finish only: plus 30 microns, oversize, scrap. Both: plus 1 micron, in tolerance, ship it. The two faults have opposite signs and they cancel.

Say why that is physically honest, because someone will ask. A chipped edge rubs instead of cutting, the workpiece heats, and after it cools the feature is undersize. A heavy cut deflects the tool away from the work and leaves the feature oversize. Error budgets with terms of opposite sign are completely ordinary in precision machining — ISO 230-3 is largely about one of them. What is unusual here is only that the two terms are a hundred samples apart.

One more design note worth stating: the inclusion raises current AND vibration, while the benign coolant surges scattered through the cycle raise only current. So detecting the event is a threshold on vibration — trivial. Remembering it is the hard part, and that is deliberate.

Question to the room: 'what is the best accuracy you could get from the last third of the cycle alone?' Let them argue. Slide 12 measures it: 50.7 percent, which is chance.



## Slide 3
A short orientation slide. The point is that these four pictures are not taxonomy for its own sake — each one corresponds to a different line of code.

Many-to-one: a whole sequence in, one label out. Our scrap classifier. You read the final hidden state and throw the rest away. Sentiment analysis is the textbook example; ours is a quality call on a finished part.

One-to-many: a fixed vector in, a sequence out. Image captioning in the book; here it would be generating an expected cycle from a part number, which is how you build a reference trace for anomaly detection.

Many-to-many synchronised: one output per input, aligned. A per-sample in-cut flag, or per-frame video labelling. You use every element of the output and mask the loss by the true sequence length.

Many-to-many delayed: input and output are both sequences but offset. Translation in the book — you must read the whole sentence before emitting the first word. Our project two is the simplest case, offset by exactly one: predict the next sample.

Point at the code box. The shape is not a philosophical choice, it is whether you write h[-1] or out. Half the bugs in student sequence code are reading the wrong one.

Question: 'which shape is a chatbot?' Delayed many-to-many, and that is also why it needs an encoder and a decoder — which is exactly where chapter 16 starts.



## Slide 4
The architecture slide. Keep it brisk; the equation is the content.

Left: the compact picture. A feedforward layer maps x to h to o. A recurrent layer adds one edge, the loop on h. That is the entire structural difference.

Right: the same thing unfolded across three time steps. Emphasise that unfolding is a way of drawing, not a different model — there is one h, one set of weights, visited repeatedly. Walk the equation: h at time t is a nonlinearity applied to W_xh times the current input, plus W_hh times the previous hidden state, plus a bias. Then the output is a second map off h.

Hammer the subscript point. None of the three matrices carries a t. The same W_hh is applied at sample 3 and at sample 130. That is why a 4,471-parameter RNN can read a 150-sample cycle, and it is the same inductive bias argument we made for convolution last week — share the parameters along the axis where the structure repeats.

Mention the alternative recurrences in one sentence: you can also feed the OUTPUT back rather than the hidden state, into either the hidden or the output layer. nn.RNN implements hidden-to-hidden, which is the most expressive because h is wider than o and is not forced through the output nonlinearity. Output recurrence is what you get for free when you feed a model its own prediction — which is exactly project two.

Question to the room: 'how many parameters does this have if x is 8-dimensional and h is 48?' Walk them through it: 8×48 plus 48×48 plus biases. They should be able to do this before the next slide.



## Slide 5
The de-mystification slide. Students treat nn.RNN as a black box; fifteen lines of numpy-style torch is enough to show there is nothing inside it.

Left panel: six bars, the two hidden units at each of three time steps, computed by hand beside nn.RNN's own output. They agree to six times ten to the minus eight, which is float32 epsilon. Not approximately — identically, up to the order of floating-point operations.

Right panel: what the layer actually holds. For nn.RNN(4, 2): W_xh is 2 by 4, W_hh is 2 by 2, and there are TWO bias vectors. Pause on that. Mathematically b_xh plus b_hh is one bias and the split is redundant. PyTorch keeps them separate to keep the input and recurrent paths symmetric, which matters once we get to gated cells where each gate has its own pair. Students who assume there is one bias get confused reading state_dict.

Read the code box aloud, especially the last line. prev_h = ot. That single vector is the complete memory of the network. Everything the model knows about the first hundred samples has to be in those numbers when it reaches sample 101. Say that slowly, because it is the sentence that motivates the next four slides.

Question: 'tanh is the default activation here — why not ReLU?' Because the recurrent path applies the same nonlinearity hundreds of times and ReLU is unbounded; the state can run away. Bounded activations keep the state in range, at the cost of saturating gradients, which is the trade-off the next slide is about.



## The mechanism slide. Derive just enough to make the picture inevitable.

Write the chain rule for the loss at time t with respect to W_hh. The piece that matters is the partial of h at time t with respect to h at time k, and that term is a PRODUCT of t minus k Jacobians — one per step you travel back. Products of many matrices do one of two things. They shrink to nothing, or they blow up. There is no third option unless you engineer one, which is what the LSTM does.

Left panel is the one-unit case, which is the book's figure 15.8. The recurrent weight to the power t minus k. At 0.97 you are down to two percent after 130 steps; at 1.03 you are up by a factor of 45. Read the annotation: a three percent error in one weight becomes a fiftyfold error over a cycle. These are not extreme values — they are rounding.

Right panel is the same thing measured on the real model. We scale W_hh at initialisation, compute the spectral radius, and take the gradient norm on one batch. Nothing is trained. Flat at about 0.45 to 0.65 while rho is below 1.2, then 13 at rho 1.43, then twenty-eight thousand at rho 1.65. Say the numbers; the jump is the point.

Then the sting: PyTorch's default initialisation puts our RNN at rho about 0.55. We are comfortably in the VANISHING regime, which is the far more common failure and the quieter one — nothing errors, nothing diverges, the loss just sits there.

Question: 'if rho below one makes gradients vanish and above one makes them explode, why not initialise at exactly one?' Good instinct, and it is a real technique — orthogonal and identity initialisation for RNNs. But the Jacobian includes the derivative of tanh, which is at most one and usually much less, so rho of W_hh equal to one still contracts.



## Slide 7
The diagnostic slide, and the one I would most like them to steal.

Explain the experiment first, because it is almost too simple. Take an untrained cell. Run it forward for 130 steps on random input. Put a loss on the LAST hidden state only. Backpropagate, and record the norm of the gradient at every step along the way. Normalise by the last step. Twelve repetitions, median. Nothing is trained, nothing is tuned, and it runs in about a second and a half.

Now read the curves right to left. At ten steps back all three are within a factor of ten of each other — over short distances everything works, which is why RNNs have a reputation for working. At fifty steps back the vanilla RNN is at four times ten to the minus thirteen. At a hundred steps back it is EXACTLY zero: the number underflowed float32. There is no learning signal, not a small one.

Then the result that should surprise them. The dashed red line is a default-initialised LSTM, and it vanishes too — ten to the minus twenty-one at a hundred steps. Better than the RNN by ten orders of magnitude and still utterly useless. 'Use an LSTM for long-range dependencies' is repeated everywhere and, with default initialisation at this distance, it is simply false.

The blue line is flat. That is the same LSTM with one line changed, and slide 9 is about that line. Do not explain it yet — let the gap sit.

Point at the dotted orange line at 103: that is where our inclusion event lives.

Question: 'the grey curve hits zero around 90 steps. What happens if I switch to float64?' You buy maybe another 150 steps of underflow headroom and change nothing that matters — the signal is already far below the noise of the optimiser. Precision is not the problem.



## Slide 8
The architecture that fixes slide 7, and the one students find most intimidating. Reduce it to one idea before you show any equation.

The idea: in a vanilla RNN every path from the past to the present goes through a weight matrix and a nonlinearity, so the gradient is a product of Jacobians. The LSTM adds a second state — the cell state, the green line across the top — and nothing multiplies it except an element-wise gate. Trace the green line with a finger: one times, one plus, straight through. That is the whole invention. Everything else is machinery for deciding what to put on that line and what to take off it.

Now the three gates, each a sigmoid of the same two inputs — the current sample and the previous hidden state. Forget: how much of the old cell state survives. Input, with the tanh candidate: what new information gets written. Output: how much of the cell state is exposed as this step's hidden state. Note that the cell state and the hidden state are different things, and only the hidden state goes to the next layer — a point the diagrams in most textbooks blur.

Read the two equations at the bottom. The first is the conveyor: old state times forget, plus new candidate times input. The second is the read-out.

History worth thirty seconds: the 1997 cell had no forget gate. Gers, Schmidhuber and Cummins added it in 2000 because without it the cell state grew without bound on long sequences. The gate that fixes our problem is a bug fix from three years after the original paper.

GRU in one sentence: merge forget and input into one update gate, drop the separate cell state, get roughly three-quarters of the parameters. We measure it on slide 11.

Question: 'if the cell state has no matrix on it, why does it not just blow up?' Because the forget gate is a sigmoid, bounded by one. It can only shrink the state, never amplify it.



## Slide 9
The payoff of slide 7, and the most immediately useful thing in the lecture.

Start with the arithmetic, because it is the whole argument. At initialisation PyTorch draws every bias near zero. So the forget gate is sigmoid of roughly zero, which is 0.5. The cell multiplies its memory by one half at every step. Over the 103 steps between our inclusion and the loss that is 0.5 to the 103 — about ten to the minus thirty-one. The information is gone before training starts.

Set the forget-gate bias to plus two and the gate sits at 0.88. Now 0.88 to the 103 is two times ten to the minus six. Still small, but twenty-five orders of magnitude more signal, and crucially it is above the floor where the optimiser can work with it.

Left panel is the consequence: 47.7 percent, which is the majority class, against 98.0 percent. Same architecture, same data, same 22 epochs, same seed. One line.

Make the chicken-and-egg point explicitly, because it is the deep reason. The LSTM is perfectly capable of learning to hold its forget gate open — but to learn that it needs a gradient from the distant past, and that gradient has already vanished because the gate is closed. It cannot bootstrap. You have to hand it the initial condition.

The code box is worth photographing. PyTorch stores LSTM gate biases in the order input, forget, cell, output, so the forget slice is hidden to two-hidden. A GRU stores reset, update, new — and its update gate occupies the same slice, so one helper covers both.

Cite it: Gers et al. 2000 proposed it, and Jozefowicz, Zaremba and Sutskever measured it across thousands of architectures in 2015 and recommend initialising the forget bias to 1. This is not folklore.

Question to the room: 'why not set the bias to 10 and never forget anything?' Then the cell can never clear itself, the state saturates, and on a long sequence everything blends together. Exercise 3 has them find the ceiling.



## Slide 10
A practical slide. It is the single most common bug in student sequence code and it never raises an error.

Set up the problem. Our cycles are 110 to 150 samples. A tensor is rectangular, so a batch has to be padded to its longest member. Fine so far. The trap is what happens next: if you hand the padded tensor to nn.LSTM and read hidden[-1], you get the hidden state after stepping through all the padding. For the shortest sequence in a batch with the longest, that is forty steps of zeros processed after the cycle ended. The state you read has been decaying through zeros the whole time.

The notebook measures it: on a toy batch, the final hidden state of a 4-sample sequence padded to 10 differs by 0.36 between the packed and unpacked versions, on values of order 0.2. It is not a rounding difference, it is a different vector.

pack_padded_sequence carries the true lengths alongside the data so the layer stops each sequence at its own end. Use enforce_sorted=False and PyTorch sorts and unsorts internally, handing hidden back in the original batch order. The lengths must be on the CPU even when the data is on a GPU — that catches people.

Also on this slide, briefly: the embedding. Our op_code channel is one of five G-code block types. One-hot would do for five, but nn.Embedding is the general answer — a trainable lookup table — and you need it the moment the vocabulary is tool IDs or part numbers. It is exactly the embedding layer from the book's text chapter, doing the same job on categorical process data.

Question: 'does this matter if I pad to a fixed length for every batch?' It matters MORE — then every short sequence carries a long tail of zeros.

Expect: 'what about masking instead?' Equivalent for the loss, but it does not stop the recurrent state being corrupted. Packing fixes it at the source.



## Slide 11
The results slide. Four separate lessons; make sure each one lands on its own.

One: the vanilla RNN never leaves the majority class, 47.7 percent, which is exactly the in-tolerance rate. Not a tuning failure — slide 7 showed its gradient is literally zero at the distance that matters. There is nothing to optimise.

Two, and this is the one to dwell on: the default-initialised LSTM does the same. Identical 47.7. The architecture everybody reaches for, failing completely, until one line of initialisation takes it to 98. If they remember one thing from this lecture, make it this.

Three: the bidirectional LSTM does not win. 97.8 against the unidirectional 98.0, for twice the parameters and twice the training time. It is natural to assume that reading backwards makes the early event easy — and it does, for the backward pass, whose final state sits at t equals zero. But that same backward pass must then carry the LATE factor all the way back. Each direction has one easy end and one hard end. Bidirectionality earns its keep when the evidence is spread through the middle, not pinned at the ends. Exercise 5 has them build that version.

Four: the dilated 1-D CNN gets 92.3 percent in ten seconds, against fifty for the LSTM. Seven dilated convolutions, receptive field 255 samples, no recurrence at all — so the gradient path back to sample 25 is seven layers, not 103 time steps. This is the modern answer for fixed-horizon sequence tasks and it connects straight back to last week. Its weakness is the fixed receptive field: a 400-sample cycle would need more layers.

Question: 'the GRU beat the LSTM with fewer parameters — should I always use a GRU?' The honest answer is that the difference here, 98.2 against 98.0, is within noise on 600 test cycles. The defensible claim is that the GRU is cheaper and did not lose, which is the usual finding in the literature.



## Slide 12
The scientific-hygiene slide, and the habit most worth transplanting into their own work.

Set it up as a claim under test. We have been asserting all lecture that the label depends on an event a hundred samples before the end. That is a claim about the DATA, not the model, and it is falsifiable. So: take the same gates-open LSTM, the same recipe, and show it only the last 45 percent of every cycle. If the information really is early, this must fail.

Left panel: 98.2 percent on the whole cycle, 50.7 on the last 45 percent, against a 47.7 majority baseline. It is at chance. And note what the late window DOES contain — the entire finishing pass, which is one of the two factors. One factor alone is worth nothing, exactly as the 2-by-2 table on slide 2 predicted.

Right panel is the per-cell breakdown of the best model. Neither fault: 100 percent. Both faults: 100 percent. The two single-fault cells, which are the ones that actually require combining the two ends, are 96 and 97 percent. Those are also the cells where the irreducible label noise lives — the generator adds four microns of noise against a twelve micron band, so a single-fault part sitting at minus twenty can occasionally land inside tolerance. About two percent of them do. So 96 to 97 is close to the Bayes rate, not a model deficiency.

Then generalise the habit. Any time you claim a model learned a long-range dependency, hide the range and check it breaks. Any time you claim it uses a particular channel, remove the channel. It is cheap, and it converts an assertion into a measurement.

Question: 'what if the late-window model had got 80 percent?' Then there is leakage — something late is correlated with the early event — and the long-range story is partly wrong. That is exactly why you run it.



## Slide 13
The closing content slide: the second project, and the bridge to next week.

First the mapping, which is the clever part and worth making explicit. The book's second project is a character-level language model on a Jules Verne novel. Ours is the same model with load levels instead of characters: discretise the spindle current into 48 bins, and 'predict the next character' becomes 'predict the next load bin'. Same embedding, same CrossEntropyLoss over a vocabulary, same Categorical sampling, same temperature. Perplexity came out at 4.9 effective bins out of 48.

Defend the discretisation, because it looks odd to an engineer: it buys a full DISTRIBUTION over the next value instead of a point estimate. The model can represent 'either still cutting at 11 amps, or the tool has just retracted to 1 amp' — which a squared-error forecaster cannot; it would average them and predict 6, a value that never occurs. Exercise 7 has them build the MSE version and watch it smear the phase transitions.

Now the three panels. Same model, same seed samples, only the logit scale changes. At alpha 0.5 the distribution is flattened and the trace jumps between bins constantly — novel and wrong. At alpha 2.0 it is sharpened, and you get long flat plateaux with clean transitions between them, which is what a real cycle looks like. Alpha one is the trained distribution. Note this is the reciprocal of the temperature you see in text generation.

Then the bridge, and keep it short. Everything in this lecture funnels a sequence through one fixed-width vector — ours compresses 130 samples into 48 numbers. We made that work by hand-initialising a gate so a single bit could survive being multiplied a hundred times. Attention stops pretending that is a good idea: let the read-out look at every step and learn weights over them. The gradient path to sample 25 becomes one multiplication. The notebook shows the mechanism with untrained weights so they can see the shape of it.

Question to leave hanging: 'if attention can see every step directly, why keep the recurrence at all?' That is the transformer, and that is next week.



## Slide 14
Close by walking the strip left to right, naming the step and what goes wrong at it: sequences — scaling fitted on the wrong split; pad and pack — forgetting the pack and reading a state full of zeros; embed — one-hotting a vocabulary that should have been embedded; gated cell — using the default initialisation on a long dependency; read out — taking h[-1] when the task needed every step; guard — no clipping, and no idea how far the gradient reaches.

Then pick three rules rather than reading all seven. Rule 4 is the one that makes them better engineers: the gradient-reach measurement costs a second and a half and would have told them on day one that the default LSTM could not do this job. Rule 5 is the one they will use this week. Rule 7 is the one that makes their results trustworthy.

Point at the notebook. Six minutes on two CPU cores, no downloads, every number on these slides printed in it. Exercises 2 and 3 are the ones to do first — find the distance at which the default LSTM starts to fail, and find the ceiling on the forget-gate bias. Exercise 5 is the one to argue about next session: change the data so the bidirectional model actually wins, which forces them to articulate what bidirectionality is for.

Flag the chapter's code, honestly: both of its projects are built on torchtext, which PyTorch deprecated and archived in 2024, and a Gutenberg download. The mechanics in the chapter are sound and unchanged; only the data pipeline needed replacing.

Preview: chapter 16 keeps the weighted read-out from slide 13 and throws away the recurrence entirely. Every limitation we worked around today — the fixed-width bottleneck, the sequential unroll, the vanishing path — is what the transformer is a response to.

Final question to leave them with: 'your sequence model works. How do you know it is using the part of the sequence you think it is?' Slide 12 is the answer.



