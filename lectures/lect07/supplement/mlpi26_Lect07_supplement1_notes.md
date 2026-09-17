# Neural network fundamentals with PyTorch — deck + notebook (generated 2026-09-08)
 
Deliverables produced in this session:
 
- `PyTorch_NN_fundamentals_lecture.pptx`, renamed to `mlpi26_Lect07_supplement1.pptx` — 14 slides (dark title + dark summary, 12 light
  content slides), speaker notes on every slide (960–1270 characters each), QA'd via a
  LibreOffice render at 90 dpi. Page size 959.98 × 540 pt.
- `PyTorch_NN_fundamentals_arm_dynamics.ipynb`, renamed to `mlpi26_Lect07_nbsup1.ipynb` — 42 cells, executed end to end, all outputs
  embedded, zero errors and zero stderr warnings, ~21 s total runtime on CPU.
Audience level requested: **upper-undergraduate engineering** (same as the model-evaluation
lecture). Scope held to the six listed topics — MLP architecture, backpropagation, PyTorch
tensors, autograd, DataLoader, building/training a model — **plus save/load & deployment**,
which the user selected as the one extra. No GPU section, no overfitting/regularisation
section, no convolutional or sequence models.
 
## Running example
 
**Inverse dynamics of a 2-DOF planar robot arm.** Synthetic but physics-motivated
(`np.random.default_rng(7)`), 9000 samples. Torques come from the textbook rigid-body equation
τ = M(q)q̈ + C(q,q̇)q̇ + g(q) + f(q̇), plus 6 % torque-sensor noise.
 
Six inputs (`q1, q2, qd1, qd2, qdd1, qdd2`) → two outputs (`tau1, tau2` in N·m).
Split 6000 / 1500 / 1500 train / val / test, `default_rng(8).permutation`.
 
Link parameters: `M1,M2 = 3.2, 1.8 kg`; `L1,L2 = 0.40, 0.32 m`; `LC1,LC2 = 0.20, 0.16 m`;
`I1,I2 = 0.055, 0.021 kg m²`; viscous friction `0.85, 0.42`; Coulomb `0.55, 0.28` via
`tanh(12·q̇)`; `TAU_NOISE = 0.06`.
 
Model: `Linear(6,64) → Tanh → Linear(64,64) → Tanh → Linear(64,2)`, 4738 parameters.
Adam `lr=3e-3`, `StepLR(step_size=25, gamma=0.5)`, batch 64, 60 epochs, ~9–10 s on CPU.
Inputs and targets both standardised with **training** statistics.
 
## Headline results (reproducible — deck and notebook agree exactly)
 
| quantity | value |
|---|---|
| linear baseline, test RMSE τ₁ / τ₂ | 4.859 / 2.021 N·m |
| linear baseline, test R² τ₁ / τ₂ | 0.336 / 0.268 |
| **MLP, test RMSE τ₁ / τ₂** | **0.439 / 0.178 N·m** |
| **MLP, test R² τ₁ / τ₂** | **0.9946 / 0.9943** |
| sensor noise floor (0.06 × sd) | 0.358 / 0.142 N·m |
| improvement over linear | 11.1× on τ₁; MLP sits 1.23× above the noise floor |
| final train / val MSE (standardised) | 0.00522 / 0.00591 — no overfitting |
| trajectory RMSE (unseen 4 s move) | MLP 0.18 / 0.08 vs linear 4.02 / 1.58 N·m |
| hand-coded backprop vs autograd | max abs difference 1.5 × 10⁻⁸ |
| finite differences vs autograd (120 weights) | max abs difference 9.5 × 10⁻⁵ |
| batch-size effect, 30 epochs, val MSE | bs 64 → 0.0090; bs 512 → 0.0122; full batch → 0.580 |
| learning-rate effect, 30 epochs, val MSE | 1e-4 → 0.121; 3e-3 → 0.0090; 1e-1 → 0.205 |
| checkpoint size / single-sample latency | ~21 KB / tens of µs on CPU (< 5 % of a 1 kHz budget) |
 
The linear model's poor R² is pedagogically load-bearing: it comes from the `cos(q2)` inertia
coupling, the `cos(q1)` / `cos(q1+q2)` gravity terms and the quadratic velocity products, none
of which a linear map can represent. **Do not weaken those terms** or the whole "why a neural
network" narrative collapses.
 
## Deck outline
 
1. Title (dark)
2. What torque should each motor produce? — arm schematic + torque trace
3. A straight line cannot do this — linear vs MLP predicted-vs-true, both panels
4. The multilayer perceptron — architecture schematic with weight shapes
5. Where the nonlinearity comes from — ReLU/tanh/sigmoid + the q₂ sweep collapse
6. Backpropagation is the chain rule, node by node — forward/backward graph
7. Tensors: shape, dtype, device, gradient — shape-flow strip + code box
8. Autograd: every gradient from one call — code box + gradient check & loss slice
9. Dataset and DataLoader — pipeline schematic + batch-size comparison
10. `nn.Module`: `__init__` declares, `forward` connects — full class + Sequential/subclass pair
11. The training loop is four lines — code box + loss curves & learning-rate sweep
12. Does it work? Look at the test set once — trajectory + RMSE bars vs noise floor
13. Save the weights, not the model object — deployment strip + save/load code
14. What to take away (dark) — workflow strip + 7-rule protocol
## Notebook sections
 
Environment · the data (inline generator, self-contained) · train/val/test split and
standardisation · tensors (shape/dtype/device, the shape-error message) · autograd
(`requires_grad`, accumulation and `zero_grad`, `no_grad`, `detach`) · backpropagation by hand
vs autograd + finite-difference check · `nn.Module` (subclass and Sequential, no output
activation) · `Dataset`/`DataLoader` (custom class and `TensorDataset`, knob table) · the
training loop + loss curves + the loss-curve diagnostic table · held-out evaluation vs the
linear baseline and the noise floor + trajectory test · save / load / deploy + latency ·
the 8-rule protocol · 8 exercises · references.
 
## Notes for reuse
 
- Figures come from `make_figs.py` (imports `arm_common.py`); the deck is built by
  `build_deck.js` (pptxgenjs) and the notebook by `build_notebook.py` (nbformat). Those
  generator scripts live only in the session workspace, not in this project.
- **pptxgenjs gotcha (again):** `LAYOUT_16x9` is 10 × 5.625 in. Use
  `defineLayout({name, width: 13.333, height: 7.5})`.
- **Reproducibility gotcha, new this time:** a `DataLoader` with `shuffle=True` and no explicit
  `generator` draws its shuffle seed from the *global* torch RNG at iteration time. Anything that
  touches the global RNG earlier — including a single `next(iter(loader))` demo cell — silently
  changes the training run. Fix used here: pass `generator=torch.Generator().manual_seed(1234)`
  to the DataLoader **and** call `g.manual_seed(1234)` immediately before training. Both the
  notebook and `make_figs.py` do this, which is why every number matches to the last digit.
- `torch` is not preinstalled in the session container and the pytorch.org wheel index is
  blocked by the egress proxy; plain `pip install torch --break-system-packages` from PyPI works
  (installed 2.14.0+cu130, runs fine on CPU).
- Palette identical to the previous two decks (blue `#2a78d6`, orange `#eb6834`, aqua `#1baf7a`,
  yellow `#eda100`, ink `#0b0b0b`/`#52514e`, surface `#fcfcfb`), light content slides, dark
  title/summary (`#141413`, accent `#3987e5`).
- Figure-layout fixes that were needed after eyeballing the PNGs: legends colliding with curves
  (fixed by raising `ylim` headroom rather than shrinking type), a schematic footer colliding
  with layer labels (removed from the figure, moved to the slide callout), and a
  gradient-descent path that exploded to 1e6 (fixed by making the two swept weights real leaf
  tensors and stepping them with SGD+momentum on a precomputed frozen-feature loss).
- Inference latency is machine-dependent and drifts between runs (33–45 µs observed). The
  deployment figure therefore says "tens of µs" and "well under 5 % of the 1000 µs budget"
  instead of a hard number; the notebook prints the measured value.
- Natural follow-on lectures from here: regularisation and overfitting (dropout, weight decay,
  early stopping) on this same arm data; convolutional networks; physics-informed / residual
  learning, which is already seeded as notebook exercise 8 and cites Lutter et al. 2019.

# Slide Notes

## Slide 1
Open by naming the deliverable: by the end of this lecture everyone will have trained a neural network that outputs motor torques for a robot arm, and will understand every line of the code that did it. Stress that we use one example the whole way through — the inverse dynamics of a two-link planar arm — because switching examples per topic is how students lose the thread. Say why this example is not a toy: learning inverse dynamics is a real, published technique for arms with belts, harmonic drives and joint flexibility, where the rigid-body model you derive in a dynamics course is simply wrong. Point at the footer: 9,000 samples, 4,738 parameters, under ten seconds on a CPU — nobody needs a GPU for this lecture. Ask the room: who has trained a neural network before, and who has derived the equations of motion of a two-link arm? The split tells you which half of the lecture to slow down on. Tell them the notebook runs end to end and is the homework.


## Slide 2
Set up the engineering problem before any machine learning appears. Left panel: the arm, two joints, two motors, gravity pulling the links down. Write the manipulator equation on the board if the room has not seen it — inertia term, Coriolis and centrifugal term, gravity term, friction term. Say what each one does physically: the inertia matrix depends on the elbow angle because folding the arm changes how hard it is to swing; the Coriolis term is what makes a fast elbow motion push back on the shoulder. Right panel: the torque actually required along one four-second pick-and-place move. Point out that it is smooth but nowhere near a straight line in any input. Read the callout numbers aloud — six in, two out, nine thousand samples. Ask the room: why split off a validation set at all, why not just train and test? Answer: because we are about to choose a learning rate, a width and a number of epochs, and every one of those choices burns a little test-set integrity. The objection you will get: 'we already know the physics, why learn it?' Answer that real arms have flexibility, backlash and temperature-dependent friction that no textbook model captures, and that this same pipeline is how you learn the residual.


## Slide 3
This is the slide that justifies the whole lecture. Left panel: the linear fit. Point at the cloud — a perfect model puts every point on the dashed diagonal, and this one is spread across ten newton-metres. Say what that means on hardware: the arm sags on the way up and overshoots on the way down. Read the number: R-squared 0.34, so the linear model explains about a third of the variance in shoulder torque. Right panel: the same plot for the MLP we will build today — a thin line on the diagonal, R-squared 0.995. Now the important second point: the MLP's error, 0.44 newton-metres, is only 1.2 times the torque-sensor noise. That is the floor. No architecture, no amount of training and no extra data can go below it, because the labels themselves are that noisy. Ask the room: if your test error equals the noise floor, what should you do next? Answer: stop tuning the model and go improve the sensor. The objection you will hear: 'that's not fair, the linear model got raw inputs.' It is a fair objection — hold it for exercise 6, where they add cosine and velocity-product features by hand and watch the linear model close most of the gap. That is exactly what the hidden layers are learning.


## Slide 4
Walk the diagram left to right. Six input units, one per number the controller has. Then a hidden layer: each unit computes a weighted sum of all six inputs, adds a bias, and passes the result through tanh. Sixty-four of those in parallel, then sixty-four more, then two output units for the two torques. Emphasise the shapes written above the arrows — W1 is 64 by 6, W2 is 64 by 64, W3 is 2 by 64 — because reading a shape error message is the skill they will use most this week. Count the parameters out loud: 384 plus 64 plus 4096 plus 64 plus 128 plus 2 equals 4,738. Compare that with 6,000 training samples and ask the room: does that ratio worry you? It should prompt the overfitting conversation, and we will answer it with the loss curves in a few slides. Note that the word 'perceptron' is historical baggage — this is just a function with tunable coefficients. The question a good student asks here: how do I choose 64? Honest answer: you do not know in advance; you start small, you widen while validation loss improves, and you stop. That is exercise 2.


## Slide 5
Do the algebra on the board, it takes fifteen seconds and it lands: W2 times (W1 x plus b1) plus b2 equals (W2 W1) x plus something. The product of two matrices is a matrix. So without an activation, depth buys you literally nothing. Left panel: the three activations they will meet first. ReLU is the default for hidden layers — cheap, and its gradient is either 0 or 1, so it does not shrink gradients as they travel back. Tanh is smooth and symmetric, which suits a smooth regression target like torque, and it is what this notebook uses. Sigmoid is mostly of historical interest for hidden layers because it saturates on both sides. Right panel is the payoff: hold five of the six inputs fixed and sweep the elbow angle. The true torque is that hump — inertia coupling through cos(q2) plus gravity. The tanh MLP tracks it; the best possible straight line is the red dashes and it is not close. Ask the room: which activation would you pick here and why? Then flag the exercise: swap tanh for ReLU and look for the piecewise-linear kinks in this same sweep. The student objection to expect: 'ReLU is not differentiable at zero.' True, and irrelevant in practice — the subgradient 0 is used and it never matters.


## Slide 6
The message of this slide is that backpropagation is not an algorithm to memorise; it is bookkeeping for the chain rule. Trace the top row left to right: input, affine map, tanh, affine map, prediction, loss. Now the bottom row right to left, and say each step out loud. dL/dL is one. dL/d-tau-hat is 2(tau-hat minus tau) over the number of elements. To get dL/dz2 you multiply by the local derivative of that node, and so on. The key sentence: each node needs only its own local derivative and the gradient handed to it from the right — it never needs to know anything about the rest of the network. That locality is why the backward pass costs about the same as the forward pass, and it is why training networks with billions of parameters is possible at all. Derive dW2 = (dL/d-tau-hat)^T h on the board so they see where the transpose comes from. Then tell them the notebook implements exactly this in seven lines of tensor algebra and checks it against .backward(), agreeing to ten decimal places, and that they should run it themselves. Question for the room: why do we not just use finite differences? Answer: two forward passes per weight versus one backward pass for all of them.


## Slide 7
Keep this slide brisk — it is plumbing, but plumbing that causes most first-week bugs. Four properties, and every one of them will bite someone this term. Shape: the first axis is the batch, always; a single sample must be unsqueezed into a batch of one before a Linear layer will accept it. Dtype: models are float32; NumPy defaults to float64, so torch.tensor(np_array) without an explicit dtype is a classic error. Device: the model and the data must be on the same device, and 'expected all tensors to be on the same device' means you moved one and forgot the other. requires_grad: the switch that turns tracking on. Then read the error message in the code box out loud and decode it — mat1 is 64 by 8, mat2 is 6 by 64, the inner dimensions 8 and 6 disagree, so the input has the wrong number of columns. Tell them the fix is always the same: print both shapes on the line before. Ask the room what x.T does to a (64, 6) tensor and why you would ever want it. The follow-up question you will get: does .numpy() copy? On CPU it shares memory, so mutating one mutates the other — a good thing to know before it surprises them.


## Slide 8
Start with the scalar example and let them check it in their heads: f equals 3w squared plus 2w, so df/dw is 6w plus 2, which is 26 at w equals 4 — and that is exactly what .backward() puts in w.grad. Then the single most important operational fact in the lecture: gradients accumulate. Calling backward twice without clearing gives you 52, not 26. That is a deliberate feature — it lets you split a large batch across several forward passes — but forget zero_grad in your loop and your gradients become the running sum of every batch you have ever seen, and the loss explodes after a handful of steps. Left panel: we verified autograd against finite differences on 120 weights and the largest disagreement is about 1e-4, which is finite-difference truncation error, not an autograd error. Right panel: the loss surface over two weights with everything else frozen, and the path gradient descent takes down it. Say clearly that the real surface is 4,738-dimensional and this is a two-dimensional slice through it — do not let them think optimisation is a nice bowl in general. Ask the room what torch.no_grad() saves you: memory and time, because no graph is built.


## Slide 9
Left panel is the pipeline: a Dataset holds the logged pairs, a DataLoader shuffles them and hands out batches of 64, and 94 of those batches make one epoch. Underneath is the loop they will write in fifteen minutes. Stress the two settings that matter. shuffle equals True for training, because otherwise the network sees the log file in recording order and can learn the order rather than the physics; shuffle equals False for validation and test, because a fixed order makes per-sample debugging possible. Right panel is the batch-size experiment: identical epochs, identical learning rate, only the batch size changes. Full batch — one gradient step per epoch — has barely moved after thirty epochs. Batch size 64 gives 94 steps per epoch and gets two orders of magnitude lower. Say plainly why: it is not that small batches are magic, it is that they take far more steps for the same amount of data. Ask the room where the trade-off turns around — very small batches mean noisy gradients and poor hardware utilisation. Mention num_workers only in passing: zero for in-memory data, a few when you are reading files.


## Slide 10
Read the class aloud, it is only twelve lines. __init__ declares the pieces; forward says how they connect. super().__init__() must come first, before you assign any layer, or PyTorch cannot register the parameters and your optimiser will silently train nothing — a genuinely nasty bug, worth ten seconds of warning. Explain why you call model(x) rather than model.forward(x): __call__ runs registered hooks around forward, and skipping it breaks anything that relies on them. Now the red line at the bottom, which is the most commonly repeated beginner mistake in regression: no activation on the output layer. Our torques span minus twelve to plus twenty-nine newton-metres; a tanh there would clamp every prediction into plus or minus one and no amount of training would help. Mention that classification output layers are also left bare, because CrossEntropyLoss applies the softmax internally — applying it yourself as well is the matching classification mistake. Ask the room when they would choose Sequential over subclassing. Expected question: how do I add a physics prior? Answer: return the analytic rigid-body torque plus the network output — that is exercise 8, and it is what published work does.


## Slide 11
Number the four lines with your finger as you say them: clear, forward, backward, update. Tell them that if they remember nothing else from today, remember this order. model.train() and model.eval() do not train or evaluate anything — they flip a flag that dropout and batch-norm read. This network has neither, so it changes nothing here, but get into the habit now, because the day they add dropout and forget model.eval() their validation loss goes mysteriously noisy and they lose an afternoon. Left panel: our run. Training and validation loss lie on top of each other for sixty epochs, which means no overfitting and that more data would buy very little. Teach the diagnostic explicitly: both high and flat means underfitting, so go bigger; train low and validation rising means overfitting, so regularise or stop early; both low and together means done. Right panel: the learning rate is the one knob that decides everything. At 1e-4 the run is still descending at epoch 30 — correct but wasteful. At 1e-1 it bounces around and never settles. Ask the room how they would find a good learning rate without this plot; steer them to 'try a log-spaced sweep of three or four values for a few epochs each'.


## Slide 12
Two panels, two different questions. Left: a smooth pick-and-place trajectory the network has never seen. The black curve is the true torque, the blue dashes are the network, and they are indistinguishable at this scale; the red dotted line is the linear model, and it is wrong by five newton-metres in places. This is the plot a control engineer wants, because random test samples do not tell you whether the model holds together along a continuous motion. Right: test RMSE per joint, with the sensor-noise floor drawn as a dashed line. Read the numbers. Then make the point that matters most: the MLP bar is nearly touching the noise floor, so further architecture tuning is wasted effort — the next improvement has to come from a better torque sensor or more informative data. Say once more that we looked at this test set exactly one time, at the very end, and that every choice of width, learning rate and epoch count was made against the validation set. Ask the room: what would you check before putting this on real hardware? The answer you want is extrapolation — this network has only ever seen velocities up to about three radians per second, and it has no idea what happens beyond that. That is exercise 7, and it is the safety case.


## Slide 13
This is the slide that turns a notebook into something that runs on a robot. First rule: save the state_dict, which is just an ordered dictionary of tensors, not the model object. torch.save(model) pickles your class definition alongside the weights, so renaming a module or refactoring ArmMLP silently breaks every checkpoint you own. Second rule, and the one people actually get wrong in the field: the normalisation constants are part of the model. Ship them in the same file. If the controller standardises with the wrong mean and standard deviation, the network receives garbage and outputs garbage, and nothing in the code will complain. Third: call model.eval() and wrap inference in torch.no_grad(), so no graph is built and dropout and batch-norm behave correctly. Point at the figure: the checkpoint is about twenty kilobytes, and a single forward pass takes tens of microseconds on a CPU, so it uses a few percent of the one-millisecond budget of a one-kilohertz control loop. Ask the room what else you would want before trusting this in a loop: a saturation limit on the commanded torque, and a check that the input state lies inside the training distribution.


## Slide 14
Close on the workflow strip: log the data, get it into tensors, define the model, wrap it in a loader, train, evaluate, deploy. Say that steps two to five are literally the same few lines in every PyTorch project they will ever write — only the model and the data change — and that the reason this lecture used one running example is so they can see that shape clearly once. Then read the seven rules; do not paraphrase them, they are the revision list. Spend an extra sentence on rule five, the loss-curve diagnostic, because it is the one skill that transfers to every model they will ever train, and on rule six, because knowing when to stop is what separates an engineer from someone tuning hyperparameters forever. Point them at the notebook: it runs end to end in about twenty seconds, every number on these slides comes out of it, and the eight exercises at the bottom are the homework. Flag exercises 4 and 5 as the fastest way to build intuition — deliberately breaking standardisation and deliberately forgetting zero_grad teaches more than any amount of reading. Finish by asking what they would need to add to put this on a real arm, and let the discussion run: safety limits, distribution checks, and a residual formulation over the nominal rigid-body model.


