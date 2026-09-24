# The mechanics of PyTorch — deck + notebook (generated 2026-09-18)
 
Source: Raschka, Liu & Mirjalili, *Machine Learning with PyTorch and Scikit-Learn*, **ch. 13**
"Going Deeper — The Mechanics of PyTorch". Follows the ch. 12 lecture
(`pytorch-nn-fundamentals-lecture-notes.md`) in the same series.
 
## Deliverables
 
- `PyTorch_mechanics_lecture.pptx` — 13 slides (dark title + dark summary, 11 light content
  slides), speaker notes on **every** slide (1330–1880 characters each), QA'd via a
  LibreOffice render at 90 dpi. Page size 959.98 × 540 pt.
- `PyTorch_mechanics_AGV_fleet.ipynb` — 64 cells (28 code), executed end to end, all outputs
  embedded, **zero errors and zero stderr warnings**, ~6 min total runtime on CPU.
Audience level requested: **upper-undergraduate engineering** (same as the previous two
lectures). Scope decision by the user: **drop PyTorch Lightning entirely** (the chapter's
Lightning section targets the 2021 v1.5 API — `training_epoch_end`, `resume_from_checkpoint`,
`gpus=1` — all since removed) and drop the MNIST project. Covered: computation graphs,
tensors/`requires_grad`/initialisation, autograd, `nn.Sequential` + loss/optimizer choice,
model capacity, `nn.Module`, custom layers, feature columns, end-to-end regression project.
 
## Running example
 
**A warehouse AGV (automated guided vehicle) fleet**, two datasets, both synthetic but
physics-motivated, both generated inline in the notebook (`np.random.default_rng`).
 
### Q1 — docking (classification, the capacity problem)
 
`make_docking_data(n=600, seed=7)`. Two features drawn `Uniform(-1, 1)`:
`x1` = lateral offset (× 50 mm tolerance), `x2` = heading error (× 4°).
Label: `dock_ok = 1 iff sign(x1) != sign(x2)` — **exact XOR with a physical mechanism**: a
differential-drive AGV on a short final straight can remove one error, not both; if the nose
points back toward the centreline the errors cancel, otherwise they compound.
Split 300 train / 300 valid. Validation class balance 0.543, so the trivial baseline is 54.3 %.
 
Training: `nn.BCELoss`, plain `SGD(lr=0.05)`, `batch_size=4`, **400 epochs**, DataLoader with
an explicit `generator=torch.Generator().manual_seed(seed)`.
 
### Q2 — trip energy (regression, the feature-column problem)
 
`make_fleet_data(n=5000, seed=7, noise=0.035)`, 5000 jobs, split 3500 / 750 / 750 by
`default_rng(8).permutation`.
 
Columns: numeric `payload_kg, speed_mps, grade_pct, distance_m`; ordinal
`battery_age_months`; nominal `floor_type, vehicle_model`; target `energy_wh`.
 
Generator constants (**these are load-bearing — see below**):
`CRR = {epoxy 0.011, sealed_concrete 0.017, steel_grating 0.026}`;
`TARE_KG = {T2 320, T4 520, H6 980}`; `DRIVE_EFF = {0.84, 0.86, 0.89}`;
`MAX_PAYLOAD = {250, 800, 1800}`; `AGE_BOUNDARIES = [12, 30, 48]`,
`AGE_LOSS = [0.01, 0.04, 0.09, 0.16]`; `P_AUX_W = 95`; `REGEN_FRACTION = 0.25`;
`STOP_SPACING_M = 110`; `grade ~ N(0, 1.8) clipped to ±4.5`; `payload = U(0,1)**1.3 * pay_max`.
 
Encoding → 14 columns: 4 standardised numeric (**train-split statistics only**) +
`torch.bucketize` age into 4 bands one-hot + `one_hot` floor (3) + `one_hot` model (3).
Model `Linear(14,32)-ReLU-Linear(32,16)-ReLU-Linear(16,1)`, 1025 params, Adam `lr=3e-3`,
`StepLR(step_size=30, gamma=0.5)`, batch 64, 80 epochs, target standardised with train stats.
 
## Headline numbers (deck and notebook agree exactly — both read `results.json`)
 
| quantity | value |
|---|---|
| docking, no hidden layer (3 params) — valid. accuracy / BCE | **0.610 / 0.6978** (≈ ln 2 = 0.6931) |
| docking, 2 hidden × 8 units (105 params) — train / valid. accuracy | 0.987 / **0.973** |
| capacity ladder, mean over seeds 1–3 (min–max) | 0 hidden **0.590** (0.563–0.610) · 1×4 **0.921** (0.820–0.973) · 1×8 **0.939** (0.863–0.977) · 2×8 **0.972** (0.970–0.973) |
| always-predict-"docks" baseline | 0.543 |
| autograd vs hand-derived gradient (1 example, BCE) | max abs difference **0.0** (exact) |
| gradients: `dL/dw = [-0.18859, +0.12573]`, `dL/db = -0.31432`, loss 0.37734 | |
| NoisyLinear robustness, σ = 0 / 0.1 / 0.2 / 0.3 / 0.4 | plain 0.973 / 0.915 / 0.851 / **0.783** / 0.736 · noisy 0.933 / 0.906 / 0.849 / 0.784 / 0.741 |
| fleet, least squares (same 14 features) | MAE **5.951** Wh · RMSE 9.150 · R² **0.7597** · MAPE 74.5 % |
| **fleet, MLP 14-32-16-1** | MAE **0.895** Wh · RMSE 1.431 · R² **0.9941** · MAPE 6.4 % |
| measurement noise floor (3.5 % of mean job) | 0.710 Wh — MLP sits 1.26× above it |
| final train / valid MSE (standardised) | 0.00269 / 0.00494 — not overfitting |
 
## Pedagogically load-bearing facts — do not "fix" these
 
- **The flat model's loss sits at ln 2.** That is the whole capacity argument: the loss itself
  announces "this model has learned nothing", and no number of epochs changes it. If you retune
  and it drops below ~0.69, the slide loses its point.
- **The linear baseline's R² of 0.76 comes from four specific nonlinearities**: mass × grade and
  mass × distance products, the kink at grade = 0 (downhill returns only 25 %), the `d/v`
  auxiliary term, and the discrete battery-age bands. Weaken any of them (raise `REGEN_FRACTION`
  toward 1, drop `P_AUX_W`, smooth `AGE_LOSS`) and the MLP's advantage collapses.
- **`REGEN_FRACTION` must stay ≤ ~0.25 given the `crr` values**, or downhill trips produce
  *negative* energy (net regeneration). It was 0.32 with `grade ~ N(0,2.2)` in the first draft
  and 173 rows went negative.
## The honest negative result (slide 10 / notebook §8)
 
`NoisyLinear` (the book's custom layer) **does not help** on the docking problem, and costs ~4
points of clean accuracy. Verified across training σ ∈ {0, 0.25, 0.40} and evaluation
σ ∈ {0 … 0.4}: from σ = 0.2 the curves are indistinguishable. Reason: the true boundary lies
exactly on the axes, so input noise moves points across it regardless of the model — the ~78 %
at σ = 0.3 is a Bayes-rate ceiling set by the sensor. The book reaches the same conclusion.
This is taught as a feature, not hidden: the engineering fix is a better pose estimate, a wider
dock funnel or a re-try manoeuvre, not a bigger network. **Do not spend time trying to tune this
into a win — it is not available.**
 
## Deck outline
 
1. Title (dark)
2. One AGV fleet, two questions — dock scatter + energy histogram
3. PyTorch records a graph as your code runs — DAG schematic + code box
4. Four attributes decide what a tensor can do — attribute strip + `nn.Parameter` trap code box
5. One backward() call, every gradient — hand vs autograd bars + loss curve
6. nn.Sequential, and choosing the loss — code box + loss/optimizer contrast boxes
7. A straight line cannot dock an AGV — decision regions, 61 % vs 97 %
8. How much model is actually enough? — capacity ladder (3 seeds, error bars) + accuracy curves
9. nn.Module: you write forward(), so it can branch — chain vs branch schematic
10. Writing a layer that PyTorch does not have — code box + train/eval demo + robustness ceiling
11. Getting mixed columns into one tensor — feature-column schematic
12. End to end: what will this job cost? — loss curves + predicted-vs-measured
13. What to take away (dark) — workflow strip + 7 rules
## Notebook sections
 
Environment · the two datasets (inline generators) · computation graphs (rank 0/1/2, Python
control flow in the graph) · tensors (shape/dtype/device/requires_grad, leaf vs non-leaf,
`requires_grad_`, Xavier init, `nn.Parameter` vs bare tensor) · autograd (hand check, non-scalar
`backward`, accumulation and `zero_grad`, `no_grad`/`detach`) · `nn.Sequential` + loss/optimizer
tables with a wrong/right softmax block · capacity (flat vs deep, learning-curve diagnostic
table, capacity ladder, decision regions) · `nn.Module` + `predict()` · custom layers
(`NoisyLinear`, `self.training` idiom, the honest robustness result) · feature columns (split
first, bucketize, one-hot, embedding note) · end-to-end project (training, held-out evaluation
vs least squares, residuals, two speed sweeps) · the 9-rule protocol · **8 exercises** ·
references.
 
## Notes for reuse
 
- Scripts live only in the session workspace: `agv_common.py` (generators), `train_all.py`
  (all analyses → `results.json`), `make_figs.py` (11 figures + a `analysis_cache.pt` cache so
  layout iterations do not re-train), `build_notebook.py`, `build_deck.js`, `nbtools.py`.
- **`nbformat` / `nbclient` could not be installed** — PyPI returned 503 for the whole session
  (torch had installed fine an hour earlier). Worked around with `nbtools.py`, a ~130-line
  notebook builder + executor: writes nbformat 4.5 JSON directly and executes cells in one
  namespace with `ast` splitting (trailing `ast.Expr` → `execute_result`), `redirect_stdout`,
  `warnings.catch_warnings(record=True)` → stderr outputs, and `plt.get_fignums()` → base64
  `display_data`. **Reusable; keep it.** It also stubs `plt.show` to a no-op, because matplotlib
  ≥ 3.8 under Agg emits a `UserWarning` from `show()` that would otherwise pollute every figure
  cell's stderr.
- pptxgenjs gotcha, third lecture running: `LAYOUT_16x9` is 10 × 5.625 in — always
  `defineLayout({name, width: 13.333, height: 7.5})`.
- **New layout lesson:** compute figure width from the *available vertical space*, not the other
  way round. `figureFit(s, name, yTop, yBottomMax, maxW)` sets `w = min(maxW, (yBottom-yTop)*aspect)`.
  Six of eleven callouts fell off the bottom of the slide on the first render because widths
  were hard-coded at 11.1–11.6 in for figures with aspect ratios between 2.2 and 2.9.
- `float(tensor_with_requires_grad)` emits a `UserWarning` in torch 2.14; `.item()` does not.
- Figure fixes needed after eyeballing: rounded `FancyBboxPatch` boxes butt together at spacings
  that look fine numerically (pad is in data units) — leave ≥ 0.03 of the axis between them;
  multi-line labels overflow a box height of 0.13, use 0.22; a scatter legend placed below the
  axes with `bbox_to_anchor=(0.5,-0.22)` pushed fig 01's aspect to 2.23 and broke the slide, fixed
  by raising `ylim` headroom and putting the legend inside.
- The speed-sweep figure originally claimed a U-shaped energy curve; for the reference job the
  minimum is at ~2.35 m/s, outside the 0.5–1.9 m/s operating range, so the curve is monotone.
  Fixed by sweeping **two** jobs — a long light haul (monotone falling) and a short loaded H6
  shuttle (genuine interior minimum at 1.19 m/s). Check the physics before writing the sentence.
- Palette identical to the previous three decks (blue `#2a78d6`, orange `#eb6834`, aqua
  `#1baf7a`, yellow `#eda100`, violet `#4a3aa7`, ink `#0b0b0b`/`#52514e`, surface `#fcfcfb`),
  light content slides, dark title/summary (`#141413`, accent `#3987e5`).
## Natural follow-on lectures
 
Convolutional networks (ch. 14) — the only new idea is weight sharing, everything here carries
over. Also available from this same data: regularisation and overfitting (the fleet model is at
the noise floor, so you would have to shrink the training set to demonstrate it); physics-informed
residual learning (seeded as notebook exercise 5, cites Lutter et al. 2019); and deployment /
latency on an embedded controller, which was cut from this lecture for scope.
 
# Slide Notes

## Slide 1
Opening slide. The job of this lecture is to replace 'PyTorch is a library I copy training loops from' with 'PyTorch is a graph recorder plus a differentiator, and I know where each of my lines lands in it.'

Say up front that we use one running example for the whole hour: a warehouse AGV fleet, the driverless pallet trucks that move stock between racking and dispatch. Two questions, deliberately chosen. Q1 is a two-feature classification problem whose answer surface is XOR-shaped, which is the cleanest way to show what model capacity means. Q2 is an ordinary industrial regression with the messy column types you actually get out of a fleet database.

Point at the last line and mean it: every percentage and every watt-hour on these slides is printed by the companion notebook, which runs end to end on a laptop CPU in about six minutes and downloads nothing.

Question to open with: 'who here has written a training loop that ran, produced a number, and you could not say whether the number was right?' Most hands go up. That is the gap we are closing.

Expect: 'why not just use a higher-level trainer?' Answer: because the day it breaks, the stack trace is in this layer. Also note the book's Lightning section is written against a 2021 API that no longer exists — a good reminder that the mechanics outlive the wrappers.



## Slide 2
Set up both datasets before any PyTorch appears, because the rest of the hour keeps coming back to them.

Left panel: each dot is one final approach to a conveyor dock. The horizontal axis is lateral offset in units of the 50 mm tolerance, the vertical axis is heading error in units of 4 degrees. Blue docked, orange clipped the frame. Ask the room to find a straight line that separates the colours — give them ten seconds, then point out that blue occupies the top-left and bottom-right quadrants. The mechanism is physical: a differential-drive AGV on a short final straight can remove one error, not both; if the nose points back toward the centreline the errors cancel, if it points away they compound. So the label is the sign-XOR of the two errors.

Right panel: energy per job, 5 000 rows, heavily right-skewed because trip length and payload both vary by an order of magnitude. Read the annotation aloud — seven raw columns, four numeric, one ordinal (battery age in months), two categorical (floor surface and vehicle model).

Question to the room: 'which of these two do you expect to be harder?' Most say Q2 because it has more columns. It is the opposite — Q2 is a well-behaved regression once the columns are encoded, and Q1 defeats a model with three parameters no matter how long you train it.

Someone will ask whether the data is real. It is synthetic, generated from the road-load equation and from a geometric docking argument. Say so plainly, and say why: it runs anywhere, and we know the true mechanism, which is what lets us say later whether a model should have been able to fit it.



## Slide 3
This is the slide that explains why PyTorch feels different from the frameworks that came before it.

Walk the figure left to right. F_roll, F_grade and the trip distance are input tensors — the blue boxes. r1 adds the two forces, r2 multiplies by distance, and the loss compares the result with the measured energy. Point out that distance feeds r2 directly, skipping r1: the graph is a DAG, not a chain, and that matters because the gradient with respect to distance arrives along exactly one path here but would arrive along several in a real network, where it gets summed.

Then the orange arrow. One call to loss.backward() walks the same graph in reverse. Nobody wrote the backward pass; it exists because the forward pass was recorded.

Read the code box: nothing in it is PyTorch-specific except the two function names. No placeholders, no session, no shape declared in advance. The same function takes a scalar, a vector of three jobs, or a column matrix, by broadcasting.

Question to the room: 'the graph is freed after backward() — why would a framework throw away something it just built?' Answer: because it will be rebuilt next iteration anyway, possibly with a different shape, and holding it costs memory. That is also why calling backward() twice on the same loss raises an error unless you pass retain_graph=True — and if you find yourself passing that flag, you have usually made a mistake.

Expect: 'what about torch.compile?' Fair question. It traces the dynamic graph and compiles a static one for speed. It is an optimisation on top of what is on this slide, not a replacement for it.



## Slide 4
A short slide but it prevents a whole class of wasted afternoons.

Go across the four boxes. Shape: mismatches announce themselves loudly, 'mat1 and mat2 shapes cannot be multiplied'. dtype: gradients need floating point, which is why labels stored as int64 blow up when you try requires_grad_() on them. device: both operands must live on the same one. requires_grad is the dangerous one because it fails quietly — the loop runs, the loss is printed, and nothing moves.

Then the leaf idea underneath: a tensor you created with requires_grad=True is a leaf and keeps a .grad after backward(); anything computed from it is a non-leaf with a grad_fn and, by default, keeps no gradient.

The code box is the bug worth memorising. The second line looks right and is wrong: a bare tensor attribute is not registered with the module, so it is absent from model.parameters(), absent from state_dict(), and does not follow model.to('cuda'). The same trap catches a plain Python list of layers — use nn.ModuleList.

On initialisation: ask why not initialise everything to zero. Answer: symmetry. Identical weights receive identical gradients forever, so an eight-unit layer stays as expressive as one unit. Random is necessary, and Glorot fixes the scale so deep stacks neither saturate nor vanish. The notebook checks the 512x512 draw against the theoretical standard deviation of 0.0442 and they agree to four decimals.

Expect: 'should I use Xavier or He?' He for ReLU stacks, Xavier for tanh and sigmoid. nn.Linear's own default is a Kaiming-uniform variant, which is why you usually do not have to touch this at all.



## Slide 5
The slide that turns autograd from magic into arithmetic.

Take the smallest docking model: one neuron, two weights, a bias, binary cross-entropy. Derive on the board or recall from the notebook that the sigmoid derivative and the cross-entropy derivative cancel, leaving dL/dz = p - y. So dL/dw_j is just (p - y) times x_j and dL/db is (p - y). For the example in the notebook p = 0.6857, y = 1, loss = 0.3773, and the gradients come out at -0.1886, +0.1257 and -0.3143.

Left panel: those three numbers computed by hand beside the same three computed by loss.backward(). The difference is exactly zero — not small, zero, because autograd is doing the same multiplications in the same order, not a numerical approximation. Say that explicitly: automatic differentiation is not finite differences.

Right panel: the loss curve of the real docking model. Use it to introduce the accumulation rule. Ask the room why PyTorch accumulates instead of overwriting — someone will say 'a bug'. It is deliberate: it lets you split a batch too large for memory across several backward passes and step once. The price is that clearing is your job, every step.

Mention reverse versus forward accumulation in one sentence: a network has millions of inputs and one scalar output, so accumulating from the output backwards is cheaper by a factor of millions. That is also why backward() insists on a scalar, and why you pass a gradient argument if you really want it on a vector.

Expect: 'when do I need detach()?' Whenever a value leaves the training computation — logging, a target you do not want gradients through, or converting to NumPy.



## Slide 6
The build-a-model slide. Keep it brisk — most of the room has seen Sequential — and spend the time on the two boxes, which is where marks and afternoons are lost.

On the code: the model is a list of modules and it stays indexable afterwards, which is how you initialise or regularise a single layer. Show model[0] and model[2] and note that ReLU sits at odd indices because it is itself a module in the chain. 105 parameters for this network — write that number down, it comes back on the capacity slide.

Left box: the loss is chosen by the task, not by taste. The one that bites everybody is CrossEntropyLoss. It expects raw logits and applies log-softmax internally; if you add a Softmax layer you apply it twice, the gradients squash, and training crawls without ever erroring. Same story for BCEWithLogitsLoss versus Sigmoid + BCELoss — the fused version is numerically stable, the split version overflows for confident predictions.

Right box: SGD when you want to see what the model is doing, Adam when you would rather not tune a learning rate. We use SGD on the docking problem precisely because it keeps the capacity argument clean — nobody can claim the flat model failed because Adam did something clever.

Question to the room: 'why is there no activation on the output layer of a regression model?' Because a sigmoid or ReLU would bound the output, and energy in watt-hours is not bounded in [0,1].

Expect: 'why weight_decay rather than adding an L2 term?' Because the optimiser can fold it into the update cheaply. AdamW does it properly decoupled; Adam's weight_decay is subtly different, which is worth a footnote.



## Slide 7
The centrepiece experiment. Same data, same loss, same optimizer, same 400 epochs — only the architecture changes.

Left: the 3-parameter model. The shaded regions are its decision regions; the boundary is a single straight cut, because that is the only shape a linear model has. It scores 61 percent on this particular run, against 54.3 percent for the trivial always-predict-docks baseline, and its validation cross-entropy is 0.6978 — which is ln 2 to three decimals. Say what ln 2 means: it is the loss of a model that outputs one half for every input. The model has learned essentially nothing, and it says so in the loss.

Right: two hidden layers of eight ReLU units, 105 parameters, 97 percent. Trace the boundary with a finger — it is made of straight segments, because ReLU networks are piecewise linear. Four half-planes intersect to carve out the XOR quadrants. That is the whole trick: composition of linear maps with a nonlinearity in between buys you regions that a single linear map cannot express.

Question to the room, and wait for it: 'the flat model is stuck — should we train it longer, lower the learning rate, or add a layer?' The instinct is always to train longer. Show them the loss curve from the previous slide: it is flat from epoch 20 to epoch 400. No amount of optimisation fixes a representation problem.

Expect: 'couldn't you just add the feature x1*x2?' Yes — and that is exercise 3 in the notebook. The 3-parameter model solves it instantly with that feature. That is the honest answer, and it is worth saying: a hidden layer is a way of *learning* the interaction terms you would otherwise have to know about in advance.



## Slide 8
The follow-up question to the previous slide: how much model do we need? Three seeds per architecture, because a small network on 300 points is seed-sensitive and one run would mislead you. The error bars are the min-max range over those three seeds.

Left panel, read the numbers out: no hidden layer 59 percent; one hidden layer of four units 92 percent with 17 parameters; one hidden layer of eight units 94 percent with 33; two hidden layers 97 percent with 105. Two things to draw out. First, the jump happens at the *first* hidden layer, not at the second — that is the universal approximation theorem showing up in a 17-parameter model. Second, look at the error bars: the single-layer models range from 82 to 97 percent depending on the seed, while the two-layer model lands between 97.0 and 97.3 every time. Depth bought stability more than it bought accuracy here.

Right panel is the diagnostic habit worth carrying away. Four patterns: train loss high and flat means underfitting, add capacity; train low with validation rising means overfitting, add data or regularisation; both low and together means you are at the noise floor, stop; loss jumping around means the learning rate is too high or zero_grad() is missing. The grey curve here is pinned at 0.61 accuracy for 400 epochs — row one.

Question to the room: 'why not just always use the biggest network you can afford?' Answers worth drawing out: more parameters overfit small data, deep stacks suffer vanishing and exploding gradients, and on an AGV the model runs on an embedded controller with a latency budget.

Expect: 'is 97 percent good?' Not necessarily — a missed dock costs a manual recovery. The right next question is what the errors look like, and they cluster near the axes where the true boundary lies.



## Slide 9
The distinction students most often get wrong by defaulting to Sequential and then fighting it.

Top row: Sequential is one straight chain. It is the right answer for most feedforward models and there is no virtue in avoiding it.

Bottom row: the moment the data has to go two ways and come back together, Sequential cannot express it, because a list has no way to say 'and also'. Here the raw pose is carried around the noisy layer and concatenated — a skip connection in miniature. Everything modern, from ResNets to transformers, is this picture repeated.

The contract is three lines and worth dictating: sub-modules are assigned as attributes in __init__, which is what registers their parameters; forward() defines the flow; and you call model(x), not model.forward(x), so that hooks and the train/eval flag are handled.

The notebook shows the same 105-parameter network written both ways and confirms they reach the same 97.3 percent — the choice is about expressiveness, not accuracy. It also adds a predict() method that takes a NumPy array and returns class labels, which is the interface the fleet controller actually wants.

Question to the room: 'what happens if I keep my layers in a Python list, self.layers = [l1, l2, l3]?' Answer: they run fine in forward() and they never train, because model.parameters() cannot see them. nn.ModuleList exists for exactly this. Same failure shape as the bare tensor on slide 4 — worth naming as one rule: if PyTorch is to manage it, hand it to PyTorch.

Expect: 'can I mix them?' Yes, constantly — a Module whose attributes are Sequentials is the common pattern.



## Slide 10
Two lessons here, and the second one is the more valuable.

First, the mechanism. A custom layer is an nn.Module whose forward you write. NoisyLinear holds its own weight and bias as nn.Parameter, initialises them with Xavier, and perturbs its input while training. The idea was plausible: the AGV's on-board pose estimate is noisy, so train the network on noisy poses and it should learn a boundary that tolerates them — data augmentation, implemented as a layer.

Point at the left panel. Six calls with the identical input tensor. The first four are in training mode and every one differs; the last two are in inference mode and are bit-identical. This is the training/inference split that dropout and batch norm also have, and forgetting model.eval() before validating is one of the most common silent bugs in the field — your validation score wobbles and you blame the data. The book passes a training flag explicitly; in your own code read self.training, which model.train() and model.eval() set for the whole tree at once.

Second, and say this slowly: it did not work. The two curves lie on top of each other from sigma 0.2 onward, and noise injection cost about four points of clean accuracy. The reason is instructive — the true boundary lies exactly on the axes, so pose noise pushes points across it regardless of what the model does. That 78 percent at sigma 0.3 is a Bayes-rate ceiling set by the sensor.

Question to the room: 'the docking rate at sigma 0.3 is not good enough. What do you change?' The engineering answers are a better pose estimate, a wider dock funnel, or a re-try manoeuvre — not a bigger network. Knowing when the model is not the bottleneck is a senior skill, and this is a clean example of it.

Expect: 'so why teach the layer at all?' Because the day the method you need is in a paper and not in torch.nn, this is how you add it.



## Slide 11
Back to Q2, and the step that takes longer in practice than the modelling.

Go down the rows. The four continuous columns are standardised — subtract the mean, divide by the standard deviation — which matters because payload runs to 1800 kilograms and grade is a couple of percent; without scaling the optimiser spends its life on one axis. Battery age is ordinal, and we bucketize it into four bands at 12, 30 and 48 months with torch.bucketize, then one-hot the band. Floor type and vehicle model are nominal and get one-hot vectors of width three. Four plus four plus three plus three is fourteen columns.

Two choices to defend. First, why bucket a perfectly good continuous number? Because the physical effect is not smooth — internal resistance rises in steps as cells degrade, and the warranty bands are where the fleet already keeps records. Say where your boundaries come from; arbitrary ones are a modelling decision masquerading as a fact. Second, one-hot or embedding? One-hot is fine for three categories. With four hundred vehicle serial numbers you would use nn.Embedding, which is a trained lookup table — exactly a one-hot vector times a weight matrix, computed without forming the one-hot vector.

The callout is the exam question. Ask it directly: 'I standardised using the mean of the whole dataset before splitting. How bad is that?' For standardisation, mild. For target encoding of a categorical column, catastrophic. The habit is what matters: split first, then fit every statistic on the training rows only. Exercise 7 has them measure the bias themselves.

Also mention the target: energy is skewed, so we standardise it for training and convert back to watt-hours before reporting. Reporting an error in standardised units is a good way to be misunderstood.



## Slide 12
The payoff slide: the whole workflow, on data that looks like work.

Left panel, the learning curves on a log scale. Train MSE 0.0027, validation 0.0049, in standardised units. Both low, both flat, close together — that is the 'fit is as good as the noise allows' pattern from the diagnostic table. Tell them what it rules out: we are not overfitting, so regularisation would not help, and more data would not help much either.

Right panel, predicted against measured on the held-out 750 jobs, touched exactly once. Grey is least squares on the same fourteen features — note that this is a fair fight, the baseline gets identical feature engineering, so the whole gap is attributable to nonlinearity. Read the numbers: MAE 5.95 versus 0.89 watt hours, R-squared 0.760 versus 0.994, MAPE 75 percent versus 6 percent. The grey cloud bends away from the diagonal at both ends, which is exactly what a linear fit to a curved surface does.

Then the noise floor, which is the number that tells you when to stop. The generator adds 3.5 percent measurement noise, which on a mean job is 0.71 watt-hours. The network is at 0.89. It is within a factor of 1.3 of the best achievable error — anything further is chasing measurement noise.

Why the nonlinearity matters physically, in one line each: mass multiplies the rolling, gravity and acceleration terms, so those are products of features; downhill returns only a quarter of its energy through regeneration, so the grade term is kinked at zero; and auxiliaries cost d over v, so a slow trip costs more. The notebook sweeps speed for two jobs and shows the model recovered a genuine interior minimum for the short heavy shuttle — a number a scheduler can act on.

Question to the room: 'the residuals fan out with trip size. What does that mean for how you use the model?' Answer: budget with a percentage margin, not a fixed one.



## Slide 13
Close by walking the strip once, left to right, naming the step and the one thing that goes wrong at it: feature columns — leakage; tensors and DataLoader — dtype and shuffling; model — Sequential when it should have been a Module; loss and optimizer — softmax applied twice; training loop — the missing zero_grad(); evaluation — peeking at the test set more than once.

Then the seven rules. Do not read all seven aloud; pick three and make them land. Rule 2 and rule 3 between them account for most 'my loss does not move' posts on the forums. Rule 5 is the one that makes them better engineers, because underfitting and overfitting look similar on a bad day and call for opposite fixes. Rule 7 is the one that makes their results trustworthy — the least-squares baseline in this lecture got the same fourteen features, which is what made the comparison mean something.

Point them at the notebook. It runs in about six minutes on a laptop CPU, downloads nothing, and every number on these slides is printed in it. Exercises 2 and 3 are the ones to do first: break the training loop on purpose and watch how it fails, then prove the flat model's failure is capacity rather than optimisation. Exercise 5 asks them to write a custom layer that genuinely earns its place, which is the honest sequel to slide 10.

Preview what comes next: the same mechanics carry straight into convolutional networks, where the only new idea is weight sharing. Everything on this slide stays true.

Final question to leave with them: 'if your model beats the baseline by a mile, what is the first thing you should suspect?' Leakage.



