# Chapter 9. Training Craft: Modules, Losses, Early Stopping and Saving

*Part III. Deep learning for robot signals and images*

> **Case study:** a warehouse AGV fleet. Will this vehicle dock? How much energy will this job draw?
> **Companion notebook:** [`lectures/lect08/supplement/mlpi26_Lect08_nbsup1.ipynb`](../../lectures/lect08/supplement/mlpi26_Lect08_nbsup1.ipynb)
> **Lab:** [`labs/lab08/mlpi26_lect08_Lab.ipynb`](../../labs/lab08/mlpi26_lect08_Lab.ipynb) (spindle thermal-drift compensation: custom blocks, Huber loss, early stopping)
> **Built from:** Lecture 8 deck, Lecture 8 Supplement 1 and its speaker notes, Lab 8

Chapter 8 gave you a network that works and a loop that trains it. This chapter is about the mechanics underneath, so that when a training run produces a number you can say whether the number is right, and when it does not move you can find out why. The goal is to replace "PyTorch is a library I copy training loops from" with "PyTorch records a graph and differentiates it, and I know where each of my lines lands in that graph".

Two questions from one warehouse fleet carry the chapter. The first is a two-feature classification problem whose answer surface is XOR-shaped, the cleanest possible demonstration of what *model capacity* means. The second is an ordinary industrial regression with the messy mix of column types you get from a real fleet database. The chapter then adds the training-craft tools the case study did not need but your next project will: robust losses, regularisation, early stopping and checkpoints, illustrated with the spindle thermal-drift problem of Lab 8.

## Learning objectives

After this chapter you should be able to:

1. Explain what PyTorch records when you run a line of tensor arithmetic and what `loss.backward()` does with it.
2. Distinguish leaf from non-leaf tensors, and `nn.Parameter` from a bare tensor, and predict which ones the optimiser updates.
3. Derive a gradient by hand for a one-neuron classifier and confirm it with autograd.
4. Choose the output layer and loss for regression, binary and multi-class problems, and avoid the double-softmax trap.
5. Diagnose a **capacity** failure, as opposed to an optimisation failure, from the learning curves.
6. Decide between `nn.Sequential` and a subclass of `nn.Module`, and write a custom layer with train/inference behaviour.
7. Turn continuous, ordinal and nominal columns into one input tensor without leaking test information.
8. Use a robust loss, weight decay, dropout and **early stopping**, and save a checkpoint that can be reloaded and verified.

---

## 9.1 One fleet, two questions

A fleet of **automated guided vehicles (AGVs)** moves pallets between racking and dispatch. The fleet controller has to answer two questions.

| | Question | Task | Why it is here |
|---|---|---|---|
| Q1 | Given the pose error on final approach, will the vehicle dock or miss? | Binary classification, 2 inputs | The decision boundary is XOR-shaped, which exposes model capacity |
| Q2 | How many watt-hours will this transport job draw from the battery? | Regression, 7 raw columns | The columns are of mixed types, which exposes feature encoding and the full workflow |

Both datasets are synthetic, generated from physics inside the notebook. That is a deliberate choice: it runs anywhere, and because we know the true mechanism, we can say whether a model *should* have been able to fit it.

### 9.1.1 Docking: an XOR with a mechanism

An AGV closing on a conveyor dock has two pose errors when it enters the final straight: a lateral offset `x1` (in units of the 50 mm tolerance) and a heading error `x2` (in units of the 4° tolerance). A differential-drive vehicle on a short straight can correct one error, not both. If the nose points back toward the centreline, the errors cancel and the vehicle docks. If it points further away, they compound and the vehicle clips the frame. So

$$
\text{dock\_ok} = 1 \iff \operatorname{sign}(x_1) \neq \operatorname{sign}(x_2),
$$

which is exactly the XOR function. Docks occupy the top-left and bottom-right quadrants; misses occupy the other two. Any straight line you draw puts two same-coloured quadrants on opposite sides of it. The dataset has 600 approaches, split 300/300 into training and validation. In the validation set 54.3% of approaches dock, so "always predict docks" already scores **54.3%**.

### 9.1.2 Trip energy: mixed columns, real non-linearity

The energy log has one row per job, 5000 jobs in all. The energy follows textbook vehicle dynamics,

$$
E = \frac{W_\text{roll} + W_\text{grade} + W_\text{accel}}{\eta_\text{drive}\,(1-\text{age loss})} + P_\text{aux}\,\frac{d}{v},
$$

with four features that make it non-linear in the raw columns. Mass (tare plus payload) multiplies the rolling, gravity and acceleration terms, so each is a product of features. Downhill returns only 25% of the gravity energy through regeneration, so the grade term has a kink at zero. Auxiliaries (controller, lidar, safety scanners) draw a constant 95 W for the duration of the trip, so their cost goes as d/v and a slow trip costs *more*. Battery age enters through four discrete resistance bands.

The seven raw columns are of three kinds: four continuous (`payload_kg`, `speed_mps`, `grade_pct`, `distance_m`), one ordinal (`battery_age_months`), and two nominal (`floor_type`, `vehicle_model`). The target `energy_wh` has a mean of 20.4 Wh and a long right tail up to 220 Wh, because trip length and payload both vary by an order of magnitude.

## 9.2 The computation graph

PyTorch evaluates expressions on a **directed acyclic graph** whose nodes are operations and whose edges are tensors. The graph is *dynamic*: it is recorded as your Python runs, one operation at a time, and freed after `backward()`.

```python
def traction_work(f_roll, f_grade, distance):
    r1 = torch.add(f_roll, f_grade)
    r2 = torch.mul(r1, distance)
    return r2
```

The same function accepts a scalar (one job), a vector of three jobs, or a column matrix, by broadcasting. There is no session, no compile step and no placeholder. `torch.add` runs immediately *and* records a node, so you can print an intermediate tensor or branch on its value inside a model. A function that applies one regeneration stage per 2% of downhill grade, using a Python `for` loop whose length depends on the data, is perfectly legal; a static-graph framework could not express it without special operators.

Note that `distance` feeds `r2` directly, skipping `r1`. The graph is a DAG, not a chain. In a real network a tensor often reaches the loss along several paths, and its gradient is the sum of the contributions along each.

Why throw the graph away after `backward()`? Because the next iteration rebuilds it, possibly with a different shape, and holding it costs memory. That is also why calling `backward()` twice on the same loss raises an error unless you pass `retain_graph=True`; if you find yourself passing that flag, you have usually made a mistake. `torch.compile` traces this dynamic graph and compiles a static one for speed; it is an optimisation layered on top, not a replacement.

## 9.3 Tensors that hold parameters

Four attributes decide what a tensor can do in a model, and each has its own characteristic error:

| Attribute | What it controls | The mistake it causes |
|---|---|---|
| `shape` | Which operations line up | `mat1 and mat2 shapes cannot be multiplied` |
| `dtype` | Precision, and whether gradients are possible | `only Tensors of floating point dtype can require gradients` |
| `device` | Where the arithmetic happens | `Expected all tensors to be on the same device` |
| `requires_grad` | Whether a graph is recorded through it | Silently no `.grad`, so the optimiser does nothing |

The last one is dangerous because it fails quietly: the loop runs, the loss is printed, and nothing moves.

A tensor you create with `requires_grad=True` is a **leaf**. Anything computed from it is a **non-leaf** and carries a `grad_fn` that records how it was made. Only leaves keep a `.grad` after `backward()`.

### 9.3.1 `nn.Parameter` versus a bare tensor

Inside a module, a weight must be wrapped in `nn.Parameter`. That registers it with the module, so it appears in `model.parameters()` and reaches the optimiser, moves with `model.to(device)`, and is saved by `state_dict()`. A bare tensor attribute with `requires_grad=True` does none of that: it accumulates gradients that nobody applies. The same trap catches a plain Python list of layers, `self.layers = [l1, l2, l3]`: the layers run in `forward()` and never train. Use `nn.ModuleList` or `nn.ModuleDict`. One rule covers both cases: *if PyTorch is to manage it, hand it to PyTorch.*

### 9.3.2 Initialisation

Why not initialise every weight to zero? Symmetry. Identical weights receive identical gradients forever, so an eight-unit layer is never more expressive than one unit. Random initialisation is necessary. Its *scale* is what Glorot (Xavier) initialisation fixes: drawing weights with variance 2/(n_in + n_out) keeps the variance of activations and gradients roughly constant from layer to layer, so deep stacks neither saturate nor vanish.

$$
W \sim \mathcal U\!\left(-\sqrt{\tfrac{6}{n_\text{in}+n_\text{out}}},\ \sqrt{\tfrac{6}{n_\text{in}+n_\text{out}}}\right).
$$

The notebook checks a 512 × 512 Xavier-normal draw against the theoretical standard deviation of 0.0442; they agree to four decimals. Use He (Kaiming) initialisation for ReLU stacks and Xavier for tanh and sigmoid. `nn.Linear` already uses a Kaiming-uniform variant by default, which is why you rarely have to touch this.

## 9.4 Autograd, exactly

Take the smallest docking model: one neuron with two weights and a bias, trained with binary cross-entropy,

$$
z = w_1x_1 + w_2x_2 + b,\qquad p = \sigma(z),\qquad L = -\big[y\log p + (1-y)\log(1-p)\big].
$$

The sigmoid and the cross-entropy derivatives cancel, which is the classic result

$$
\frac{\partial L}{\partial z} = p - y,\qquad \frac{\partial L}{\partial w_j} = (p-y)\,x_j,\qquad \frac{\partial L}{\partial b} = p - y .
$$

For one approach (x = (0.6, −0.4), y = 1, w = (0.8, −0.5), b = 0.1) the model gives p = 0.6857 and loss 0.3773. The hand-derived gradients are ∂L/∂w = (−0.18859, +0.12573) and ∂L/∂b = −0.31432, and `loss.backward()` returns exactly the same values. The difference is **0.0**, not merely small, because automatic differentiation performs the same multiplications in the same order. It is not a numerical approximation like finite differences.

### 9.4.1 Why reverse mode

For a chain y = f(g(h(x))) you can accumulate the derivative from either end. Forward accumulation is cheap when there are few inputs; reverse accumulation is cheap when there are few outputs. A network has millions of inputs (its weights) and one output (the scalar loss), so reverse accumulation wins by a factor of millions. That is what `backward()` does, and it is why it insists on a scalar. On a vector output you must pass a weighting, `out.backward(torch.ones_like(out))`, which is equivalent to `out.sum().backward()`.

### 9.4.2 Three habits that prevent most autograd bugs

1. **Gradients accumulate.** `.grad` is added to, never overwritten. Without a reset, the gradient on a weight goes 18, 36, 54 over three steps instead of 18, 18, 18. Clear it once per step.
2. **Evaluate under `torch.no_grad()`.** No graph is recorded, so it is faster and uses less memory. Forgetting it on a validation loop is a classic cause of running out of memory.
3. **`.detach()` takes a value out of the graph**, for logging, for converting to NumPy, or for a target you do not want gradients to flow through.

The order of the three optimiser calls is flexible as long as the gradient is cleared between one `step()` and the next `backward()`. The notebook uses forward → `backward()` → `step()` → `zero_grad()`; Chapter 8 used `zero_grad()` first. Both are correct (Exercise 2).

## 9.5 Choosing the output layer and the loss

`nn.Sequential` chains modules in the order you list them, and the layers stay indexable afterwards, so you can initialise or regularise one layer individually:

```python
model = nn.Sequential(
    nn.Linear(2, 8), nn.ReLU(),
    nn.Linear(8, 8), nn.ReLU(),
    nn.Linear(8, 1), nn.Sigmoid(),
)                                   # 105 parameters
nn.init.xavier_uniform_(model[0].weight)
l1_penalty = 1e-3 * model[2].weight.abs().sum()   # add to the loss before backward()
```

The loss is chosen by the task, not by taste:

| Task | Last layer | Loss | Note |
|---|---|---|---|
| Regression | Plain `Linear` | `nn.MSELoss`, `nn.L1Loss` or `nn.HuberLoss` | MSE is in squared target units; see Section 9.10 for outliers |
| Binary classification | `Sigmoid` | `nn.BCELoss` | Works, but can overflow for confident predictions |
| Binary classification | None (logits) | `nn.BCEWithLogitsLoss` | **Preferred**: numerically stable |
| Multi-class | None (logits) | `nn.CrossEntropyLoss` | Applies log-softmax itself |

The trap worth memorising: `nn.CrossEntropyLoss` expects raw logits. If you also add `nn.Softmax` to the model, the softmax is applied twice, gradients are squashed, and training crawls without ever raising an error.

```python
# WRONG: softmax applied twice
model = nn.Sequential(nn.Linear(16, 3), nn.Softmax(dim=1)); loss_fn = nn.CrossEntropyLoss()
# RIGHT: let the loss do it
model = nn.Sequential(nn.Linear(16, 3));                    loss_fn = nn.CrossEntropyLoss()
```

For the optimiser, use plain **SGD** when you want to see what the model is doing and **Adam** when you would rather not tune a learning rate. The docking experiment uses SGD (learning rate 0.05, batch size 4, 400 epochs) precisely so that nobody can claim a model failed because the optimiser did something clever. The energy regression uses Adam.

## 9.6 Capacity: the docking experiment

Same data, same loss, same optimiser, same 400 epochs. Only the architecture changes.

| Model | Parameters | Train accuracy | Validation accuracy | Final validation BCE |
|---|---|---|---|---|
| No hidden layer (logistic regression) | 3 | 0.603 | 0.610 | **0.6978** |
| Two hidden layers of 8 ReLU units | 105 | 0.987 | **0.973** | 0.0703 |

The three-parameter model can only draw one straight cut, and it scores 61% against a 54% trivial baseline. Its validation cross-entropy is 0.6978, which is ln 2 = 0.6931 to two decimals. ln 2 is the loss of a model that outputs one half for every input. The model has learned essentially nothing, and its loss says so.

The 105-parameter network carves the plane into the four XOR quadrants. Its boundary is made of straight segments, because ReLU networks are piecewise linear: several half-planes, composed through a non-linearity, produce regions that no single linear map can express.

The tempting response to the flat model is to train it longer. Its training loss is flat from epoch 20 to epoch 400. **No amount of optimisation fixes a representation problem.** Exercise 3 proves it three ways: train for 4000 epochs, sweep the learning rate, or add the single engineered feature x₁x₂, with which the same three-parameter model solves the problem at once. That last result is the honest summary of what a hidden layer does: it *learns* interaction terms you would otherwise have to know about in advance.

### 9.6.1 How much model is enough?

A small network on 300 points is sensitive to its random seed, so the notebook trains each architecture with three seeds and reports the mean and range of validation accuracy:

| Architecture | Parameters | Mean accuracy | Min | Max |
|---|---|---|---|---|
| No hidden layer | 3 | 0.590 | 0.563 | 0.610 |
| 1 hidden layer, 4 units | 17 | 0.921 | 0.820 | 0.973 |
| 1 hidden layer, 8 units | 33 | 0.939 | 0.863 | 0.977 |
| 2 hidden layers, 8 units | 105 | 0.972 | 0.970 | 0.973 |

Two things stand out. The jump happens at the *first* hidden layer: that is the universal approximation theorem appearing in a 17-parameter model. And depth bought *stability* more than accuracy here: the single-layer models range from 82% to 97% depending on the seed, while the two-layer model lands between 97.0% and 97.3% every time.

Why not always use the biggest network you can afford? Because more parameters overfit small data, deep stacks suffer vanishing and exploding gradients, and on an AGV the model runs on an embedded controller with a latency budget.

### 9.6.2 Reading the curves

| What you see | What it means | What to do |
|---|---|---|
| Train loss flat and high (here, pinned at ln 2) | Underfitting: too little capacity | Add width or depth; add features |
| Train loss low, validation loss rising | Overfitting | More data, regularisation, early stopping |
| Both low and together | Fit is as good as the noise allows | Stop; more data will not help |
| Loss jumps around or diverges | Learning rate too high, or `zero_grad()` missing | Lower the rate; check the loop |

Underfitting and overfitting can look similar on a bad day and call for opposite fixes. Read the curves before changing anything.

## 9.7 `nn.Module` when `Sequential` is not enough

`nn.Sequential` expresses one straight chain, and for most feed-forward models that is the right tool. As soon as data must branch, rejoin, be reused, or flow differently in training and inference, subclass `nn.Module` and write `forward()` yourself. A skip connection, which carries an input around a block and concatenates or adds it back, is the smallest example; everything from ResNets (Chapter 10) to transformers (Chapter 12) is that picture repeated.

The contract is three lines. Sub-modules are assigned as attributes in `__init__`, which registers their parameters; `forward()` defines the flow; and you call `model(x)`, not `model.forward(x)`, so that hooks and the train/eval flag are handled.

```python
class DockNet(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.l1, self.a1 = nn.Linear(2, hidden), nn.ReLU()
        self.l2, self.a2 = nn.Linear(hidden, hidden), nn.ReLU()
        self.l3, self.a3 = nn.Linear(hidden, 1), nn.Sigmoid()

    def forward(self, x):
        x = self.a1(self.l1(x))
        x = self.a2(self.l2(x))
        return self.a3(self.l3(x))

    def predict(self, x):
        """Class labels for a NumPy array: the interface the fleet controller wants."""
        x = torch.as_tensor(x, dtype=torch.float32)
        with torch.no_grad():
            return (self.forward(x)[:, 0] >= 0.5).float()
```

Written this way, the network has the same 105 parameters and reaches the same 97.3% as the `Sequential` version. The choice is about expressiveness, not accuracy. The `predict()` method shows a useful habit: give the module the interface its caller actually needs. Modules and `Sequential` mix freely; a module whose attributes are `Sequential` blocks is the common pattern, and Lab 8 builds its regressor from a reusable `DenseBlock` in exactly this way.

## 9.8 Writing a custom layer, and an honest negative result

A custom layer is an `nn.Module` whose `forward()` you write. `NoisyLinear` computes wᵀ(x + ε) + b with ε ~ N(0, σ²) *only while training*. The idea was plausible: the AGV's on-board pose estimate is noisy, so train on noisy poses and the network should learn a boundary that tolerates them. It is data augmentation implemented as a layer.

```python
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, noise_stddev=0.1):
        super().__init__()
        w = torch.empty(in_features, out_features)
        nn.init.xavier_uniform_(w)
        self.w = nn.Parameter(w)                          # registered, so trained
        self.b = nn.Parameter(torch.zeros(out_features))
        self.noise_stddev = noise_stddev

    def forward(self, x):
        if self.training:                                 # set by model.train()/.eval()
            x = x + torch.normal(0.0, self.noise_stddev, x.shape)
        return x @ self.w + self.b
```

Reading `self.training`, which `model.train()` and `model.eval()` set for the whole module tree, is the idiomatic way to switch behaviour. Called repeatedly on the same input in training mode, the layer returns a different result each time; in evaluation mode the outputs are bit-identical. Dropout and batch normalisation have the same split, and forgetting `model.eval()` before validation is one of the most common silent bugs in the field.

Now the result. Both networks were evaluated with increasing pose noise σ added to the validation inputs, averaged over 40 noise draws:

| Evaluation noise σ | Plain network | `NoisyLinear` network |
|---|---|---|
| 0.0 | 0.973 | 0.933 |
| 0.1 | 0.918 | 0.905 |
| 0.2 | 0.848 | 0.840 |
| 0.3 | 0.786 | 0.786 |
| 0.4 | 0.733 | 0.735 |

**Noise injection did not help**, and it cost four points of clean accuracy. From σ ≈ 0.2 the two curves coincide. The reason is instructive. The true boundary lies exactly on the axes, so pose noise pushes points across it no matter what the model does. The 78.6% at σ = 0.3 is a Bayes-rate ceiling set by the sensor, not by the architecture.

The engineering conclusion is therefore about the vehicle, not the network. If docking reliability at σ = 0.3 is not good enough, buy a better pose estimate, widen the dock funnel or add a retry manoeuvre. No network trained on these two features will beat 78%. Knowing when the model is not the bottleneck is a senior skill. The mechanism still earns its place: the day the method you need is in a paper and not in `torch.nn`, this is how you add it.

## 9.9 Feature columns: mixed data into one tensor

Back to the energy question. A network eats one kind of tensor, and the fleet log has three kinds of column:

| Kind | Columns | Conversion | Width |
|---|---|---|---|
| Continuous | `payload_kg`, `speed_mps`, `grade_pct`, `distance_m` | Standardise with training mean and std | 4 |
| Ordinal | `battery_age_months` | `torch.bucketize` into 4 bands, then one-hot | 4 |
| Nominal | `floor_type`, `vehicle_model` | One-hot (or `nn.Embedding` for large vocabularies) | 3 + 3 |

```python
# 1. split FIRST: 3500 train / 750 validation / 750 test
# 2. statistics from the training split only
stats = df_train[NUMERIC].agg(["mean", "std"])

def encode(df):
    z = (df[NUMERIC] - stats.loc["mean"]) / stats.loc["std"]
    x_num = torch.tensor(z.values, dtype=torch.float32)
    band = torch.bucketize(torch.tensor(df["battery_age_months"].values, dtype=torch.float32),
                           torch.tensor([12.0, 30.0, 48.0]))
    parts = [x_num, nn.functional.one_hot(band, num_classes=4).float()]
    for col, vocab in (("floor_type", FLOORS), ("vehicle_model", MODELS)):
        code = torch.tensor([vocab.index(v) for v in df[col]])
        parts.append(nn.functional.one_hot(code, num_classes=len(vocab)).float())
    return torch.cat(parts, dim=1), torch.tensor(df["energy_wh"].values, dtype=torch.float32)
# feature tensor: (3500, 14)
```

Three choices here deserve a defence.

**Why standardise?** Payload runs to 1800 kg and grade is a few percent. Without scaling, the optimiser spends its life on one axis.

**Why bucket a continuous number?** Because the physical effect is not smooth: internal resistance rises in steps as cells degrade, and the warranty bands at 12, 30 and 48 months are where the fleet already keeps records. Bucket boundaries are a modelling decision, not a fact; say where yours come from.

**One-hot or embedding?** One-hot is fine for three categories. With 400 vehicle serial numbers or 2000 SKUs, use `nn.Embedding(num, dim)`, a trainable lookup table that maps each index to a dense vector. It is exactly a one-hot vector multiplied by a weight matrix, computed without forming the one-hot vector, and it can learn that some categories are similar (Exercise 6).

Two more rules. **Split before you fit any statistic.** For standardisation the leak from using whole-dataset statistics is mild; for target encoding of a categorical column it can be catastrophic (Exercise 7). And **standardise or log-transform a skewed target**, then convert predictions back to physical units before reporting. An error quoted in standardised units is a good way to be misunderstood.

## 9.10 End to end: what will this job cost?

Nothing in this section is new, which is the point. The model is a 14–32–16–1 ReLU MLP with 1025 parameters, trained with MSE on the standardised target, Adam at 3 × 10⁻³ with a `StepLR` halving every 30 epochs, batch size 64, for 80 epochs.

| Epoch | Train MSE | Validation MSE |
|---|---|---|
| 0 | 0.548 | 0.163 |
| 10 | 0.0121 | 0.0137 |
| 30 | 0.0043 | 0.0054 |
| 60 | 0.0030 | **0.0045** |
| 79 | 0.0027 | 0.0049 |

Both curves are low, flat and close together. That is the "fit is as good as the noise allows" pattern: regularisation would not help, and more data would not help much.

On the 750 held-out jobs, opened once:

| Model (same 14 features) | MAE (Wh) | RMSE (Wh) | R² | MAPE |
|---|---|---|---|---|
| Least squares | 5.951 | 9.150 | 0.760 | 74.5% |
| **MLP 14–32–16–1** | **0.895** | **1.431** | **0.994** | **6.4%** |
| Measurement noise floor (3.5% of mean job) | 0.710 | | | |

This is a fair fight. The baseline gets exactly the same 14 encoded features, so the entire gap is attributable to non-linearity, not to better feature engineering. The least-squares predictions bend away from the diagonal at both ends, which is what a linear fit to a curved surface does. The MLP's MAE sits **1.26 times above the noise floor**; anything further would be chasing measurement noise.

The residuals fan out with trip size, which means the *relative* error is roughly constant. Budget energy with a percentage margin, not a fixed ±1 Wh.

The model can also answer questions a scheduler cares about. Sweeping speed for two jobs shows two different shapes. For a long, light haul, auxiliary cost (∝ d/v) dominates and energy falls monotonically with speed; the cheapest speed is at the fast end of the range, 1.90 m/s. For a short, heavily loaded H6 shuttle, acceleration cost (∝ ½mv² per stop) takes over and there is a genuine interior minimum at **1.19 m/s**. A linear model has one slope for speed and must be wrong at one end of the range on both jobs. The visible kinks in the network's curves are its ReLU boundaries approximating 1/v piecewise-linearly.

## 9.11 Robust losses, regularisation and early stopping

The fleet model never overfit, so it needed none of the tools in this section. Most real projects do. Lab 8 develops them on a problem where they matter: **spindle thermal-drift compensation**. As a machining centre's spindle heats up, its bearings, motor windings and column casting expand at different rates, and the tool tip drifts 40 to 80 µm from where the controller believes it is. On a job with a ±5 µm tolerance that is the difference between a good part and scrap. The task is to predict the displacement from temperature sensors and operating state, accurately enough for the controller to apply an equal and opposite offset.

### 9.11.1 Split by group, not by row

Lab 8 logs *sessions* of 150 minutes, sampled every minute. Readings one minute apart are nearly identical, so a random row split would put near-duplicates on both sides and inflate every score. The lab splits by session into training, validation and test sets, the grouped split of Section 7.3.3 applied to time series.

### 9.11.2 The loss can matter more than the architecture

Squared error punishes a reading that is 20 µm wrong 400 times as much as one that is 1 µm wrong. A few corrupted labels (a probe that misread, a chip on the artefact) then pull the fitted curve toward themselves. The **Huber loss** is quadratic for small errors and linear beyond a threshold δ:

$$
\ell_\delta(r) = \begin{cases} \tfrac12 r^2 & |r| \le \delta \\ \delta\left(|r| - \tfrac12\delta\right) & |r| > \delta \end{cases}
$$

With δ = 2 µm (`nn.HuberLoss(delta=2.0)`), errors beyond 2 µm are penalised linearly, so outliers lose their leverage while ordinary errors are still fitted smoothly. Lab 8's central experiment changes only the loss, with the same architecture, data, optimiser and seed, and the error drops substantially. Choose the loss from what you know about the noise: MSE for Gaussian noise, Huber or L1 when the labels contain gross errors.

### 9.11.3 Weight decay and dropout

**Weight decay** adds a penalty λ‖w‖² that pulls weights toward zero, limiting how sharply the network can bend. Pass it to the optimiser (`weight_decay=1e-4`) rather than adding the term to the loss yourself; the optimiser folds it into the update cheaply. With Adam, prefer `torch.optim.AdamW`, which decouples the decay from the adaptive step size; Adam's own `weight_decay` behaves subtly differently (Loshchilov and Hutter, 2019). L1 penalties, which drive weights exactly to zero, are added to the loss as in Section 9.5.

**Dropout** (`nn.Dropout(p)`) zeroes each hidden activation with probability p during training and rescales the rest by 1/(1 − p), so that no unit can rely on any other. At inference it does nothing. It is the clearest example of a layer whose behaviour depends on `model.train()` and `model.eval()`, and the reason those calls matter.

### 9.11.4 Early stopping

The cheapest regulariser of all is to stop training when the validation loss stops improving. Track validation loss every epoch, keep a copy of the best weights seen so far, and stop once validation loss has failed to improve for `patience` epochs:

```python
import copy

def train_with_early_stopping(model, train_dl, val_dl, loss_fn, optimizer,
                              max_epochs=500, patience=20):
    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val = sum(loss_fn(model(xb), yb).item() * len(yb) for xb, yb in val_dl) \
                  / len(val_dl.dataset)
        if val < best_val - 1e-6:
            best_val, best_state, wait = val, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)       # restore the best, not the last
    return model, best_val, epoch
```

Restoring `best_state` at the end is what makes this early stopping rather than merely stopping early. Without it you keep the weights from the *worst* epoch of the patience window. The energy model shows why this matters even when nothing overfits badly: its best validation MSE (0.0045) came at epoch 60, and the final epoch was about 9% worse.

Early stopping uses the validation set to make a decision, so the best validation loss is an optimistic estimate of performance, for the same reason `best_score_` was in Section 7.6.3. Report the test-set error of the restored model, never the best validation loss.

## 9.12 Saving, reloading and verifying

Chapter 8 established the rule: save the `state_dict`, the arguments needed to rebuild the architecture, and every preprocessing statistic, in one file. For the fleet model that means the 14-column encoder's means, standard deviations, bucket boundaries and category vocabularies, plus the target mean and standard deviation. A model trained on standardised inputs produces confident nonsense if fed raw values, and nothing in the code will complain.

After reloading, verify. Feed the same batch to the original and reloaded models and check that the outputs match bit for bit; then measure single-sample latency on the target hardware. The fleet controller re-plans at 10 Hz, a 100 ms budget, which a 1025-parameter MLP uses a negligible fraction of (Exercise 8). The same check on Lab 8's spindle model closes the loop: reload, predict a held-out session, and report the error in micrometres against the ±5 µm tolerance.

## 9.13 The decisive experiment

When a model beats its baseline by a mile, the first thing to suspect is leakage. When it fails, the first thing to establish is *why*. The decisive experiment for this chapter separates capacity from optimisation from sensing:

- **Capacity:** give the failing model the missing interaction as a feature (x₁x₂ for docking). If it then succeeds, the architecture was the limit.
- **Optimisation:** train ten times longer and sweep the learning rate. If nothing changes, optimisation was not the limit.
- **Sensing:** add realistic noise to the inputs and compare architectures. If they converge to the same accuracy, as the plain and noisy networks did at σ ≥ 0.2, the sensor sets the ceiling and no model will lift it.

Run all three before spending another week on architecture.

## Summary

- PyTorch records a dynamic graph as your code runs and differentiates it in reverse. Autograd is exact, not an approximation.
- Parameters must be `nn.Parameter` or registered sub-modules (`nn.ModuleList`, not a Python list), or they never reach the optimiser.
- Choose the loss from the task: logits with `BCEWithLogitsLoss` or `CrossEntropyLoss`, and never a softmax before `CrossEntropyLoss`. Use Huber or L1 when labels contain gross errors.
- A loss pinned at ln 2 on a balanced binary task means the model has learned nothing. No amount of training fixes a capacity problem.
- Depth bought stability across seeds as much as accuracy. Use the smallest model that reliably works.
- Subclass `nn.Module` when data must branch or rejoin, and use `self.training` for train/inference differences.
- Encode continuous, ordinal and nominal columns by type, with every statistic fitted on the training split only.
- Always compare with a dumb baseline on the same features and quote the noise floor. The energy MLP reached R² 0.994 against 0.760 for least squares, 1.26 times above the noise floor.
- Regularise with weight decay, dropout and early stopping, and restore the best checkpoint. Report test error, not best validation loss.

## Exercises

1. **Bias gradient by hand.** Derive ∂L/∂b for the one-neuron model on paper, then extend the check to a batch of three docking examples. Does autograd return the sum or the mean over the batch? Which does `nn.BCELoss` use by default, and what does `reduction="sum"` change?
2. **Break the loop on purpose.** Delete `optimizer.zero_grad()` from the docking training function and plot the loss. Then move it before `loss.backward()` instead of after `optimizer.step()` and explain why that version still works.
3. **Capacity versus optimisation.** Prove that the flat model's failure is capacity by (a) training for 4000 epochs, (b) trying learning rates 0.005, 0.05 and 0.5, and (c) adding the feature x₁x₂ and retraining the same three-parameter architecture.
4. **Activation study.** Replace ReLU with Tanh, LeakyReLU and ELU in the 2 × 8 docking network, three seeds each. Tabulate final validation accuracy and the epoch at which it first exceeds 90%. Which changes, the ceiling or the speed?
5. **A custom layer that earns its keep.** Write `PhysicsFeatures(nn.Module)` that appends payload × distance, payload × grade and distance/speed to the 14 encoded features, and train the same MLP on 17 columns. Does the test MAE fall, or was the network already reconstructing those products?
6. **Embeddings.** Replace the two one-hot blocks with `nn.Embedding(3, 2)` each and compare test MAE and parameter count. Print the learned vectors for the three vehicle models: do they order themselves by tare mass?
7. **Leakage, measured.** Compute the standardisation statistics on the whole dataset instead of the training split, retrain, and report test MAE. How large is the bias, and why is it small for standardisation but potentially huge for target encoding?
8. **Deploy it.** Save the energy model's `state_dict` with its encoder statistics, reload it into a fresh module, confirm bit-identical predictions, and measure single-sample latency with `time.perf_counter`. What fraction of the 100 ms re-planning budget does one prediction use?
9. **Early stopping on the fleet.** Wrap the energy model's training in the early-stopping function of Section 9.11.4 with patience 10, 20 and 40. Compare the restored epoch, the test MAE and the training time with the fixed 80-epoch run.
10. **Huber on corrupted labels.** Corrupt 3% of the training targets in the fleet data by multiplying them by 3. Retrain with MSE and with `HuberLoss` at several values of δ (in standardised units). Plot test MAE against δ and explain the shape.

## Further reading

- X. Glorot and Y. Bengio, "Understanding the difficulty of training deep feedforward neural networks," *AISTATS* 2010.
- K. He, X. Zhang, S. Ren and J. Sun, "Delving deep into rectifiers: surpassing human-level performance on ImageNet classification," *ICCV* 2015. The source of He initialisation.
- A. G. Baydin, B. A. Pearlmutter, A. A. Radul and J. M. Siskind, "Automatic differentiation in machine learning: a survey," *JMLR* 18 (2018) 1–43.
- N. Srivastava et al., "Dropout: a simple way to prevent neural networks from overfitting," *JMLR* 15 (2014) 1929–1958.
- I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," *ICLR* 2019.
- L. Prechelt, "Early stopping — but when?" in *Neural Networks: Tricks of the Trade*, Springer, 2012.
- P. J. Huber, "Robust estimation of a location parameter," *Annals of Mathematical Statistics* 35(1) (1964) 73–101.
- C. Guo and F. Berkhahn, "Entity embeddings of categorical variables," arXiv:1604.06737, 2016.
- PyTorch documentation: autograd mechanics, `torch.nn`, and serialisation semantics, <https://pytorch.org/docs/stable/>.
