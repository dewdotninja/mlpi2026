# Chapter 8. Neural Networks and PyTorch

*Part III. Deep learning for robot signals and images*

> **Case study:** learning the inverse dynamics of a two-link robot arm
> **Companion notebook:** [`lectures/lect07/supplement/mlpi26_Lect07_nbsup1.ipynb`](../../lectures/lect07/supplement/mlpi26_Lect07_nbsup1.ipynb)
> **Lab:** [`labs/lab07/mlpi26_Lect07_Lab.ipynb`](../../labs/lab07/mlpi26_Lect07_Lab.ipynb) (the Chapter 3 fault task rebuilt in PyTorch)
> **Built from:** Lecture 7 deck, Lecture 7 Supplement 1 and its speaker notes

Part II ended with a protocol for building and judging classical models. Part III changes the tool. From here on, models are neural networks written in PyTorch, and the training loop that scikit-learn hid inside `.fit()` becomes code you write and can read line by line.

It is worth being honest about why. On the tabular fault data of Chapters 3 to 7, a tuned gradient-boosting model is often as good as a neural network, sometimes better. Neural networks earn their place in three situations that classical models handle poorly: raw, unstructured data such as camera images and long vibration waveforms, where there are no hand-made features to feed a tree (Chapters 10 to 12); smooth, high-dimensional function approximation, where the shape of the map is known only implicitly (this chapter); and control policies trained by interaction, which have no `fit/predict` interface at all (Chapters 13 and 14). This chapter introduces the machinery on a problem where the payoff is immediate and measurable.

## Learning objectives

After this chapter you should be able to:

1. Explain what a **multilayer perceptron (MLP)** computes and why the activation function is what gives depth its value.
2. Derive the **backpropagation** equations for a one-hidden-layer network, implement them, and check them against PyTorch.
3. Create, reshape and move **tensors**, and read the most common shape error.
4. Use **autograd** (`requires_grad`, `.backward()`, `.grad`, `torch.no_grad()`) and explain what `optimizer.zero_grad()` protects you from.
5. Wrap data in a **`Dataset`** and iterate it with a **`DataLoader`**.
6. Build a model with **`nn.Module`**, write the four-line **training loop**, and diagnose a run from its loss curves.
7. **Save, load and deploy** a trained network into a control loop, including its normalisation constants.

---

## 8.1 The engineering problem: what torque should each motor produce?

A robot controller answers one question a thousand times a second: given where the joints are, how fast they are moving and how fast I want them to accelerate, what torque should each motor produce? That map is the **inverse dynamics** of the arm. For a rigid arm it has a standard form,

$$
\tau = M(q)\,\ddot q + C(q,\dot q)\,\dot q + g(q) + f(\dot q),
$$

where M(q) is the configuration-dependent inertia matrix, C(q, q̇)q̇ collects the Coriolis and centrifugal terms, g(q) is gravity, and f(q̇) is joint friction. Each term has a physical meaning a control engineer can point to. The inertia matrix depends on the elbow angle because folding the arm changes how hard it is to swing. The Coriolis term is what makes a fast elbow motion push back on the shoulder.

For a rigid two-link arm you can write this down by hand. For a real arm with belts, harmonic drives, cable routing and flexible joints, the textbook model is simply wrong by an amount that matters. What you *can* do is log joint states and measured torques and learn the map. Learning inverse dynamics is a real and widely published technique (Nguyen-Tuong and Peters, 2011), and the pipeline in this chapter is the same one you would use to learn the residual between a nominal model and a real arm.

### 8.1.1 The data

The notebook simulates a tabletop two-link planar arm (link masses 3.2 and 1.8 kg, lengths 0.40 and 0.32 m) and logs **9000 motion samples**. Torques come from the exact rigid-body equation, with viscous and Coulomb friction, plus **6% torque-sensor noise**, typical of a current-based torque estimate. Everything is generated inside the notebook; there is no download.

| Symbol | Meaning | Unit |
|---|---|---|
| q₁, q₂ | Joint angles | rad |
| q̇₁, q̇₂ | Joint velocities | rad/s |
| q̈₁, q̈₂ | Desired joint accelerations | rad/s² |
| τ₁, τ₂ | Shoulder and elbow torques (targets) | N·m |

Six inputs, two outputs. The shoulder torque ranges from about −12 to +29 N·m. The data is split once, up front, into **6000 training, 1500 validation and 1500 test** samples. The three sets have three jobs: the optimiser touches only the training set; the validation set is watched during training to choose the learning rate, width and number of epochs; the test set is opened once at the end. If you tune anything against the test set, it stops being a test set (Chapter 7).

### 8.1.2 Standardise both sides, with training statistics only

The inputs live on very different scales (angles near 1 rad, accelerations near 5 rad/s²) and the targets are tens of newton-metres. A freshly initialised network produces outputs near zero; asking it to emit 25.0 immediately means a huge first gradient and an unstable start. Standardise inputs *and* targets to zero mean and unit variance, using statistics computed on the training set only:

```python
x_mu, x_sd = Xtr.mean(0), Xtr.std(0)
y_mu, y_sd = ytr.mean(0), ytr.std(0)
norm_x   = lambda a: (a - x_mu) / x_sd
norm_y   = lambda a: (a - y_mu) / y_sd
denorm_y = lambda a: a * y_sd + y_mu
```

Computing the statistics on the whole dataset would leak test information into training. Exercise 4 removes the standardisation entirely and shows how the run fails.

### 8.1.3 Why not a straight line?

Before building anything, fit the obvious baseline. A linear regression from the six inputs to the two torques achieves a test **R² of 0.336 for the shoulder and 0.268 for the elbow**. It explains about a third of the variance. On hardware, that means an arm that sags on the way up and overshoots on the way down.

The reason is in the physics. The inertia couples through cos q₂, gravity enters through cos q₁ and cos(q₁ + q₂), and the Coriolis terms are products of velocities. None of these is linear in the inputs. Exercise 6 adds those terms by hand as extra features and watches the linear model close most of the gap. The point of a neural network is that its hidden layers learn equivalent features from data, without being told the equations.

## 8.2 The multilayer perceptron

An MLP is a stack of layers, each of which computes a weighted sum of its inputs, adds a bias and passes the result through an element-wise non-linear **activation function**. The network for this chapter is

```
6 inputs -> Linear(6, 64) -> tanh -> Linear(64, 64) -> tanh -> Linear(64, 2) -> 2 torques
```

Each hidden unit is the same artificial neuron introduced in Chapter 1: a weighted sum of all six inputs plus a bias, then tanh. Sixty-four of them run in parallel, then sixty-four more, then two linear output units. The weight shapes are worth reading aloud because shape errors are the most common bug you will meet: W₁ is 64 × 6, W₂ is 64 × 64 and W₃ is 2 × 64. Counting weights and biases gives

384 + 64 + 4096 + 64 + 128 + 2 = **4738 parameters**,

trained on 6000 samples. Whether that ratio is worrying is answered by the loss curves in Section 8.6. How wide to make the hidden layers is not known in advance: start small, widen while validation loss improves, and stop (Exercise 2).

### 8.2.1 Where the non-linearity comes from

Without an activation, depth buys nothing. Two stacked linear layers collapse into one:

$$
W_2(W_1x + b_1) + b_2 = (W_2W_1)\,x + (W_2b_1 + b_2).
$$

Ten layers still collapse into one linear map. The activation is the only reason depth adds expressive power. Three activations cover most needs:

| Activation | Formula | Typical use |
|---|---|---|
| ReLU | max(0, x) | Default for hidden layers; cheap, gradient is 0 or 1, so it does not shrink gradients |
| tanh | (eˣ − e⁻ˣ)/(eˣ + e⁻ˣ), range (−1, 1) | Smooth, zero-centred; suits smooth regression targets such as torque |
| Sigmoid | 1/(1 + e⁻ˣ), range (0, 1) | Binary output probabilities; rarely used in hidden layers because it saturates on both sides |

This chapter uses tanh because inverse dynamics is a smooth function. Exercise 1 swaps in ReLU and asks you to look for its piecewise-linear kinks along a sweep of the elbow angle. (ReLU is not differentiable at zero; PyTorch uses the subgradient 0, and in practice it never matters.)

The universal approximation theorem (Hornik, Stinchcombe and White, 1989) says that one hidden layer with enough units can approximate any continuous function on a bounded domain. It says nothing about how many units "enough" is, or how easy the weights are to find. In practice, two or three moderately wide layers train more easily than one very wide one.

### 8.2.2 No activation on a regression output

The torques span −12 to +29 N·m. A tanh on the output layer would clamp every prediction into (−1, 1), and no amount of training could fix it. Activations belong on hidden layers. Classification output layers are also left bare, because `nn.CrossEntropyLoss` applies the softmax internally; adding your own softmax is the classification version of the same mistake.

## 8.3 Tensors

A `torch.Tensor` is a NumPy array with four properties that matter for learning:

| Property | What it is | Why you care |
|---|---|---|
| `.shape` | The dimensions | Most PyTorch bugs are shape bugs |
| `.dtype` | Element type | Models are `float32`; NumPy defaults to `float64` |
| `.device` | `cpu` or `cuda:0` | Tensor and model must be on the same device |
| `.requires_grad` | Track for autograd | How gradients are computed at all |

If you know NumPy you know most of the tensor API: slicing, broadcasting, `@` for matrix multiplication and `.reshape` all behave the same way.

**The first axis is always the batch.** A batch of 64 arm states has shape `(64, 6)`. A single state has shape `(6,)` and must be given a batch axis with `.unsqueeze(0)` before a `Linear` layer will accept it.

The error you will see most often is

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x8 and 6x64)
```

Decode it as follows: the input `mat1` is 64 × 8, the layer's weight `mat2` (transposed) is 6 × 64, and the inner dimensions 8 and 6 disagree. The input has the wrong number of columns. The fix is always the same: print `.shape` on both sides on the line before.

Two further details save time later. `torch.tensor(np_array)` keeps the NumPy dtype, so pass `dtype=torch.float32` explicitly. And on the CPU, `tensor.numpy()` shares memory with the tensor, so changing one changes the other.

Device handling should be a single variable set once:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Every tensor and the model are moved with `.to(device)`. On a machine with a GPU nothing else in the code changes. The case study trains in under ten seconds on a CPU; no GPU is needed for this chapter.

## 8.4 Autograd

Mark a tensor with `requires_grad=True` and PyTorch records every operation applied to it in a **computational graph**. Calling `.backward()` on a scalar walks that graph in reverse and deposits the derivative of the scalar with respect to each tracked tensor in its `.grad` attribute.

Start with something you can check on paper. For f(w) = 3w² + 2w, df/dw = 6w + 2, which is 26 at w = 4:

```python
w = torch.tensor(4.0, requires_grad=True)
f = 3*w**2 + 2*w
f.backward()
print(w.grad)          # tensor(26.)
```

Now the most important operational fact in the chapter: **gradients accumulate**. Run the forward and backward pass a second time without clearing, and `w.grad` becomes 52, not 26. Accumulation is deliberate (it lets you split a large batch across several forward passes), but if you forget to clear gradients in a training loop, each step uses the sum of every gradient you have ever computed, and the loss explodes after a handful of steps. That is why every training loop begins with `optimizer.zero_grad()`. Exercise 5 removes that line and asks you to explain what happens.

Two context tools complete the picture. Inside `with torch.no_grad():` no graph is built, which saves memory and time during evaluation and inference. `.detach()` returns a tensor cut out of the graph, which you use when you want a value without its history.

## 8.5 Backpropagation is the chain rule

Autograd is not magic. It is the chain rule applied node by node. Take a one-hidden-layer network with tanh and mean squared error over N samples and D outputs:

$$
z_1 = XW_1^\top + b_1,\quad h = \tanh z_1,\quad \hat\tau = hW_2^\top + b_2,\quad L = \frac{1}{ND}\sum(\hat\tau-\tau)^2 .
$$

The backward pass is five lines:

$$
\frac{\partial L}{\partial\hat\tau} = \frac{2(\hat\tau-\tau)}{ND},\qquad
\frac{\partial L}{\partial W_2} = \Big(\frac{\partial L}{\partial\hat\tau}\Big)^{\!\top}h,\qquad
\frac{\partial L}{\partial h} = \frac{\partial L}{\partial\hat\tau}W_2,
$$

$$
\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial h}\odot(1-h^2),\qquad
\frac{\partial L}{\partial W_1} = \Big(\frac{\partial L}{\partial z_1}\Big)^{\!\top}X .
$$

The bias gradients are the column sums of ∂L/∂τ̂ and ∂L/∂z₁. Look at the shape of the work: each step multiplies *one local derivative* by *the gradient arriving from the layer above*. No node needs to know anything about the rest of the network. That locality is why a backward pass costs about the same as a forward pass, and why networks with billions of parameters can be trained at all.

The notebook implements these lines in plain tensor algebra and compares them with `.backward()` on a 6–16–2 network and a batch of 128 samples:

| Gradient | Shape | Max \|hand − autograd\| |
|---|---|---|
| ∂L/∂W₁ | (16, 6) | 1.5 × 10⁻⁸ |
| ∂L/∂b₁ | (16,) | 7.5 × 10⁻⁹ |
| ∂L/∂W₂ | (2, 16) | 0 |
| ∂L/∂b₂ | (2,) | 0 |

The hand-derived and automatic gradients agree to floating-point precision.

### 8.5.1 A model-free check: finite differences

When you write a custom layer, verify its gradient with central differences: nudge one weight by ±ε, rerun the forward pass, and compare [L(w + ε) − L(w − ε)]/(2ε) with the autograd value. Checking 60 weights of W₁ with ε = 10⁻³ gives a largest disagreement of about **10⁻⁴**, which is the truncation error of the finite difference, not an autograd error. The check took 120 forward passes for 60 weights, while one backward pass gave all 146 gradients at once. That cost ratio is the reason nobody trains by finite differences.

## 8.6 Building and training the model

### 8.6.1 `nn.Module`

There are two ways to write the same MLP. `nn.Sequential` is a straight pipeline and is enough for a plain MLP. Subclassing `nn.Module` is needed as soon as data flow is anything other than a straight line: skip connections, two inputs, or a physics term added to the network output.

```python
class ArmMLP(nn.Module):
    """6 joint-state numbers in, 2 joint torques out."""
    def __init__(self, n_in=6, n_hidden=64, n_out=2):
        super().__init__()                    # always first
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_out)
        self.act = nn.Tanh()

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        return self.fc3(h)                    # no activation on a regression output
```

`__init__` declares the pieces and `forward` says how they connect. `super().__init__()` must come before any layer is assigned; otherwise PyTorch cannot register the parameters and the optimiser silently trains nothing. Call `model(x)`, never `model.forward(x)`: `__call__` runs registered hooks around `forward`, and bypassing it breaks anything that relies on them.

### 8.6.2 `Dataset` and `DataLoader`

A `Dataset` answers two questions: how many samples are there (`__len__`), and what is sample *i* (`__getitem__`). A `DataLoader` wraps it and yields shuffled mini-batches. For arrays already in memory, `TensorDataset` is a one-liner; for a real robot log you would write a small class that reads from disk inside `__getitem__`.

```python
g = torch.Generator().manual_seed(1234)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, generator=g)
val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False)
# 6000 samples / batch_size 64 = 94 batches per epoch
```

| Knob | What it does | Sensible default |
|---|---|---|
| `batch_size` | Samples per gradient step | 32 to 256 |
| `shuffle` | Reorder every epoch | `True` for training, `False` for validation and test |
| `num_workers` | Background loading processes | 0 for in-memory data, 2 to 8 when reading files |
| `drop_last` | Discard a short final batch | `True` if the model needs a fixed batch size |

Shuffle the training loader so that the network does not learn the order in which the log was recorded. Never shuffle validation or test loaders; it changes nothing about the score and makes per-sample debugging miserable.

> **Reproducibility.** A shuffling `DataLoader` without an explicit `generator` draws its seed from the global torch random state at iteration time. Anything that touches that state earlier, even a single `next(iter(loader))` used to peek at a batch, silently changes the training run. Pass a seeded `generator`, and reseed it immediately before training.

### 8.6.3 The four-line training loop

Every PyTorch project you will write uses the same four lines inside two loops:

```python
for epoch in range(1, epochs + 1):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()              # 1. clear last step's gradients
        loss = criterion(model(xb), yb)    # 2. forward
        loss.backward()                    # 3. backward: fills every .grad
        optimizer.step()                   # 4. update every parameter
    scheduler.step()
    val_loss = evaluate(model, val_loader) # under torch.no_grad(), model.eval()
```

Clear, forward, backward, update, in that order every time. The case study uses mean squared error, the **Adam** optimiser (Kingma and Ba, 2015) with learning rate 3 × 10⁻³, a `StepLR` schedule that halves the rate every 25 epochs, batch size 64 and 60 epochs. Training 4738 parameters takes about nine seconds on a laptop CPU.

`model.train()` and `model.eval()` do not train or evaluate anything. They flip a flag that dropout and batch-normalisation layers read. This network has neither, so the calls change nothing here. Form the habit anyway: the day you add dropout and forget `model.eval()`, your validation loss will be mysteriously noisy.

### 8.6.4 Reading the loss curves

| Epoch | Train MSE | Validation MSE | Learning rate |
|---|---|---|---|
| 1 | 0.608 | 0.311 | 3.0 × 10⁻³ |
| 10 | 0.0116 | 0.0137 | 3.0 × 10⁻³ |
| 30 | 0.0069 | 0.0077 | 1.5 × 10⁻³ |
| 60 | 0.0052 | 0.0059 | 7.5 × 10⁻⁴ |

(MSE is on standardised targets.) Training and validation loss lie on top of each other for all sixty epochs, with a final gap of 0.0007. There is no overfitting, and more data would buy very little. With 4738 parameters and 6000 noisy samples one might have expected otherwise; the smoothness of the target function and the small learning rate late in training keep the network well behaved.

The loss-curve diagnostic is the single most transferable skill in Part III:

| What you see | Diagnosis | What to do |
|---|---|---|
| Both high and flat | Underfitting | Bigger network, train longer, better features |
| Train low, validation higher and rising | Overfitting | More data, weight decay, dropout, early stopping (Chapter 9) |
| Both low and together | Good fit | Stop, and compare with the noise floor |
| Validation spiky or exploding | Learning rate too high | Lower it, or add a schedule |

### 8.6.5 Two knobs that decide everything

The notebook runs two short sweeps of 30 epochs each and reports validation MSE.

| Experiment | Setting | Validation MSE after 30 epochs |
|---|---|---|
| Batch size | 64 | 0.0090 |
| | 512 | 0.0122 |
| | Full batch (6000) | 0.580 |
| Learning rate | 10⁻⁴ | 0.121 |
| | 3 × 10⁻³ | 0.0090 |
| | 10⁻¹ | 0.205 |

Full-batch training has barely moved after 30 epochs. Small batches are not magic; they simply take 94 steps per epoch instead of one. Very small batches eventually cost more than they gain, through noisy gradients and poor hardware utilisation, which is why 32 to 256 is the practical range.

The learning rate matters even more. At 10⁻⁴ the run is correct but still descending after 30 epochs. At 10⁻¹ it bounces and never settles. Without a plot, find a good value with a log-spaced sweep of three or four rates for a few epochs each.

## 8.7 Does it work? The test set, once

Three numbers decide whether the network was worth building: the error in engineering units, the error of a dumb baseline, and the **noise floor**. The torque sensor carries about 6% noise, so no model can predict the *measured* torque more accurately than that, however large it is or however long it trains.

| Model | τ₁ RMSE (N·m) | τ₂ RMSE (N·m) | τ₁ R² | τ₂ R² |
|---|---|---|---|---|
| Linear regression | 4.859 | 2.021 | 0.336 | 0.268 |
| **MLP** | **0.439** | **0.178** | **0.9946** | **0.9943** |
| Sensor noise floor | 0.358 | 0.142 | | |

The MLP cuts the shoulder-torque error by a factor of **11.1** and sits only **1.23 times above the noise floor**. That second ratio tells you what to do next: stop tuning the network. Further gains must come from a better torque sensor or more informative data, not from architecture.

Random test samples are not what a control engineer cares about most. A trajectory is. The notebook generates a smooth four-second pick-and-place move the network has never seen and asks for the feed-forward torque at every instant. Along that trajectory the MLP's RMSE is **0.18 and 0.08 N·m**, against **4.02 and 1.58 N·m** for the linear model. On a plot, the network's prediction is indistinguishable from the true torque; the linear model is off by up to five newton-metres.

Before trusting this on hardware, check extrapolation. The network has seen joint velocities only within the range of the training data. Neural networks interpolate well and extrapolate badly, and nothing in the network warns you when an input is out of range. Exercise 7 quantifies how badly; Chapter 16 returns to input-distribution checks as a deployment requirement.

## 8.8 Save, load, deploy

A trained network is worth nothing until it runs on the controller. Three rules cover most of what goes wrong.

**Save the `state_dict`, not the model object.** `torch.save(model)` pickles your class definition along with the weights, so renaming a file or refactoring `ArmMLP` silently breaks every checkpoint you own. A `state_dict` is an ordered dictionary of tensors that depends only on parameter names.

**The normalisation constants are part of the model.** If the controller standardises with the wrong mean and standard deviation, the network receives garbage and returns garbage, and nothing in the code complains. Ship them in the same file.

```python
ckpt = {
    "state_dict": model.state_dict(),
    "arch": {"n_in": 6, "n_hidden": 64, "n_out": 2},
    "x_mu": x_mu, "x_sd": x_sd, "y_mu": y_mu, "y_sd": y_sd,   # never forget these
    "features": FEATURES, "targets": TARGETS,
}
torch.save(ckpt, "arm_mlp.pt")              # about 22 KB
```

**Infer in evaluation mode, without a graph.** At boot, the controller rebuilds the architecture, loads the weights and calls `.eval()`; every call runs under `torch.no_grad()`.

```python
loaded = torch.load("arm_mlp.pt", weights_only=False)
robot_model = ArmMLP(**loaded["arch"])
robot_model.load_state_dict(loaded["state_dict"])
robot_model.eval()

def feedforward_torque(q1, q2, qd1, qd2, qdd1, qdd2):
    s = np.array([[q1, q2, qd1, qd2, qdd1, qdd2]], dtype=np.float32)
    with torch.no_grad():
        out = robot_model(torch.tensor((s - loaded["x_mu"]) / loaded["x_sd"])).numpy()
    return out[0] * loaded["y_sd"] + loaded["y_mu"]
```

For a test state the reloaded network predicts (17.91, 2.07) N·m against a true (17.96, 2.16) N·m, and its outputs are bit-identical to the trained model's. A single-sample forward pass takes **tens of microseconds** on a laptop CPU (45 µs in the recorded run), a few percent of the 1000 µs budget of a 1 kHz control loop. Latency is machine-dependent, so measure it on the target hardware.

Before closing the loop on a real arm you would add two more things: a saturation limit on the commanded torque, and a check that the input state lies inside the training distribution. Chapter 16 develops both, along with export formats that do not need Python on the robot.

## 8.9 The decisive experiment

Is the network learning the dynamics or merely the training distribution? The decisive experiment is to **step outside the training envelope**: build a test set with shoulder velocities between 3 and 5 rad/s, beyond almost everything in the log, and evaluate (Exercise 7). A model that has learned the physics degrades gracefully; one that has learned the data cloud fails abruptly. The answer determines whether the network can be trusted as a feed-forward term or needs a guard that falls back to the nominal model outside its envelope. The residual formulation of Exercise 8, in which the network learns only the correction to a rigid-body model, is the standard way to make that fallback safe, and it is the bridge to physics-informed models such as Deep Lagrangian Networks (Lutter, Ritter and Peters, 2019).

## Summary

- A neural network is the neuron of Chapter 1, stacked. The activation function is what makes depth more than one linear map.
- Tensors are NumPy arrays that track gradients. Watch shape, dtype and device; the first axis is the batch.
- Autograd applies the chain rule node by node, at about the cost of a forward pass. Gradients accumulate, so clear them every step.
- The training loop is `zero_grad`, forward, backward, step. Every model in the rest of the book uses it.
- Standardise inputs and targets with training statistics only. Put no activation on a regression output.
- Read train and validation loss together. Compare against a dumb baseline and against the noise floor; a model at the noise floor cannot be improved by more training.
- Ship the `state_dict` together with the normalisation constants; load with `.eval()` and infer under `torch.no_grad()`.
- On the arm, a 4738-parameter MLP cuts shoulder-torque error 11-fold versus a linear model and sits 1.23 times above the sensor noise floor.

## Exercises

1. **ReLU versus tanh.** Swap `nn.Tanh()` for `nn.ReLU()` and retrain with the same seed. Compare validation loss, test RMSE and wall-clock time. Plot the prediction along a sweep of q₂ with the other inputs fixed: can you see the piecewise-linear kinks, and do they matter for a torque command?
2. **Capacity sweep.** Train hidden widths 4, 8, 16, 64 and 256 for 60 epochs each and plot final train and validation loss against width on a log axis. Where does underfitting stop? Does the widest network overfit, and if not, why not?
3. **Data efficiency.** Retrain on 250, 500, 1000, 2000 and 6000 samples and plot test RMSE against training-set size. How much logged robot time would you need for the accuracy reached here?
4. **Break the standardisation.** Train on raw inputs and targets with the same learning rate. Describe how it fails and explain why in terms of the first gradient.
5. **Forget `zero_grad`.** Comment out `optimizer.zero_grad()` and train for five epochs. Print the norm of `fc1.weight.grad` at each step and explain the pattern.
6. **Physics-informed features.** Add cos q₂, sin q₂, cos(q₁ + q₂), q̇₁q̇₂ and q̇₂² as extra inputs and retrain the linear model. How close does it get to the MLP, and what does that tell you about what the hidden layers learn?
7. **Extrapolation.** Build a test set with |q̇₁| between 3 and 5 rad/s and evaluate. Quantify the degradation and state what it means for the safety case of putting this network on a real arm.
8. **A residual model.** Compute the rigid-body torque from a nominal model with link masses 10% wrong, and train the network on the residual between that and the measured torque. Compare accuracy and the network size needed against the end-to-end model.

## Further reading

- D. E. Rumelhart, G. E. Hinton and R. J. Williams, "Learning representations by back-propagating errors," *Nature* 323 (1986) 533–536.
- D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," *ICLR* 2015.
- X. Glorot and Y. Bengio, "Understanding the difficulty of training deep feedforward neural networks," *AISTATS* 2010.
- K. Hornik, M. Stinchcombe and H. White, "Multilayer feedforward networks are universal approximators," *Neural Networks* 2(5) (1989) 359–366.
- I. Goodfellow, Y. Bengio and A. Courville, *Deep Learning*, MIT Press, 2016, Chapters 6 and 8.
- D. Nguyen-Tuong and J. Peters, "Model learning for robot control: a survey," *Cognitive Processing* 12(4) (2011) 319–340.
- M. Lutter, C. Ritter and J. Peters, "Deep Lagrangian networks: using physics as model prior for deep learning," *ICLR* 2019.
- B. Siciliano, L. Sciavicco, L. Villani and G. Oriolo, *Robotics: Modelling, Planning and Control*, Springer, 2009, Chapter 7.
- PyTorch documentation: "Learn the Basics" tutorial, autograd mechanics and `torch.utils.data`, <https://pytorch.org/docs/stable/>.
