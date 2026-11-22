# Chapter 11. Sequence Models for Machine and Robot Data

*Part III. Deep learning for robot signals and images*

> **Case studies:** (A) predicting the finished diameter of a turned part from its machining cycle; (B) heading, terrain and wheel slip on a differential-drive rover
> **Companion notebooks:** [`lectures/lect10/supplement/mlpi26_lect10_nbsup1.ipynb`](../../lectures/lect10/supplement/mlpi26_lect10_nbsup1.ipynb) (CNC cell), [`lectures/lect10/supplement/mlpi26_lect10_nbsup2.ipynb`](../../lectures/lect10/supplement/mlpi26_lect10_nbsup2.ipynb) (rover)
> **Lab:** [`labs/lab10/mlpi26_Lect10_Lab.ipynb`](../../labs/lab10/mlpi26_Lect10_Lab.ipynb) (forecasting tool wear with an LSTM)
> **Built from:** Lecture 10 deck, Lecture 10 Supplements 1 and 2 and their speaker notes

Every model so far has assumed that training examples are independent: shuffle them and nothing is lost. A sequence breaks that assumption by definition. Sample 25 of a machining cycle means something different from sample 100, and a gyro reading from ten seconds ago still determines where a robot is pointing now. Convolution gave us a model that respects locality in space. Recurrence gives us one that respects order in time.

This chapter uses two machines. A CNC turning cell supplies the cleanest possible test of long-range memory: two faults about 100 samples apart that cancel each other. A small rover then shows that robotics already runs a recurrent network every day, dead reckoning, and asks what a *learned* one can add. Along the way we will measure, rather than assert, why plain recurrent networks fail on long dependencies, and we will see the most commonly recommended fix, the LSTM, fail too until one line of initialisation is changed.

## Learning objectives

After this chapter you should be able to:

1. Choose the sequence shape (many-to-one, one-to-many, synchronised or delayed many-to-many) from where the answers are needed, and read the right output tensor for it.
2. Write the forward pass of a recurrent network by hand and match `nn.RNN` to floating-point precision.
3. Explain backpropagation through time, why gradients vanish or explode, and measure a model's **gradient reach**.
4. Explain the LSTM and GRU, and initialise their gates open so that a long dependency is learnable at all.
5. Batch ragged sequences correctly with padding, packing or bucketing, and embed categorical channels.
6. Run the ablation that tests whether a model really uses the part of the sequence you think it does.
7. Compare recurrent models with a physics baseline, hand features and a dilated 1-D CNN, and report honestly when they do not win.

---

## 11.1 What makes data sequential

A sequence model is defined less by its architecture than by the **shape** of the question. Four shapes cover almost everything:

| Shape | Input → output | Machine example | Which output you read |
|---|---|---|---|
| Many-to-one | Sequence → one label | Quality call on a finished part; terrain class for a run | The final hidden state |
| One-to-many | One vector → sequence | Generate an expected cycle from a part number, as a reference trace for anomaly detection | Every step of a generated sequence |
| Synchronised many-to-many | One output per input step | Per-sample "tool in cut" flag; per-step wheel-slip flag; heading at every step | Every step, with the loss masked by true length |
| Delayed many-to-many | Sequence → offset sequence | Forecast the next 40 steps; predict the next sample | Outputs after the input ends |

The shape is not a philosophical choice. It decides whether you write `h_n[-1]` or `out`, and reading the wrong one is half the bugs in student sequence code. Shape and memory are separate decisions, too. Wheel slip and heading have the same shape (one output per step), but slip at step *t* can be decided from steps *t* − 2 to *t*, while heading at step *t* depends on every gyro reading since the start. A model that is excellent at one can be useless at the other.

Two rules follow from the loss of independence. **Never shuffle within a sequence**: shuffle whole sequences, not their samples. And **split by sequence, not by sample**, for the reasons of Section 7.3.3: windows cut from the same run are near-duplicates.

## 11.2 Case study A: two faults, 100 samples apart, that cancel

A CNC turning cell logs four analogue channels at 50 Hz for every part it makes: spindle current (A), vibration RMS (g), spindle temperature (°C) and normalised feed rate, plus a categorical `op_code` naming the G-code block (approach, rough, dwell, finish, retract). Cycles last 110 to 150 samples, so batches are ragged for a physical reason. After machining, the part is gauged. The target is the finished-diameter class: `undersize`, `in tolerance` or `oversize`.

The data generator encodes an error budget with two terms of opposite sign:

- A **hard inclusion** in the stock at about sample 25 micro-chips the cutting edge. A chipped edge rubs instead of cutting, the workpiece heats, and after it cools the feature is **undersize** by about 21 µm.
- A **heavy finishing pass** at about sample 100 deflects the tool away from the work and leaves the feature **oversize** by about 21 µm.

With 4 µm of gauge noise and a ±12 µm tolerance band, the outcomes are:

| | No heavy finish | Heavy finish |
|---|---|---|
| **No inclusion** | In tolerance | Oversize (scrap) |
| **Inclusion** | Undersize (scrap) | In tolerance: the errors cancel |

Error budgets with terms of opposite sign are ordinary in precision machining. What is unusual is only that the two terms are about **103 samples apart**, and the label depends on both. The inclusion raises current *and* vibration together, while benign coolant surges scattered through the cycle raise current only, so *detecting* the event is a threshold on vibration. *Remembering* it is the hard part, by design. Nothing after the inclusion carries any trace of it.

One more design detail is load-bearing. A binary scrap/no-scrap label would be a pure XOR of the two faults, with zero marginal correlation with either one, and no model can get a gradient until it has discovered both at once. The three-class *signed* target gives each fault a marginal signal (an inclusion pushes toward undersize), while the scrap decision remains an XOR. The splits are 800 training, 300 validation and 600 test cycles. The majority class, `in tolerance`, is **47.7%** of the test set.

## 11.3 The recurrent network

A feed-forward layer maps x to h. A recurrent layer adds one edge: a loop on h.

$$
h^{(t)} = \tanh\!\left(W_{xh}\,x^{(t)} + W_{hh}\,h^{(t-1)} + b_h\right),\qquad o^{(t)} = W_{ho}\,h^{(t)} + b_o .
$$

Unfolding the loop across time steps is a way of drawing, not a different model. There is one h and one set of weights, visited repeatedly. **None of the matrices carries a t.** The same W_hh is applied at sample 3 and at sample 130. That is how a 4471-parameter RNN can read a 150-sample cycle, and it is the same inductive-bias argument as for convolution: share parameters along the axis where the structure repeats.

For an 8-dimensional input (four channels plus a 4-dimensional op-code embedding) and 48 hidden units, the recurrent layer holds 8 × 48 + 48 × 48 + 2 × 48 = 2784 parameters. Note the *two* bias vectors. Mathematically b_xh + b_hh is one bias; PyTorch keeps them separate to keep input and recurrent paths symmetric, which matters once each gate of an LSTM has its own pair.

Why tanh and not ReLU? The recurrent path applies the same non-linearity hundreds of times, and ReLU is unbounded, so the state can run away. A bounded activation keeps the state in range, at the cost of saturating gradients.

### 11.3.1 The forward pass by hand

There is nothing inside `nn.RNN` that fifteen lines cannot reproduce:

```python
rnn = nn.RNN(input_size=4, hidden_size=2, batch_first=True)
W_xh, W_hh = rnn.weight_ih_l0, rnn.weight_hh_l0
b_xh, b_hh = rnn.bias_ih_l0,  rnn.bias_hh_l0

prev_h = torch.zeros(2)
outs = []
for t in range(x.shape[1]):
    ot = torch.tanh(x[0, t] @ W_xh.T + b_xh + prev_h @ W_hh.T + b_hh)
    outs.append(ot)
    prev_h = ot                     # the complete memory of the network
```

The hand-computed hidden states match `nn.RNN` to **6 × 10⁻⁸**, float32 machine precision. Read the last line again. `prev_h` is the entire memory of the network. Everything it knows about the first hundred samples must be in those numbers when it reaches sample 101.

### 11.3.2 Dead reckoning is a recurrent network

Before going further, note that robotics has been using this structure for decades. Integrating a gyro to get heading,

$$
\psi_t = 1\cdot\psi_{t-1} + \Delta t\cdot\omega_t ,
$$

is a recurrent network with one hidden unit, recurrent weight 1, input weight Δt and no non-linearity. Kalman filters, complementary filters and PID integrators all carry a state forward in the same way. A learned RNN replaces a hand-written transition with a learned one.

The recurrent weight being exactly 1 is the crucial property. If it is ρ < 1, an impulse decays as ρᵏ: with ρ = 0.95 the half-life is 13.5 steps, with ρ = 0.8 it is 3.1. A tanh RNN that wants to integrate must keep its effective gain at 1 for hundreds of steps. That is exactly the knife edge of the next section.

## 11.4 Why recurrent networks forget

### 11.4.1 Backpropagation through time

Training a recurrent network unfolds it over the sequence and backpropagates through the unfolded graph: **backpropagation through time (BPTT)**. The gradient of a loss at time t with respect to the state at an earlier time k contains

$$
\frac{\partial h^{(t)}}{\partial h^{(k)}} = \prod_{i=k+1}^{t} \frac{\partial h^{(i)}}{\partial h^{(i-1)}}
= \prod_{i=k+1}^{t} \operatorname{diag}\!\left(1-\big(h^{(i)}\big)^2\right) W_{hh},
$$

a product of t − k Jacobians, one per step travelled back. Products of many matrices do one of two things: shrink to nothing or blow up. In the one-unit case the factor is w^(t−k). At w = 0.97 a gradient is down to 2% after 130 steps; at w = 1.03 it is up by a factor of 45. A 3% error in one weight becomes a fifty-fold error over one machining cycle.

Measured on the real model by scaling W_hh at initialisation and taking the gradient norm on one batch (nothing trained):

| Spectral radius ρ(W_hh) | 0.33 | 0.99 | 1.21 | 1.43 | 1.65 |
|---|---|---|---|---|---|
| Gradient norm | 0.45 | 0.56 | 0.65 | **13.3** | **2.8 × 10⁴** |

The threshold is sharp. PyTorch's default initialisation puts this RNN at ρ ≈ 0.55, comfortably in the **vanishing** regime. That is the more common failure and the quieter one: nothing errors, nothing diverges, the loss simply does not move. Why not initialise at ρ = 1 exactly? Orthogonal and identity initialisations do just that, but the Jacobian also includes the tanh derivative, which is at most 1 and usually much less, so the product still contracts.

### 11.4.2 Gradient reach, measured

Here is a diagnostic worth stealing for every sequence project. Take an untrained cell, run it forward 130 steps on random input, put a loss on the *last* hidden state only, backpropagate, and record the gradient norm at every earlier step, normalised by the last. Twelve repetitions, median. It takes about a second and a half.

| Steps back | 10 | 50 | 100 |
|---|---|---|---|
| Vanilla RNN | 5.6 × 10⁻³ | 4.4 × 10⁻¹³ | **0.0** (underflow) |
| LSTM, default initialisation | 2.2 × 10⁻³ | 2.8 × 10⁻¹¹ | 4.3 × 10⁻²¹ |
| LSTM, forget gate opened | 2.1 × 10⁻² | 2.2 × 10⁻² | 3.8 × 10⁻² |

At ten steps back, all three are within a factor of ten of each other. Over short distances everything works, which is why recurrent networks have a reputation for working. At 100 steps back the vanilla RNN's gradient is *exactly* zero; float32 underflowed. There is no learning signal at all, not a small one. Switching to float64 buys a little more underflow headroom and changes nothing that matters.

The surprise is the second row. A default-initialised LSTM vanishes too, ten orders of magnitude better than the RNN and still useless. "Use an LSTM for long-range dependencies" is repeated everywhere and, with default initialisation at this distance, it is simply false. The third row is the same LSTM with one line changed (Section 11.5.2). Our inclusion lives at about 103 steps back.

### 11.4.3 Clipping and truncation

**Gradient clipping** rescales the gradient whenever its norm exceeds a threshold:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # after backward(), before step()
```

With W_hh inflated 2.6-fold to provoke explosion, the raw gradient norm reached 88.6 (median 12.4), and every step was rescaled to 1.0. Clipping costs nothing and turns an exploding run into a slow one. It does nothing for vanishing gradients.

**Truncated BPTT** backpropagates only k steps at a time to save memory on long sequences. It is a memory decision with an accuracy consequence: **never truncate shorter than the dependency you need.** With k = 60, the inclusion at step 25 can never influence a loss at step 130 (Exercise 4).

## 11.5 Gated cells: LSTM and GRU

### 11.5.1 A path with no matrix on it

In a vanilla RNN every path from the past to the present passes through a weight matrix and a non-linearity, so the gradient is a product of Jacobians. The **long short-term memory (LSTM)** cell (Hochreiter and Schmidhuber, 1997) adds a second state, the **cell state** c, and nothing multiplies it except an element-wise gate:

$$
\begin{aligned}
f_t &= \sigma(W_f[x_t, h_{t-1}] + b_f) &&\text{forget gate}\\
i_t &= \sigma(W_i[x_t, h_{t-1}] + b_i) &&\text{input gate}\\
g_t &= \tanh(W_g[x_t, h_{t-1}] + b_g) &&\text{candidate}\\
o_t &= \sigma(W_o[x_t, h_{t-1}] + b_o) &&\text{output gate}\\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t &&\text{the conveyor}\\
h_t &= o_t \odot \tanh(c_t) &&\text{the read-out}
\end{aligned}
$$

Trace c through time: one element-wise multiply, one add, straight through. That is the whole invention. Everything else decides what to write onto that conveyor and what to take off. The forget gate decides how much old cell state survives, the input gate and candidate decide what new information is written, and the output gate decides how much of the cell is exposed as h. Only h goes to the next layer. Because the forget gate is a sigmoid, bounded by 1, the cell state can be preserved or shrunk but never amplified.

The original 1997 cell had no forget gate. Gers, Schmidhuber and Cummins (2000) added it because the cell state grew without bound on long sequences. The gate that fixes our problem is a bug fix from three years after the original paper.

The **gated recurrent unit (GRU)** (Cho et al., 2014) merges forget and input into one update gate and drops the separate cell state. For our 8 → 48 layer, the recurrent parameters are RNN 2784, GRU 8352 and LSTM 11 136.

### 11.5.2 Initialise the forget gate open

At initialisation PyTorch draws every bias near zero, so the forget gate sits at σ(0) = 0.5. The cell halves its memory at every step. Over 103 steps that is 0.5¹⁰³ ≈ 10⁻³¹. The information is gone before training starts.

| Forget-gate bias | Gate value at init | Gate^103 |
|---|---|---|
| 0 (default) | 0.500 | 9.9 × 10⁻³² |
| 1 | 0.731 | 9.7 × 10⁻¹⁵ |
| **2** | **0.881** | **2.1 × 10⁻⁶** |
| 3 | 0.953 | 6.7 × 10⁻³ |

A bias of 2 gives 25 orders of magnitude more signal, enough for the optimiser to work with. There is a chicken-and-egg reason this must be set by hand. The LSTM can perfectly well *learn* to hold its forget gate open, but to learn that it needs a gradient from the distant past, and that gradient has vanished because the gate is closed. It cannot bootstrap.

```python
def open_gates(rnn, bias=2.0):
    """Set the forget-gate (LSTM) or update-gate (GRU) bias slice."""
    H = rnn.hidden_size
    for name, p in rnn.named_parameters():
        if "bias" in name:
            with torch.no_grad():
                p[H:2*H].fill_(bias / 2)      # b_ih + b_hh together give `bias`
```

PyTorch stores LSTM gate biases in the order input, forget, cell, output, so the forget slice is `[H:2H]`. A GRU stores reset, update, new, and its update gate occupies the same slice. Jozefowicz, Zaremba and Sutskever (2015) measured this across thousands of architectures and recommend a forget bias of at least 1. This is not folklore. Why not set it to 10 and never forget? Then the cell can never clear itself, and on a long sequence everything blends together (Exercise 3).

The learning rate interacts with this. At 3 × 10⁻³ the gates-open LSTM learns the task; at 10⁻² with the same bias it collapses back to chance. If you change the optimiser, re-check.

## 11.6 Ragged batches and categorical channels

### 11.6.1 Pad, then pack

A tensor is rectangular, so a batch of 110- to 150-sample cycles must be padded to its longest member. The trap comes next. If you pass the padded tensor to `nn.LSTM` and read `h_n[-1]`, you get the state after stepping through the padding: for the shortest cycle, forty steps of zeros processed after the cycle ended. On a toy batch, the final hidden state of a 4-sample sequence padded to 10 differs by **0.356** with and without packing, on values of order 0.2. It is a different vector.

`pack_padded_sequence` carries the true lengths so the layer stops each sequence at its own end:

```python
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

x = pad_sequence(seqs, batch_first=True)                       # (B, T_max, F)
packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                              enforce_sorted=False)
out, (h_n, c_n) = lstm(packed)                                  # h_n is now correct
```

With `enforce_sorted=False`, PyTorch sorts and unsorts internally and returns `h_n` in the original order. The lengths must be on the CPU even when the data is on a GPU.

The rover data refines the picture. With plain padding, a *forward* LSTM's output at the last real step is exactly right, by causality: the padding comes later. But `h_n` keeps stepping through zeros (error 0.16), and a *bidirectional* layer reads the padding first in its backward direction (error 0.10). Packing fixes both to float precision. The catch is speed: on a CPU a packed LSTM batch took 0.85 s against 0.10 s unpacked. The rover's run lengths are multiples of 10 steps, so the notebook **buckets** runs by length and never pads at all. Bucketing is not a bias, provided the order of batches is shuffled each epoch.

Masking the loss is equivalent to packing for a synchronised many-to-many loss, but it does not stop the recurrent *state* being corrupted by padding. Packing or bucketing fixes it at the source.

### 11.6.2 Embedding a categorical channel

The `op_code` channel takes one of five values. One-hot would do for five; `nn.Embedding(5, 4)` is the general answer, a trainable lookup table that you need as soon as the vocabulary is tool IDs or part numbers (Section 9.9). The embedding is concatenated with the four standardised analogue channels to give the 8-dimensional input. Fit the per-channel scaler on the training split only, and apply it before padding.

## 11.7 Six models on the CNC cell

Every model reads the packed 8-channel sequence into a 48-unit recurrent layer, followed by a 48 → 32 → 3 head. Training uses Adam at 3 × 10⁻³ with cosine annealing, batch size 64, 22 epochs, gradient clipping at 1.0 and cross-entropy loss.

| Model | Parameters | Train | Validation | **Test** | Time (s) |
|---|---|---|---|---|---|
| Vanilla RNN | 4 471 | 0.527 | 0.477 | **0.477** | 20 |
| LSTM, default initialisation | 12 823 | 0.527 | 0.477 | **0.477** | 53 |
| LSTM, gates open | 12 823 | 0.994 | 0.997 | **0.980** | 52 |
| GRU, gates open | 10 039 | 0.994 | 0.997 | **0.982** | 44 |
| Bidirectional LSTM, gates open | 25 495 | 0.994 | 0.993 | 0.978 | 106 |
| Dilated 1-D CNN (receptive field 255) | 20 599 | 0.934 | 0.953 | 0.923 | 9 |

Four lessons.

1. **The vanilla RNN never leaves the majority class.** 47.7% is exactly the in-tolerance rate. This is not a tuning failure; its gradient at the relevant distance is zero.
2. **The default-initialised LSTM does the same**, to the digit. One line of initialisation takes it to 98%. The result is robust across seeds (0.477, 0.488 and 0.477 for seeds 1 to 3). If you remember one thing from this chapter, make it this.
3. **The bidirectional LSTM does not win**, at twice the parameters and twice the time. It is natural to assume that reading backward makes the early event easy, and it does, for the backward pass. But that same backward pass must then carry the *late* factor all the way back. Each direction has one easy end and one hard end. Bidirectionality earns its keep when evidence is spread through the middle, not pinned at the ends (Exercise 5).
4. **A dilated 1-D CNN reaches 92.3% in a sixth of the time.** Seven dilated convolutions give a 255-sample receptive field with no recurrence, so the gradient path back to sample 25 is seven layers, not 103 time steps. For fixed-horizon tasks this is a strong modern baseline, and it connects directly to Chapter 10. Its weakness is the fixed receptive field: a 400-sample cycle would need more layers.

Should you always use a GRU, since it beat the LSTM with fewer parameters? The difference, 98.2% against 98.0% on 600 test cycles, is within noise. The defensible claim is that the GRU is cheaper and did not lose, which matches the literature (Chung et al., 2014).

### 11.7.1 Prove the information is where you think

We have claimed all chapter that the label depends on an event 100 samples before the end. That is a claim about the *data*, and it is falsifiable. Train the same gates-open LSTM, with the same recipe, on only the **last 45%** of every cycle. If the information really is early, this must fail.

| Input | Test accuracy |
|---|---|
| Whole cycle | 0.982 |
| Last 45% only | **0.507** |
| Majority class | 0.477 |

The late-window model is at chance, even though the late window contains the entire finishing pass, one of the two factors. One factor alone is worth nothing, exactly as the 2 × 2 table predicted.

The best model's per-cell accuracy is 100% when neither fault is present, 100% when both are, 96.1% with the inclusion only and 97.1% with the heavy finish only. The single-fault cells are where combining the two ends matters, and also where the irreducible label noise lives: with 4 µm of gauge noise against a 12 µm band, about 2% of single-fault parts land inside tolerance anyway. So 96 to 97% is near the Bayes rate. As a scrap/no-scrap decision, the model is right **98.2%** of the time.

The habit generalises. Any time you claim a model learned a long-range dependency, hide the range and check it breaks. Any time you claim it uses a channel, remove the channel. Had the late-window model scored 80%, something late would be correlated with the early event, and the long-range story would be partly wrong.

### 11.7.2 Generating a cycle

The second CNC project is a language model with load levels instead of characters. Discretise the spindle current into **48 bins**, and "predict the next character" becomes "predict the next load bin": same embedding, same cross-entropy over a vocabulary, same sampling. Trained on 2177 chunks, the loss falls from 2.944 to 1.586, a perplexity of **4.9 effective bins out of 48**.

The discretisation looks odd to an engineer, but it buys a full *distribution* over the next value. The model can represent "either still cutting at 11 A, or the tool has just retracted to 1 A". A squared-error forecaster cannot; it averages the two and predicts 6 A, a value that never occurs (Exercise 7). Generate by sampling from the predicted distribution, not by taking the arg-max, which collapses into repetition:

```python
logits = model(context)[:, -1] * alpha          # alpha = 1 / temperature
next_bin = torch.distributions.Categorical(logits=logits).sample()
```

At α = 0.5 the distribution is flattened and the generated trace jumps between bins constantly. At α = 2.0 it is sharpened and gives long plateaux with clean transitions, which is what a real cycle looks like. A generator like this produces reference traces for anomaly detection and synthetic cycles for testing downstream logic.

## 11.8 Case study B: a rover that needs memory

The second case study runs every idea again on a different machine: a differential-drive rover sampled at 10 Hz for runs of 8 to 20 s. Seven channels are logged: commanded linear and angular velocity, left and right wheel speeds, the z-axis gyro, and forward and vertical acceleration. The rover drives on concrete, grass or gravel, each with its own vibration signature, and wheels slip in episodes that add 0.10 to 0.35 m/s to one or both encoders (8.5% of steps on concrete, 21% on grass, 31.7% on gravel). Each run also has its own **gyro bias**, drawn from N(0, 0.04) rad/s. There are 900 training, 300 validation and 600 test runs.

Three questions, three kinds of memory:

| Question | Shape | Memory needed |
|---|---|---|
| Where is the rover pointing? | Synchronised many-to-many (regression) | The whole run: heading is an integral |
| What is it driving on? | Many-to-one (classification) | Accumulated evidence |
| Is a wheel slipping now? | Synchronised many-to-many (classification) | Mostly the last few steps |

### 11.8.1 Heading: a learned sensor fusion

Two physics baselines are available. Integrating the gyro drifts linearly because of the bias. Wheel odometry drifts in steps, one per slip episode. Mean heading error at the end of a run, with the same 48-unit hidden size and 30 epochs for every network:

| Estimator | Heading RMSE at run end (rad) |
|---|---|
| Wheel odometry | 0.66 |
| Gyro integration | 0.59 |
| Vanilla RNN | 1.33 |
| LSTM, gates open | 0.21 |
| **GRU, gates open** | **0.13** |

The vanilla RNN is worse than either sensor alone; it has not learned to integrate. The gated networks beat *both* sensors by doing what an engineer would do with a Kalman filter: while the wheels agree with the gyro, the difference between them estimates the gyro bias; during slip, the wheels are ignored. Nobody told the network this. It found it from the loss. A 0.13 rad error is about 7° after up to 20 s with a biased gyro and slipping wheels.

What the network does not give you is a covariance, a guarantee, or interpretability by construction. A Kalman filter gives you all three. Exercise 8 asks you to build one and compare. It is an honest fight.

### 11.8.2 Gradient reach after training

The same gradient-reach measurement as Section 11.4.2, now the gradient of the final heading with respect to the gyro reading k steps earlier, averaged over 64 runs of 200 steps. A perfect integrator gives a flat line at 1: a reading from 19 s ago matters as much as one from now. At initialisation all three collapse (RNN 1.2 × 10⁻²⁴, LSTM 9.6 × 10⁻⁸, GRU 2.5 × 10⁻⁹ at lag 100). Even an LSTM does not *start* with long memory; it starts with a path along which long memory can be learned. After training, the LSTM and GRU curves are close to flat (LSTM 0.35 at lag 10, 0.25 at lag 190). The vanilla RNN's curve is erratic and at the longest lags exceeds 1 (1.40 at lag 190): with only W_hh and tanh it can forget or amplify, but not hold.

When a sequence model underperforms, plot this curve before tuning anything else.

### 11.8.3 Looking inside the LSTM

Writing the LSTM step by hand (it matches `nn.LSTM` to 6 × 10⁻⁷) exposes the cell states. Correlating each of the 48 cell units with the true heading, **19 exceed |r| = 0.9** and the best reaches r = 0.974: rescaled by a straight-line fit, that one unit traces the heading almost exactly. The heading is stored in a distributed way, which is typical.

Then the honest surprise. The textbook picture says a memory unit should hold its forget gate near 1, a pure integrator. The best unit's forget gate averages **0.76**, with a minimum of 0.54. The network keeps the heading alive through the loop from c to h and back into the candidate g. Gradient descent finds *a* memory mechanism, not necessarily the one we would design.

### 11.8.4 Terrain: belief accumulates, and five features tie

Terrain is many-to-one. Because the model is causal, feeding the trained GRU the first k steps of each run gives exactly the decision it would make online after k steps. Accuracy rises from 36.8% after 0.5 s (chance is 33%) to 91.7% after 8 s, and mean confidence rises with it. The hidden state behaves like a belief that accumulates evidence.

| Terrain classifier (full runs) | Test accuracy |
|---|---|
| Vanilla RNN | 0.795 |
| LSTM | 0.818 |
| GRU | 0.963 |
| **Five hand features + logistic regression** | **0.972** |

The five features are what an engineer would compute from vertical vibration: standard deviation, lag-one autocorrelation, kurtosis, speed-normalised standard deviation and mean absolute difference. They tie the best network and beat the other two. Most network errors confuse gravel with concrete, which differ only in occasional impulses that have to be counted. Teach this as a positive result: knowing when *not* to use a recurrent network is part of the skill. Notice also the gap between the LSTM and the GRU on the same task; Chapter 12 shows why it matters which recurrent baseline you compare against.

### 11.8.5 Slip: causal is enough

Slip detection is synchronised many-to-many. A hand rule flags a step when wheel and gyro yaw rates disagree by more than 0.25 rad/s or the wheels read 0.12 m/s above the command.

| Slip detector | Per-step F1 | Parameters |
|---|---|---|
| Hand rule | 0.665 (recall 0.578) | |
| Causal LSTM | **0.972** | 12 545 |
| Bidirectional LSTM | 0.968 | 25 025 |

The single threshold cannot adapt to speed and turning; the network can. The bidirectional model again does not win, at twice the cost. The evidence (wheels disagreeing with the gyro) is present *at the moment of slip*, so the future adds nothing: the causal model's median onset latency is zero steps, and it missed 3 of 988 episodes. A fault recognisable only by its aftermath, such as a bump identified by the recovery that follows, is where looking ahead would help, and only if you can afford to wait for it (Exercise 5 of the rover notebook).

## 11.9 Forecasting tool wear (Lab 10)

Lab 10 applies the same machinery to forecasting. A carbide insert's flank wear VB follows a three-phase curve (break-in, steady state, accelerated breakdown) and the insert is swapped when VB crosses 0.55 mm, so the machine log is a sawtooth of tool lives back to back. VB cannot be measured during production, but four sensor channels can. The task: given the last window of readings, forecast VB 20 parts ahead. Forecast too low and a worn tool cuts a scrap part; too high and a tool with life left is thrown away.

The lab makes three of this chapter's habits concrete.

- **Turn a log into supervised pairs without leaking the future.** Slide a window along the log and take the target a fixed horizon after its end; off-by-one errors here leak the answer into the input. Split by *log*, not by window, because neighbouring windows overlap almost entirely.
- **Beat an order-blind baseline.** An MLP on per-channel mean, standard deviation, minimum and maximum cannot see order by construction. The same values in the order "high then low" (the tool was just replaced; future wear is near zero) and "low then high" (a fresh tool wearing in) demand opposite forecasts.
- **Prove the model uses order.** Feed the trained LSTM test windows with their time steps randomly permuted. The summary-statistics MLP is unaffected by construction, which makes it a built-in control. The LSTM's error on windows containing a tool change rises about eightfold: the information it relied on was the order, and tool changes are where order is worth most.

Finally, judge the forecast as a decision ("change the insert within the next 20 parts?"), not as a mean absolute error in micrometres.

## 11.10 The decisive experiment

The decisive experiment for a sequence model is the **ablation that could falsify your story**: hide the part of the sequence you believe matters and check that performance collapses. On the CNC cell, hiding the first 55% of each cycle dropped accuracy from 98.2% to 50.7%, which is chance, and converted "the model learned a long-range dependency" from an assertion into a measurement. Its companion is the gradient-reach curve, which costs seconds and would have told you on day one that the default LSTM could not see 100 steps back.

## 11.11 Where recurrence runs out

Every model in this chapter funnels the past through a fixed-width vector: 48 numbers for 130 samples of a machining cycle, or for a whole rover run. That was enough for one heading angle or one remembered event, and we made it work by hand-initialising a gate so that a signal could survive being multiplied a hundred times. What happens when a robot must remember several places it has seen, or a whole lap of its route, to answer a question? A fixed-size state starts to run out. Attention stops pretending a single vector is enough: it lets the read-out look at every step directly and learn weights over them, so the gradient path back to sample 25 becomes one multiplication. That is the subject of Chapter 12.

## Summary

- Pick the sequence shape from where the answers are needed. It decides which output you read and what the loss covers.
- A recurrent network shares one set of weights across time. Dead reckoning, Kalman filters and PID integrators are recurrences too.
- BPTT multiplies Jacobians, so gradients vanish (more common, silent) or explode. Measure gradient reach before choosing an architecture.
- An LSTM with default initialisation failed exactly like a vanilla RNN on a 103-step dependency (47.7%). Opening the forget gate with a bias of 2 took it to 98.0%.
- Clip gradients always; it does not help vanishing. Never truncate BPTT shorter than the dependency.
- Pad and pack, or bucket by length. Reading `h_n` from a padded batch returns a state corrupted by zeros.
- Bidirectional models helped on neither machine: each direction has one hard end, and slip evidence is present at the moment of slip.
- Always beat the physics baseline. Gated RNNs fused gyro and wheels better than either (0.13 against 0.59 rad), but five hand features tied the best network on terrain.
- Run the ablation that could falsify your claim.

## Exercises

1. **Two-layer forward pass.** Extend the manual computation of Section 11.3.1 to `nn.RNN(..., num_layers=2)`. Which weights does layer 2 use, and what is its input at step t? Verify against `nn.RNN` to float32 precision.
2. **Find the breaking distance.** Regenerate the CNC data with the inclusion at 10%, 30%, 50% and 70% of the cycle and plot test accuracy against the gap for the default and gates-open LSTMs. Where does the default version fail, and does that match the gradient-reach curve?
3. **Sweep the gate bias.** Train the LSTM with forget-gate bias 0, 0.5, 1, 2 and 4. Is there an upper limit, and what does going too far cost?
4. **Truncation, measured.** Implement truncated BPTT with k = 10, 30, 60 and 130 on the gates-open LSTM. Confirm that accuracy collapses once k is below the gap, and report the wall-clock saving.
5. **Make the bidirectional model win.** Modify the CNC generator so the decisive evidence sits in the middle of the cycle. Does the bidirectional LSTM now beat the unidirectional one?
6. **Synchronised many-to-many.** Predict a per-sample "tool in cut" flag instead of one label per cycle. Use the full output sequence, a loss masked by true length, and no `h_n`. Report accuracy on the padded region to prove the mask works.
7. **Regression instead of bins.** Replace the 48-bin language model with an MSE next-value predictor and roll it out autoregressively. What does it do at phase transitions, and why?
8. **A learned Kalman filter.** Build a two-state Kalman filter (heading, gyro bias) for the rover that uses the wheel yaw rate as a measurement, ignored when the slip label says slip. How close does it get to the GRU, and what does the GRU have that the filter does not?
9. **A physics feature.** Add the integrated gyro as an eighth rover input. Does the vanilla RNN now succeed on heading? What does that say about what was hard?
10. **Wrap-around.** Replace the heading target ψ with (cos ψ, sin ψ) and report angular error. Which runs improve?

## Further reading

- S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation* 9(8) (1997) 1735–1780.
- F. A. Gers, J. Schmidhuber and F. Cummins, "Learning to forget: continual prediction with LSTM," *Neural Computation* 12(10) (2000) 2451–2471.
- Y. Bengio, P. Simard and P. Frasconi, "Learning long-term dependencies with gradient descent is difficult," *IEEE Transactions on Neural Networks* 5(2) (1994) 157–166.
- R. Pascanu, T. Mikolov and Y. Bengio, "On the difficulty of training recurrent neural networks," *ICML* 2013.
- K. Cho et al., "Learning phrase representations using RNN encoder–decoder for statistical machine translation," *EMNLP* 2014.
- J. Chung, C. Gulcehre, K. Cho and Y. Bengio, "Empirical evaluation of gated recurrent neural networks on sequence modeling," arXiv:1412.3555, 2014.
- R. Jozefowicz, W. Zaremba and I. Sutskever, "An empirical exploration of recurrent network architectures," *ICML* 2015.
- S. Bai, J. Z. Kolter and V. Koltun, "An empirical evaluation of generic convolutional and recurrent networks for sequence modeling," arXiv:1803.01271, 2018.
- S. Thrun, W. Burgard and D. Fox, *Probabilistic Robotics*, MIT Press, 2005. Dead reckoning and Kalman filtering.
- PyTorch documentation: `nn.RNN`, `nn.LSTM`, `nn.GRU` and `torch.nn.utils.rnn`, <https://pytorch.org/docs/stable/nn.html#recurrent-layers>.
