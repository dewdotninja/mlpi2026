# Chapter 12. Attention and Transformers

*Part III. Deep learning for robot signals and images*

> **Case studies:** (A) the CNC turning cell of Chapter 11, read by a patch transformer; (B) loop closure and patrol forecasting on a warehouse rover
> **Companion notebooks:** [`lectures/lect11/supplement/mlpi26_lect11_nbsup1.ipynb`](../../lectures/lect11/supplement/mlpi26_lect11_nbsup1.ipynb) (CNC cell), [`lectures/lect11/supplement/mlpi26_Lect11_nbsup2.ipynb`](../../lectures/lect11/supplement/mlpi26_Lect11_nbsup2.ipynb) (patrol rover)
> **Lab:** [`labs/lab11/mlpi26_Lect11_Lab.ipynb`](../../labs/lab11/mlpi26_Lect11_Lab.ipynb) (attention: finding the needle in a long sequence)
> **Built from:** Lecture 11 deck, Lecture 11 Supplements 1 and 2 and their speaker notes

Chapter 11 ended with a bottleneck. A recurrent network must carry everything it will ever need through a fixed-width hidden state, and a signal from 100 steps ago has to survive 100 multiplications to reach the output. We made that work on the CNC cell only by hand-initialising a gate. **Attention** removes the bottleneck. The output reads every input token directly through a learned weighted sum, so an event is one hop away wherever it sits in the sequence. The **transformer** is what you get when you build a whole network out of that one operation.

This chapter builds attention by hand, number by number, then assembles the transformer block and applies it to machine signals. It measures where transformers win and, just as important, where the best recurrent model is as good and cheaper. It closes with the ecosystem a 2026 engineer actually meets (pre-training, model families, parameter-efficient fine-tuning) and with a warning about reading attention maps as explanations.

## Learning objectives

After this chapter you should be able to:

1. Compute self-attention by hand, explain the query–key–value formulation, and justify the 1/√d_k scale.
2. Rebuild multi-head attention from its weight matrices and match `nn.MultiheadAttention`.
3. Assemble a Pre-LN transformer block, and apply and test a causal mask.
4. Explain why attention is permutation-equivariant, when positional information matters, and how to test whether a model uses it.
5. Turn a sensor signal into tokens by patching, and reason about the quadratic cost of attention.
6. Compare a transformer with the strongest recurrent baseline fairly, and say when each is the right choice.
7. Pre-train an encoder on unlabelled machine data, adapt it with few labels, and describe the modern fine-tuning recipe (Hugging Face, LoRA).
8. Treat an attention map as a hypothesis and test it with occlusion.

---

## 12.1 One hop to anywhere

Recall the CNC cell of Section 11.2: one sequence per machined part, four sensor channels plus the G-code op code, and a diameter label that depends on a hard inclusion near sample 25 and a heavy finishing pass from about sample 75 onward. In a recurrent network the inclusion's signal must survive every step of the chain to reach the decision. With attention, the output z reads every input token through a weighted sum,

$$
z = \sum_{j} \alpha_j\, v_j ,\qquad \sum_j \alpha_j = 1,
$$

so the inclusion is one hop away regardless of its position. Attention was first added to translation models as a read-out on top of a bidirectional RNN encoder (Bahdanau, Cho and Bengio, 2015). The transformer (Vaswani et al., 2017) drops the recurrence entirely.

The price of a one-step path is that every token compares itself with every other, so the cost grows with the square of the sequence length. Section 12.5 measures it. Recurrent networks are not obsolete: they are cheap per step and naturally streaming, which matters on an embedded controller.

## 12.2 Self-attention from the ground up

### 12.2.1 Attention with no parameters

Cut a cycle into 10-sample **patches**. With 9 features per sample (4 standardised channels plus a one-hot op code), each patch is a 90-dimensional vector x_i. The simplest self-attention has no learned weights at all:

$$
\omega_{ij} = x_i^\top x_j,\qquad \alpha_{ij} = \operatorname{softmax}_j(\omega_{ij}),\qquad z_i = \sum_j \alpha_{ij}\, x_j .
$$

Computed for a 14-patch training cycle, the 14 × 14 matrix α already shows structure: finish-phase patches share weight among themselves, so phase structure appears for free. It also shows the failure mode. Rough-cut patches put nearly all their weight on the one patch that contains a current spike, because a dot product rewards magnitude, and a large-norm token captures everyone's attention whether or not it is relevant. The matrix form agrees with an explicit double loop to 2.4 × 10⁻⁷.

Softmax rather than plain normalisation, because it is differentiable, positive, and sharpens differences, which is what selection needs. But we also want patch i to look for something *different* from what it contains. That needs separate projections.

### 12.2.2 Queries, keys and values

Learned self-attention projects each token three ways, with Q = XW_Q, K = XW_K and V = XW_V:

$$
\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V .
$$

The information-retrieval analogy is exact enough to be useful. The query is the search, the keys are the index, and the values are the documents returned. Because the address is the content itself and the answer is soft, it is a lookup table through which gradients flow.

A four-place example makes it concrete. A rover has stored a four-number IMU fingerprint for each of four places (the keys, and here also the values) and is now feeling a slightly noisy copy of place 3 (the query). Scaled dot products through a softmax give place 3 a weight of **0.83**; the other three share the rest. In a trained transformer W_Q, W_K and W_V are learned, so the network learns *which features to compare*.

### 12.2.3 Why divide by √d_k?

If query and key entries have unit variance, their dot product over d_k dimensions has variance d_k. The notebook confirms the score standard deviation is 2, 4, 8, 16 and 32 for d_k = 4 to 1024, exactly √d_k. Without scaling, the softmax saturates as d_k grows:

| d_k | Largest weight, unscaled | Gradient norm, unscaled | Gradient norm, scaled |
|---|---|---|---|
| 4 | 0.44 | 0.292 | ≈ 0.29 |
| 1024 | **0.97** | **0.035** | 0.294 |

At d_k = 1024 the unscaled softmax is effectively a hard arg-max, and the gradient reaching the scores has collapsed eight-fold. Dividing by √d_k restores unit variance, so the largest weight stays near 0.25 at every width. Dividing by d_k would over-shrink and make attention uniform. √d_k is simply a sensible fixed temperature.

### 12.2.4 Multi-head attention, rebuilt

A transformer runs several attention operations in parallel on narrower projections. With model width d = 32 and 4 heads, each head works in 8 dimensions:

```python
# x: (T, 32) tokens; W_q, W_k, W_v, W_o and biases pulled from nn.MultiheadAttention
q, k, v = x @ W_q.T + b_q, x @ W_k.T + b_k, x @ W_v.T + b_v
heads = []
for h in range(4):
    s = slice(8*h, 8*(h+1))
    a = torch.softmax(q[:, s] @ k[:, s].T / 8**0.5, dim=-1)
    heads.append(a @ v[:, s])
z = torch.cat(heads, dim=-1) @ W_o.T + b_o          # mix the heads back to 32
```

Applied to 16 tokens (a [CLS] token plus 15 patches), this matches `nn.MultiheadAttention` to **8.9 × 10⁻⁸**. The layer has 4224 parameters: four 32 × 32 matrices plus biases. Going from one head to four costs no extra compute, because each head is narrower; what changes is that you get four different attention patterns. Whether heads visibly specialise varies (Exercise 2). In production code, call `F.scaled_dot_product_attention`, which dispatches to fused, memory-efficient kernels such as FlashAttention (Dao et al., 2022).

## 12.3 The transformer block

Read a block bottom to top:

```
tokens + positions
  -> LayerNorm -> multi-head attention -> + (residual)
  -> LayerNorm -> feed-forward (per token) -> + (residual)
  -> repeat
```

Each sub-layer learns a *correction* to its input, carried by residual connections, the same idea as the ResNets of Chapter 10. The feed-forward network is a small two-layer MLP applied to every token separately; attention mixes information *across* tokens, the MLP transforms it *within* each.

**Pre-LN versus Post-LN.** The original 2017 transformer normalised *after* each residual add (Post-LN). Most models since GPT-2 normalise *before* each sub-layer (Pre-LN), which leaves the residual stream untouched and trains more stably without careful learning-rate warm-up (Xiong et al., 2020). In PyTorch:

```python
layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64,
                                   dropout=0.1, batch_first=True, norm_first=True)
encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
```

**LayerNorm, not BatchNorm.** Machine cycles are padded and ragged, and at inference you may classify one part at a time. LayerNorm normalises each token over its own features and does not depend on the batch.

### 12.3.1 The causal mask

A forecaster must not see the future during training. A **causal mask** sets the scores ω_ij to −∞ for j > i before the softmax, so query i attends only to keys up to i. Test it rather than trusting it: perturb tokens 10 to 14 and check the outputs at positions 0 to 9. Without the mask they move by **0.27**; with it, by **0.0**. Decoder models for text, and any model forecasting the next window of spindle load, need this mask.

## 12.4 Position is part of the input

Self-attention is **permutation-equivariant**: shuffle the input tokens and the outputs shuffle the same way, with nothing else changed. A transformer therefore knows nothing about order unless position is added to the tokens. Shuffle the patch order of 15 test cycles and compare the class logits: without positional embeddings the change is **4.8 × 10⁻⁷** (float32 noise); with learned positional embeddings it is 0.25.

Now a surprise. On the diameter task the model *without* positions scored 99.3%, identical to the model with them. Why? The op-code channel tells each patch which phase it is in, and the inclusion looks the same wherever it occurs. Content identifies position. Change the task to one where order is everything, such as detecting cycles that a data logger wrote out of order from its ring buffer, and the picture flips:

| Scrambled-logger detection | Test accuracy |
|---|---|
| No positional information | 51.5% (chance) |
| Learned positional embedding | **99.8%** |

There are three common ways to inject position. **Learned absolute embeddings** add a trained vector per position (used here). **Sinusoidal encodings** (the 2017 original) add fixed sines and cosines of geometrically spaced frequencies. **Rotary embeddings (RoPE)** rotate queries and keys so that their dot product depends on relative offset, which generalises better to sequences longer than those seen in training and is now standard in large language models (Su et al., 2021). Always add positions; it is cheap. And test with a shuffle to know whether your model actually relies on order.

## 12.5 What is a token? Patching and quadratic cost

The most important practical decision when applying a transformer to sensor data is what counts as a token. Raw samples make attention expensive: the work grows as T² per head. Patching groups consecutive samples into one token through a linear projection, the idea behind PatchTST for time series (Nie et al., 2023) and the 16 × 16 image patches of the Vision Transformer (Dosovitskiy et al., 2021).

| Samples per patch | Tokens (incl. [CLS]) | Attention scores per head | Test accuracy |
|---|---|---|---|
| 1 | 151 | 22 801 | 0.993 |
| 5 | 31 | 961 | 0.993 |
| 10 | 16 | 256 | 0.993 |
| 25 | 7 | 49 | 0.993 |

Accuracy is the same at every patch size on this task, but 20 epochs of training took **73 s** with one sample per token and **3.4 s** with ten, a twenty-fold difference. The inclusion lasts three samples, so it falls into one or two 10-sample patches; a 50-sample patch would dilute it among normal cutting. The rule of thumb is to choose the patch so that the shortest event you care about still dominates at least one token. For a 10 kHz gearbox accelerometer, raw samples are hopeless (thousands of tokens per second); patches or spectral frames are needed.

The case-study model, `PatchTransformer`, uses 10-sample patches projected from 90 to d = 32, a learned [CLS] token and positional embedding, two Pre-LN encoder layers with 4 heads and feed-forward width 64, a final LayerNorm and a classification head on [CLS]: **23 677 parameters**.

## 12.6 Transformer versus the best LSTM

The honest comparison is with the strong baseline: the open-gate LSTM of Chapter 11, not the default one.

| Model | Parameters | Test accuracy | Epochs to ≥ 95% validation | Training time |
|---|---|---|---|---|
| Patch transformer | 23 677 | 0.993 | 7 | ≈ 6 s |
| Patch transformer, no positions | | 0.993 | | |
| Open-gate LSTM | 12 995 | 0.993 | 10 | ≈ 56 s |

On accuracy they tie. The few remaining errors sit on the ±12 µm tolerance boundary, where gauge noise flips the label; no model can see that. On cost, the transformer trains about **nine times faster**: the LSTM must process 150 time steps one after another, while the transformer processes 16 tokens in parallel. On a GPU the gap widens further, and that, more than accuracy, is why transformers displaced recurrent networks. The transformer also did not need the forget-gate trick. It has nearly twice the parameters, but parameter count is not the bottleneck here; sequential depth is.

One practical warning from building this comparison: the LSTM's learning-rate window is narrow. With AdamW's default weight decay of 0.01, the same LSTM collapsed to 64.5%. Compare models under the settings that suit each.

### 12.6.1 Lab 11: the needle in a long sequence

Lab 11 makes the length dependence explicit. A robot joint occasionally takes a brief impact, four samples of torque transient in a stream thousands of samples long. The damage depends on the impact's magnitude and on the joint load *at that same instant*; a third channel, temperature, is a pure distractor. The model must find the impact at an arbitrary position and read two channels there. You implement scaled dot-product attention from first principles, verify it against PyTorch, and measure how the LSTM-versus-attention comparison changes as the window grows at a fixed training budget. Because the target does not depend on *where* the impact happened, the lab is also a clean test of when positional information is unnecessary.

## 12.7 Case study B: where transformers earn their reputation

On the CNC cell the transformer won on cost but tied on accuracy. Where does it win on accuracy? When the model must look up *several specific things* in the past, which a fixed-size state cannot hold. Two rover tasks test that hypothesis. Both models share the same front end, a 1-D convolution with kernel 8 and stride 4; only the layer that mixes tokens over time differs. That is the fair comparison.

### 12.7.1 Loop closure: capacity, not distance

A patrol rover samples its IMU at 10 Hz for 128 steps per run. During the run it crosses K floor features, each with a 4-D vibration fingerprint. At the end it crosses a feature again: is this a place it has already seen in this run, or a new one? This is loop closure, a core problem in SLAM. After the convolutional stem each run is 31 tokens. The mixers are a 96-unit open-gate LSTM (61 169 parameters) and two Pre-LN transformer blocks with 4 heads and d = 48 (41 521 parameters). Both trained for 20 epochs on 6000 fresh runs per epoch; with a fixed set of 4000 runs, the transformer overfit to 76%.

| Places per run K | LSTM | Transformer | Oracle (clean fingerprints, best threshold) |
|---|---|---|---|
| 1 | 0.901 | 0.993 | 0.997 |
| 2 | 0.716 | 0.969 | 0.995 |
| 4 | 0.592 | 0.946 | 0.991 |
| 8 | 0.552 | 0.923 | 0.985 |

The transformer tracks the ceiling; the LSTM collapses toward chance as K grows. There are two possible limits on a recurrent model, and they can be told apart. *Path length* is how far information must travel; *capacity* is how much fits in the state. If path length were binding, the LSTM would be good when the matching feature was crossed recently and bad when it was long ago. At K = 4 its AUC is flat across that gap, mediocre everywhere. With 31 tokens, **capacity is the binding limit**: four 4-D fingerprints must share one state. A SLAM engineer would store a descriptor at every keyframe and run nearest-neighbour search at query time. Section 12.9 shows that attention learns essentially that.

### 12.7.2 Patrol forecasting: discovering the lap

A patrol rover repeats a route whose shape is random per run (straights and turns), with ±4% speed variation per lap and noisy measurements. Given two and a half laps of yaw-rate history (150 steps at 5 Hz), forecast the next 40 steps. A perfect forecaster for the next eight seconds only has to look at what happened one lap (60 steps) earlier. A transformer can do that directly; an LSTM must carry a whole lap of a two-channel signal in its state.

The forecaster is *direct*: 40 blank future slots, flagged, are appended to the context, and the model fills them in one pass. That keeps both models' inputs identical and avoids error feedback. (Teacher-forced next-step training never learned to copy; both models stalled near 0.25 to 0.30 RMSE.)

| Forecaster | Mean RMSE over 40 steps |
|---|---|
| Persistence (hold last value) | 0.388 |
| LSTM (113 474 parameters) | 0.267 |
| Copy the previous lap (told the period) | 0.050 |
| **Transformer (50 594 parameters)** | **0.040** |

Persistence is best at the first step and worst by the last. The LSTM's forecast hugs the mean and misses every turn. The transformer follows the turns, and it beats copy-one-lap because it sees two earlier laps and averages them, cancelling part of the measurement noise. It found a better rule than the obvious one: a **6.7-fold** lower error than the LSTM, with fewer than half the parameters.

Its attention maps show how. In layer 1, each future slot attends to the same phase of the previous two laps: 24% of the attention mass falls within two steps of one lap back and 20% near two laps back. Layer 2 spends 40% of its attention on the other future slots, smoothing the forecast across the horizon. The model was never told the period.

Remove the positional embedding and the forecast becomes **identical at every horizon** (spread exactly 0). The 40 future slots receive the same input, and a permutation-equivariant model cannot tell them apart. This is the positional lesson of Section 12.4 in its starkest form.

### 12.7.3 The honest boundary

| Task | Best RNN | Transformer |
|---|---|---|
| Terrain classification (Chapter 11) | GRU 0.963 | 0.970 (4.6× the parameters) |
| Loop closure, K = 1 | LSTM 0.901 | 0.993 |
| Loop closure, K = 4 | LSTM 0.592 | 0.946 |

The advantage grows with how much must be looked up in the past and nearly vanishes when the answer is a running summary such as vibration statistics. Note the caveat on the first row: Chapter 11's *LSTM* reached only 0.818 on terrain, so a transformer would have looked like a big win against the wrong recurrent baseline. Always compare with the best one.

The practical rule: if you can name a summary statistic that answers the question, start with a GRU or even hand features. If the answer needs a lookup of something specific from the past, or several things, attention is worth its T² cost.

## 12.8 Pre-training and the model landscape

### 12.8.1 Three families

The transformers you meet in practice differ mainly in their attention mask and their pre-training objective:

| Family | Attention | Pre-training objective | Good for | Examples |
|---|---|---|---|---|
| Encoder-only | Bidirectional | Fill in masked tokens | Classification, tagging, embeddings | BERT, RoBERTa, DistilBERT |
| Decoder-only | Causal | Predict the next token | Generation, few-shot use, chat | GPT series and most current chat models |
| Encoder–decoder | Bidirectional in, causal out, with cross-attention | Denoise or translate | Translation, summarisation, speech recognition | T5, BART, Whisper |

What changed after 2020 is mostly engineering. Decoder-only chat models dominate. Their internals converged on Pre-LN or RMSNorm, rotary position embeddings and grouped-query attention; FlashAttention made long contexts practical; and low-rank adaptation made fine-tuning affordable. Model names change monthly, so check current documentation rather than memorising them. Vision and robot-policy transformers use the same block with different tokens: image patches, proprioceptive readings, actions.

### 12.8.2 Pre-train on what the machine already logs

Every machine logs far more unlabelled cycles than anyone will ever gauge. The BERT idea transfers directly: pre-train an encoder to fill in masked tokens, then adapt it with a few labels. The notebook blanks 40% of the patches of **4000 unlabelled cycles** and trains the transformer to reconstruct their raw values. In 30 epochs (about 20 s) the loss falls from 0.71 to 0.071.

Then freeze the encoder, pool its token outputs (mean and max), and fit a standardised logistic regression on a few labelled parts:

| Labelled parts | 24 | 48 | 96 | 192 | 800 |
|---|---|---|---|---|---|
| Frozen pre-trained encoder + probe | 0.610 | 0.789 | **0.921** | 0.979 | 0.992 |
| Same transformer from scratch | 0.497 | 0.693 | 0.814 | 0.986 | 0.993 |
| Random (untrained) encoder + probe | 0.572 | 0.651 | 0.754 | 0.839 | 0.950 |
| Pre-trained with masked-only loss + probe | 0.532 | 0.598 | **0.686** | 0.762 | 0.848 |

With 96 labels, pre-training is worth 11 points. By 192 labels the scratch model catches up: **pre-training buys label efficiency, not a higher ceiling.**

The last row is the lesson to remember. Many recipes compute the reconstruction loss only on the masked patches. Do that here and the representation is *worse than a random encoder*. The encoder is never asked to keep a rare, unpredictable event like the inclusion in its own token, because a visible patch never contributes to the loss. Include the visible patches in the loss. The standardisation in the probe also matters; an unstandardised linear head underperformed.

Full fine-tuning of all 23 677 weights from the pre-trained encoder did not beat training from scratch with 96 labels or fewer, across several learning rates and a linear-probe-then-fine-tune schedule. With tens of labels, fine-tuning overfits; Exercise 4 explores when it starts to win.

### 12.8.3 Fine-tuning text models for maintenance logs

The same two strategies apply to text: classify shift-log entries into, say, tool damage, coolant and drive fault. The modern recipe uses the Hugging Face libraries and, for large models, **LoRA** (Hu et al., 2022), which freezes the model and trains small low-rank adapters on the attention projections:

```python
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
# from peft import LoraConfig, get_peft_model

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3)
# model = get_peft_model(model, LoraConfig(r=8, target_modules=["q_lin", "v_lin"],
#                                          task_type="SEQ_CLS"))   # ~1% of weights trainable

ds = ds.map(lambda b: tok(b["text"], truncation=True), batched=True)
args = TrainingArguments("out", num_train_epochs=3, learning_rate=2e-5,
                         per_device_train_batch_size=16)
Trainer(model=model, args=args, train_dataset=ds["train"],
        eval_dataset=ds["validation"], processing_class=tok).train()
```

This needs the Hugging Face Hub, so the notebook shows it without running it. With 60 labelled log entries, choose the feature-based route or LoRA with strong regularisation. A general-purpose chat model used zero-shot is often a reasonable *baseline* to try first, but a small fine-tuned encoder is cheaper, faster and runs on-premises, which matters for plant data.

## 12.9 Attention is a hypothesis; occlusion tests it

Attention maps are tempting explanations. Treat them as hypotheses.

On the CNC cell, the [CLS] token's last-layer attention for an undersize part peaks on the patch containing the inclusion: **0.363** on average across the test set, against a uniform level of 0.075. Test it with **occlusion**, the drop in the true-class probability when one patch is blanked:

| Patch type | Mean [CLS] attention | Single-patch occlusion drop |
|---|---|---|
| Inclusion | 0.363 | **0.822** |
| Coolant surge | 0.067 | 0.003 |
| Finish phase | 0.079 | ≈ 0 |
| Other | 0.051 | 0.002 |

For the inclusion the two agree. Coolant surges get ordinary attention and zero importance, correctly. Now the trap. Finish-phase patches get about uniform attention and essentially zero single-patch occlusion effect. But the finish load decides between in-tolerance and oversize, so it must matter. Blank *all* finish patches at once, and light-finish parts lose **0.986** of their true-class probability while heavy-finish parts lose only 0.004. Two lessons follow. The evidence is spread across four or five patches, so removing one changes nothing. And the model learned to treat a *visible light finish* as the evidence and to default to "heavy" otherwise, an asymmetry no attention map shows.

The rover loop-closure model tells the same story from the other side. On positive runs, 28% of the last-layer attention lands on the matching feature, 29% is shared by the other features, 22% goes to background and 21% to the query's own tokens; the arg-max lands on the match 88% of the time. Occluding the matching feature drops the mean "seen before" probability from **0.91 to 0.12**, and 85% of positives flip to "new place". Occluding a different feature leaves it at 0.91. Here the map and the intervention agree.

Attention weights can be changed substantially without changing predictions (Jain and Wallace, 2019), and the debate about what they mean continues (Wiegreffe and Pinter, 2019). Before a model scraps parts on a production line, audit it with interventions: single and group occlusion, and counterfactual cycles. Chapter 17 returns to explainability as part of responsible deployment.

## 12.10 The decisive experiment

For a transformer, the decisive experiments are **interventions on its inputs**: shuffle the tokens to learn whether it uses order, apply the causal-mask test to prove it cannot see the future, and occlude single patches and groups to learn what it uses. Each costs seconds. On the CNC cell they revealed that positions were unnecessary for the diameter task, that the finish-phase evidence was distributed, and that the model had learned an asymmetric rule. None of that was visible in its accuracy or its attention map.

## Summary

- Attention gives every output a one-step path to every input. The price is O(T²) work per head.
- Attention(Q, K, V) = softmax(QKᵀ/√d_k)V is a soft, content-addressed lookup. The √d_k keeps the softmax from saturating: unscaled, the largest weight reached 0.97 at d_k = 1024 and the gradient collapsed eight-fold.
- A transformer block is Pre-LN attention and a per-token MLP, each with a residual connection. Test a causal mask by perturbing future tokens.
- Attention is permutation-equivariant. Add positions and test with a shuffle; a forecaster without them produced identical forecasts at every horizon.
- Patch sensor signals. Ten-sample patches cut training time twenty-fold at no loss of accuracy on the CNC cell.
- Against the best LSTM, the transformer tied on accuracy (99.3%) and trained nine times faster. It won on accuracy when several things had to be looked up (loop closure 94.6% against 59.2% at K = 4; forecasting error 6.7 times lower), and not on running-summary tasks.
- Pre-training on unlabelled cycles bought label efficiency (92% against 81% with 96 labels), but only when the loss included visible patches.
- Attention maps are hypotheses; occlusion tests them.

## Exercises

1. **Sinusoidal versus learned positions.** Replace the learned positional embedding with the fixed sinusoidal one and repeat the scrambled-logger task. Then train on cycles of length ≤ 130 and test on 140 to 150. Which scheme copes better?
2. **Heads and layers.** Train with 1, 2, 4 and 8 heads (keeping d = 32) and with 1 to 4 layers. Which matters more, and how does the epoch-to-95% change?
3. **Without the op code.** Remove the op-code features. Does the no-position model still match the full model on the diameter task? Explain using Section 12.4.
4. **Fine-tuning versus probing.** Fine-tune all weights of the pre-trained encoder with 96 and 192 labels at learning rates 10⁻⁴ and 5 × 10⁻⁴, and try linear-probe-then-fine-tune. When does fine-tuning beat the frozen probe?
5. **Mask ratio.** Repeat pre-training with mask fractions 0.15, 0.4 and 0.75 and plot probe accuracy at 48 labels against the mask fraction.
6. **A causal forecaster.** Build a decoder-style model that predicts the next patch's mean spindle current from the patches so far. Verify with the test of Section 12.3.1 that it cannot see the future.
7. **Long cycles.** Generate cycles of 1000 to 1500 samples with the inclusion near t = 100 and the finish near the end. Compare patch sizes 10 and 50 for accuracy, time and memory. What limits a patch size of 1?
8. **Give the LSTM a fair chance.** On loop closure with K = 4, increase the LSTM's hidden size to 256 and train for 40 epochs. How much of the gap closes? Plot accuracy against parameter count for both models.
9. **A variable patrol timer.** Draw the lap period from 45 to 80 steps per run. Does the transformer still forecast well? What replaced the fixed-lag stripes in its attention map?
10. **Explanation audit.** For 20 misclassified or low-confidence CNC test parts, show [CLS] attention, single-patch occlusion and group occlusion side by side. Write a two-sentence diagnosis for each and say which method found the cause.

## Further reading

- D. Bahdanau, K. Cho and Y. Bengio, "Neural machine translation by jointly learning to align and translate," *ICLR* 2015.
- A. Vaswani et al., "Attention is all you need," *NeurIPS* 2017.
- R. Xiong et al., "On layer normalization in the transformer architecture," *ICML* 2020.
- J. Devlin, M.-W. Chang, K. Lee and K. Toutanova, "BERT: pre-training of deep bidirectional transformers for language understanding," *NAACL* 2019.
- A. Dosovitskiy et al., "An image is worth 16x16 words: transformers for image recognition at scale," *ICLR* 2021.
- Y. Nie, N. H. Nguyen, P. Sinthong and J. Kalagnanam, "A time series is worth 64 words: long-term forecasting with transformers," *ICLR* 2023.
- J. Su et al., "RoFormer: enhanced transformer with rotary position embedding," arXiv:2104.09864, 2021.
- T. Dao, D. Y. Fu, S. Ermon, A. Rudra and C. Ré, "FlashAttention: fast and memory-efficient exact attention with IO-awareness," *NeurIPS* 2022.
- E. J. Hu et al., "LoRA: low-rank adaptation of large language models," *ICLR* 2022.
- K. He et al., "Masked autoencoders are scalable vision learners," *CVPR* 2022.
- S. Jain and B. C. Wallace, "Attention is not explanation," *NAACL* 2019; S. Wiegreffe and Y. Pinter, "Attention is not not explanation," *EMNLP* 2019.
- Hugging Face Transformers and PEFT documentation, <https://huggingface.co/docs>.
