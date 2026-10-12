# Transformers and attention for machine signals — deck + notebook (generated 2026-10-12)
 
Source: Raschka, Liu & Mirjalili, *Machine Learning with PyTorch and Scikit-Learn*, **ch. 16**
"Transformers – Improving Natural Language Processing with Attention Mechanisms" (book pp. 539–~584,
PDF pages 570–~615 of the full-book scan). Follows the ch. 15 lecture
(`rnn-sequence-modeling-lecture-notes.md`) and **reuses its CNC turning-cell dataset**.
 
## Deliverables
 
- `Transformers_attention_lecture.pptx` — 14 slides (dark title + dark summary, 12 light content
  slides), speaker notes on every slide (1 000–1 360 characters), QA'd via a LibreOffice render at 90 dpi.
  Page size 959.98 × 540 pt.
- `CNC_cycle_transformer.ipynb` — 40 cells (18 code), executed end to end with nbclient, **zero errors,
  zero stderr warnings**, ~4.5–5 min on 2 CPU cores, no downloads.
Audience: **upper-undergraduate engineering** (same as the series). User-chosen extras: **modern LLM
update**, **time-series transformers (patching)**, **attention as explanation**.
 
Scope decisions: kept attention/self-attention/QKV/MHA/transformer/positional encoding/LayerNorm in full
(plus Pre-LN). Bahdanau RNN attention kept only as motivation. **Replaced** the GPT-1/2/3 table and the
GPT-2 demo pinned to `transformers==4.9.1` with an updated landscape slide. NSP mentioned only as dropped
(RoBERTa). IMDb DistilBERT fine-tuning replaced by (a) masked-patch pre-training on unlabeled CNC
cycles + few-label probe, executed; (b) a **non-executed** modern HF `Auto*`/`Trainer` + optional LoRA
recipe for maintenance-log text (the cloud workspace cannot reach the Hugging Face Hub).
 
## Running example
 
CNC generator identical to ch. 15 (see that doc for the error budget), with `make_cycle` now also
returning `meta = {surges, t_chip, bounds}` (same RNG call sequence → same data). Inputs: 4
standardised channels + one-hot op code = 9 features per sample, padded to 150. Splits 800/300/600
(seeds 7/108/209), unlabeled pre-training set 4 000 (seed 999).
 
Model `PatchTransformer`: patches of 10 samples (90 → d=32 linear), learned [CLS], learned positional
embedding, 2 × `TransformerEncoderLayer(32, 4 heads, ff 64, dropout .1, norm_first=True)`
(`enable_nested_tensor=False` avoids a warning), final LayerNorm, head on [CLS]. Adam lr 2e-3,
cosine, 30 epochs, batch 64, clip 1.0. LSTM baseline: ch. 15 open-gate LSTM, lr 3e-3, 30 epochs.
 
## Headline numbers (deck reads `results.json`; notebook reproduces the accuracies exactly)
 
| quantity | value |
|---|---|
| test class balance | 0.245 / 0.497 / 0.258 |
| transformer (23 677 params) | test **0.993**, ≥95 % valid at epoch 7, ~6–7 s |
| transformer, no PE | test **0.993** (op code reveals phase) |
| open-gate LSTM (12 995 params) | test **0.993**, ≥95 % at epoch 10, ~56–60 s → **~9× slower** |
| basic attention loop vs matrix | 2.4e-7 |
| √d_k sweep, d_k=1024 unscaled | max weight **0.97**, grad norm **0.035** (scaled 0.294) |
| manual MHA vs `nn.MultiheadAttention` | max diff **8.9e-8**; 4 224 params |
| causal mask: change tokens 10–14 | past moves **0.27** unmasked, **0.0** masked |
| shuffle patches, logit change | no PE **4.8e-7**, PE 0.25 |
| scrambled-logger task | no PE **51.5 %**, PE **99.8 %** |
| patch 1 / 5 / 10 / 25 | 151 / 31 / 16 / 7 tokens; 22 801 / 961 / 256 / 49 scores; all 0.993; patch 10 ≈ 20× faster than patch 1 |
| pre-training loss (30 ep) | 0.712 → 0.071, ~20 s |
| few-label (24/48/96/192/800) pre-trained probe | 0.610 / 0.789 / **0.921** / 0.979 / 0.992 |
| scratch transformer | 0.497 / 0.693 / **0.814** / 0.986 / 0.993 |
| random-encoder probe | 0.572 / 0.651 / 0.754 / 0.839 / 0.950 |
| masked-only-loss probe | 0.532 / 0.598 / **0.686** / 0.762 / 0.848 |
| [CLS] attention / single-patch occlusion | inclusion **0.363 / 0.822**; surge 0.067 / 0.003; finish 0.079 / ≈0; other 0.051 / 0.002; uniform 0.075 |
| occlude all finish patches | light-finish parts **−0.986**, heavy-finish −0.004 |
 
## Pedagogically load-bearing facts — do not "fix"
 
- **Pre-training loss must include visible patches** (`w_visible=1`). MAE-style masked-only loss gives
  features *worse than a random encoder*; that contrast is slide 11's key lesson.
- **Full fine-tuning from the pre-trained encoder did not beat scratch** with ≤96 labels (tried lr 5e-4,
  1e-4, LP-FT). The deck therefore uses the feature-based route (frozen + sklearn logistic regression on
  mean+max pooled tokens). A torch linear head without standardisation underperformed sklearn — the
  `StandardScaler` matters. Fine-tuning is left as exercise 4.
- **`fit` must use plain Adam** (no weight decay). With AdamW's default wd 0.01 the LSTM collapsed to
  0.645 — the ch. 15 lr window is narrow.
- No-PE model matching the PE model is real (op codes + position-free events) — it motivates the
  scrambled-logger task.
- Inclusion patch index is `(t_chip + 1) // 10` (spike middle); `t_chip // 10` mislabels straddling spikes.
- Group occlusion asymmetry (light finish only) is a genuine learned rule, used on slide 13.
## Deck outline
 
1 Title · 2 one-step path (bottleneck figure) · 3 no-parameter self-attention · 4 Q/K/V + √d_k ·
5 multi-head by hand (code box) · 6 Pre-LN block + causal mask · 7 positional information ·
8 patching and quadratic cost · 9 transformer vs LSTM · 10 three families + "since 2022" box ·
11 pre-train then few labels · 12 HF + LoRA recipe for maintenance logs (code box + freeze vs fine-tune) ·
13 attention vs occlusion · 14 summary (workflow strip + 7 rules)
 
## Notebook sections
 
Title/objectives/kept-dropped table · environment · 1 data · 2 why attention · 3 no-param attention ·
4 √d_k measurement · 5 MHA by hand · 6 block, Pre/Post-LN table, causal mask · 7 model code ·
8 transformer vs LSTM + confusion · 9 positions (shuffle, scrambled-logger, PE table) · 10 patching ·
11 model families + what changed · 12 pre-training + few-label curve · 13 HF recipe (markdown only) ·
14 attention vs occlusion + group occlusion · 15 protocol · 16 eight exercises · 17 references.
 
## Reuse notes
 
- Scripts (session workspace only): `cnc_common.py`, `train_all.py` (→ `results.json`,
  `analysis_cache.pt`, state_dicts `tf_main.pt`, `tf_pre.pt` — state_dicts avoided last lecture's
  pickling gotcha), `make_figs.py` (12 figures + `figs/aspects.json`), `build_notebook.py`, `build_deck.js`.
- PyTorch is not preinstalled; the CPU wheel index is blocked by the proxy, so `pip install torch` from
  PyPI pulls ~3 GB of CUDA wheels (>10 min) — start it in the background first.
- Don't `pgrep -f run_nb.py` in a loop from bash — it matches its own command line.
- Figures: `imshow` of an all-ones mask with a 2-colour `ListedColormap` renders the *low* colour unless
  `vmin=0, vmax=1`; twin-axis label collides with the next panel unless `wspace≈0.5`.
- pptxgenjs: custom 13.333 × 7.5 layout again; `figureFit(s, name, yTop, yBottomMax, maxW, xCenter)`.
 

# Slide Notes

## Slide 1
This slide's job: set up the lecture as a direct continuation of last week. Remind the room of the CNC turning cell from chapter 15: one sequence per machined part, four sensor channels plus the G-code op code, and a label that depends on two events far apart in time — a hard inclusion near the start that chips the tool, and a heavy finishing pass near the end that deflects it. Last week the vanilla RNN failed, and the LSTM only worked after we opened its forget gate. Today we replace recurrence with attention. Point at the topic list: the first half is the mechanism (built by hand, number by number), the second half is the ecosystem — pre-training, the model families, and how to read what a transformer has learned. Mention what we dropped from the book: the GPT-2 download demo pinned to transformers 4.9.1, and the GPT-1/2/3 table that stops at 2020. Ask: who has used a chat assistant this week? Every one of those is the decoder you will see on slide 10. A student may ask whether transformers make RNNs obsolete for control and sensor work — hold that for slide 9, where we measure it.


## Slide 2
This slide's job: motivate attention with the exact failure students saw last week. Left panel: one real cycle from the generator. Point at the vibration spike around sample 25 — the hard inclusion — and then at the finish phase shaded green from about sample 75. The label needs both. Right top: an RNN passes information along a chain; to use the inclusion at the decision, the signal must survive every one of those multiplications, which is the vanishing-gradient story from chapter 15. Right bottom: with attention, the output z reads every input token directly through a weighted sum, so the inclusion is one hop away regardless of where it sits. Historical note for the book's first section: Bahdanau attention added this read-out on top of a bidirectional RNN encoder for translation; we will not implement it, because the transformer subsumes it. Ask the room: what do we pay for path length one? Answer they should reach: every token compares itself with every other, so cost grows with the square of the sequence length — we measure that on slide 8. A student will ask whether the RNN is then useless; no, it is cheap per step and naturally streaming, which matters on embedded controllers.


## Slide 3
This slide's job: show that the core of attention is three lines of linear algebra, before any learned weights. Left: one training cycle with dotted lines cutting it into 10-sample patches; each patch is a 90-dimensional vector (10 samples times 9 features). Right: the 14 by 14 attention matrix computed as softmax of all pairwise dot products. Point out the block structure — finish-phase patches 8 to 10 share weight among themselves, so phase structure appears for free. Then point at column 3: rough-cut patches 2, 4 and 5 put nearly all their weight on patch 3, the one containing a current spike. That is the failure mode — a dot product rewards magnitude, so a large-norm token captures everyone's attention whether or not it is relevant. The notebook verifies the matrix form against an explicit double loop; the difference is 2.4e-7, float32 round-off. Question for the room: if we want patch i to look for something different from what it itself contains, what do we need? Lead them to separate query and key projections — next slide. Expected objection: why softmax and not just normalise? Softmax is differentiable, positive, and sharpens differences, which is what we want for selection.



## Slide 4
This slide's job: introduce the learned version of attention and justify the one constant everyone copies without thinking. Read the equation aloud with the information-retrieval analogy: the query is the search, keys are the index, values are the documents returned. Left panel: as the key dimension grows from 4 to 1024, the unscaled softmax's largest weight climbs from 0.44 to 0.97 — effectively a hard argmax. With the 1/√dₖ factor it stays near 0.25 at every width. Right panel: the gradient reaching the scores collapses about eight-fold unscaled (0.292 to 0.035) and stays flat scaled. The score standard deviation printed in the notebook is 2, 4, 8, 16, 32 — exactly √dₖ. Ask: why √dₖ and not dₖ? Because the variance of a sum of dₖ unit-variance products is dₖ, so dividing by its square root restores unit variance — dividing by dₖ would over-shrink and make attention uniform. A student may ask whether a temperature parameter would do the same; yes, √dₖ is just a sensible fixed temperature.


## Slide 5
This slide's job: demystify nn.MultiheadAttention by rebuilding it. Walk the figure left to right: the 16 tokens (CLS plus 15 patches, 32 dimensions each) are projected into four sets of queries, keys and values, each only 8 wide. Each head runs the scaled dot-product from the previous slide independently, the four 16-by-8 outputs are concatenated back to 32, and one more matrix W_O mixes them. The code box is the whole thing — five lines, using the weights pulled out of PyTorch's own module. The notebook's maximum absolute difference against nn.MultiheadAttention is 8.9e-8, float32 round-off; one layer has 4224 parameters, four 32-by-32 matrices plus biases. Ask the room: does going from one head to four cost more compute? No — the per-head width shrinks, so total work is the same; what changes is that you get four different attention patterns. Expected follow-up: do heads specialise? Sometimes visibly, often not; exercise 2 in the notebook varies the head count. A student may notice the book splits heads by reshaping a tensor rather than slicing — it is the same operation.


## Slide 6
This slide's job: assemble the full block and introduce the decoder's causal mask. Left: read the block bottom to top — tokens plus positions, LayerNorm, multi-head attention, residual add, LayerNorm, a two-layer feed-forward network applied to each token separately, residual add, repeat. The orange lines are the residual paths: each sub-layer learns a correction to its input, the same idea as ResNets in chapter 14. Contrast with the book's figure, which is the 2017 Post-LN arrangement — normalise after the add. Pre-LN normalises before the sub-layer, leaving the residual stream untouched, and is what GPT-2 onwards uses; in PyTorch it is norm_first=True. Why LayerNorm not BatchNorm: our cycles are padded and ragged, and at inference you may classify one part at a time. Right: the causal mask — query i may only attend to keys j up to i. The notebook perturbs tokens 10 to 14 and checks positions 0 to 9: they move by 0.27 without the mask and not at all with it. Ask: where would you need this on a machine? Any forecaster — predicting the next window of spindle load — must not see the future during training. A student may ask about cross-attention: the original decoder also attends to the encoder output; encoder–decoder models on slide 10.


## Slide 7
This slide's job: show that attention is permutation-equivariant, and that whether this matters depends on the task. Left: we take the 15 test cycles that fill all 15 patches, shuffle their patch order, and compare the class logits. Without positional embeddings the change is 4.8e-7 — float32 noise; with learned positions it is 0.25. Now the surprise to discuss: on the diameter task, the model without positions scored 99.3%, identical to the one with them. Ask the room why before revealing it. Answer: the op-code channel tells each patch which phase it is in, and the inclusion looks the same wherever it occurs — content identifies position. Right: a task where order is everything — a data logger sometimes writes its ring buffer out of order; detect scrambled cycles. Without positions: 51.5%, chance. With: 99.8%. Mention the three ways to inject position; RoPE rotates queries and keys so their dot product depends on the relative offset, which generalises better to longer sequences than seen in training. Likely question: should I always add positions? Yes — it is cheap — but test with a shuffle to know whether your model relies on order.


## Slide 8
This slide's job: the most important practical decision when applying a transformer to sensor data — what is a token? Left: four patch sizes. With one sample per token we get 151 tokens including CLS and 22,801 attention scores per head; ten samples per token gives 16 tokens and 256. Orange bars are measured training time for 20 epochs on 2 CPU cores: 73 seconds versus 3.4. Right: accuracy is 99.3% for every patch size on this task. This is the PatchTST idea from 2023 and the same trick as the Vision Transformer cutting images into 16-by-16 patches. Rule of thumb: the inclusion lasts three samples, so it falls into one or two 10-sample patches; a 50-sample patch would dilute it among normal cutting. Ask: what would you pick for a 10 kHz accelerometer on a gearbox? Students should reason that raw samples are hopeless — thousands of tokens — and that patches, or spectral frames, are needed. Objection: doesn't the quadratic cost doom long signals? Patching, FlashAttention (slide 10) and sparse or linear attention variants are the usual answers.


## Slide 9
This slide's job: an honest head-to-head with the strong baseline. Left: validation accuracy per epoch. The transformer (blue) leaves the majority-class floor at epoch 4 and is past 95 % by epoch 7; the LSTM with its forget gate opened (orange) stays at chance until about epoch 7, wobbles, and is past 95 % by epoch 10. Right: both end at 99.3% on the 600 test parts, but the transformer took 6 seconds and the LSTM 56 — the LSTM must process 150 time steps one after another, while the transformer handles 16 tokens in parallel. On a GPU that gap widens dramatically, and that, more than accuracy, is why transformers displaced RNNs. The few remaining errors sit on the ±12 µm tolerance boundary where process noise flips the label — no model can see that. Ask the room: is the transformer better? On accuracy, no; on training cost and on not needing the forget-gate trick, yes. Expected objection: the transformer has nearly twice the parameters. True, but parameter count is not the bottleneck here — sequential depth is. Also note the dashed no-PE curve tracking the full model, which we explained on slide 7.


## Slide 10
This slide's job: compress the book's survey of GPT, BERT and BART into one picture, and update it. Left: the three families differ mainly in their attention mask and their pre-training objective. Encoder-only models like BERT see the whole input in both directions and are trained to fill in masked tokens — good for classification, tagging and embeddings. Decoder-only models like the GPT series use the causal mask from slide 6 and predict the next token — good for generation and few-shot use. Encoder–decoder models like T5 and BART read the input bidirectionally and generate output causally, with cross-attention between them — translation, summarisation, speech recognition. Right: what moved since the book went to press. The book's table ends with GPT-3 in 2020; since then decoder-only chat models dominate, their internals have converged on Pre-LN or RMSNorm, rotary embeddings and grouped-query attention, FlashAttention made long contexts practical, and LoRA made fine-tuning affordable. Be explicit that this field moves fast and students should check current documentation rather than memorise model names. Ask: which family would you use to classify maintenance logs, and which to draft a shift report? Encoder for the first, decoder for the second. A student may ask about vision or robot-policy transformers — same block, different tokens.


## Slide 11
This slide's job: reproduce the central idea of the BERT section — pre-train once, adapt with few labels — on machine data, with no download. Left: pre-training loss on 4,000 unlabeled cycles, 0.71 to 0.071 in 30 epochs, about 19 seconds. We blank 40 % of the patches and predict their raw values. Right: test accuracy versus number of labelled parts. Purple, the frozen pre-trained encoder plus logistic regression: 79% at 48 labels, 92% at 96. Blue, the same transformer trained from scratch: 69% and 81%. Grey, an untrained encoder with the same probe. By 192 labels scratch catches up (99%): pre-training buys label efficiency, not a higher ceiling. The red dashed line is the lesson I most want remembered: compute the loss only on the masked patches, as many recipes do, and the representation is worse than random — the encoder is never asked to keep a rare, unpredictable event like the inclusion in its own token. Ask: what would be a good pre-training task for a robot arm's joint torques? Expected question: why freeze rather than fine-tune? With tens of labels, fine-tuning 24 thousand weights overfits; exercise 4 explores when it starts to win.


## Slide 12
This slide's job: replace the book's IMDb fine-tuning example with the recipe an engineer would actually use today, on in-domain text. Walk the code: AutoTokenizer and AutoModelForSequenceClassification download DistilBERT and attach a fresh three-way classification head on the CLS position — imagine labels like tool damage, coolant, and drive fault for shift-log entries. The map call tokenises; TrainingArguments and Trainer replace the hand-written loop the book shows. The commented line is LoRA through the peft library: freeze the model and train small low-rank adapters on the query and value projections — for DistilBERT that is about one percent of the weights, and for a billion-parameter model it is what makes fine-tuning fit on one GPU. This code is in the notebook as a non-executed cell because it needs the Hugging Face Hub; ask students to run it at home. Bottom boxes: the book's two adaptation strategies. Feature-based is what we did on slide 11; fine-tuning updates everything. Ask: you have 60 labelled log entries — which do you choose? Feature-based, or LoRA with strong regularisation. Expected question: can a decoder model like a chat assistant do this zero-shot? Often yes, and it is worth trying first as a baseline — but a small fine-tuned encoder is cheaper, faster and runs on-premises.


## Slide 13
This slide's job: teach students to distrust a pretty attention map. Left: one undersize part. Blue bars are the CLS token's last-layer attention averaged over heads; the tallest bar sits on the patch with the inclusion, 0.36 on average across the test set against a uniform level of 0.08. Right: averaged over all 600 test cycles, attention (blue) and occlusion (orange) — the drop in the true-class probability when one patch is blanked. For the inclusion they agree: occlusion drop 0.82. Coolant surges get ordinary attention and zero importance; good. Now the trap: finish-phase patches get about uniform attention and essentially zero single-patch occlusion effect. But the finish load decides between in-tolerance and oversize, so it must matter. Blank all finish patches at once and light-finish parts lose 0.99 of their true-class probability, while heavy-finish parts lose only 0.004. Two lessons: the evidence is spread across four or five patches, so removing one changes nothing; and the model learned to treat a visible light finish as the evidence, defaulting to heavy otherwise — an asymmetry no attention map shows. Ask: how would you audit a model before it scraps parts on a production line? Interventions — occlusion, group ablation, counterfactual cycles. Cite Jain and Wallace 2019 for the debate.


## Slide 14
This slide's job: leave students with a procedure, not a pile of facts. Walk the strip once: tokenise by patching, embed with a linear projection plus CLS and positions, encode with Pre-LN blocks, pre-train on the unlabeled cycles every machine already logs, adapt with a probe or fine-tuning — LoRA if the model is large — and verify with interventions rather than attention maps. Then read the seven rules; each corresponds to a measurement they saw today: the quadratic cost on slide 8, the saturating softmax on slide 4, the shuffle test on slide 7, the mask test on slide 6, the masked-only pre-training failure on slide 11, the LSTM comparison on slide 9 and the group occlusion on slide 13. Point to the notebook: every number in the deck is computed there, it runs in about four minutes on a laptop CPU with no downloads, and the eight exercises extend rather than repeat — sinusoidal versus learned positions on longer cycles, heads and layers, fine-tuning versus probing, mask ratio, a causal forecaster, long cycles, and an explanation audit. For next time: a natural follow-on is generative models in chapter 17, or applying this same encoder to remaining-useful-life regression. Final question to the room: which of today's rules would have saved you time on your own project data?


