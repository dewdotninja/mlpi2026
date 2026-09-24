# Classifying images with deep CNNs — deck + notebook (generated 2026-09-24)
 
Source: Raschka, Liu & Mirjalili, *Machine Learning with PyTorch and Scikit-Learn*, **ch. 14**
"Classifying Images with Deep Convolutional Neural Networks". Follows the ch. 13 lecture
(`pytorch-mechanics-lecture-notes.md`) in the same series.
 
## Deliverables
 
- `CNN_image_classification_lecture.pptx` — 14 slides (dark title + dark summary, 12 light
  content slides), speaker notes on **every** slide (1430–1960 characters each), QA'd via a
  LibreOffice render at 90 dpi. Page size 959.98 × 540 pt.
- `PCB_solder_CNN.ipynb` — 53 cells (26 code), executed end to end with nbformat + nbclient,
  all outputs embedded, **zero errors and zero stderr warnings**, ~4.5 min of training inside
  a ~7 min total runtime on 2 CPU cores.
Audience level: **upper-undergraduate engineering** (same as the previous three lectures).
Scope decisions by the user: replace both downloads (MNIST and the 1.5 GB CelebA, whose link
the book itself calls unstable) with one synthetic in-domain dataset; **add** a CNN-vs-MLP
parameter/accuracy comparison, first-layer filter and feature-map visualisation, and a
transfer-learning note (explanatory only, not executed).
 
## Running example
 
**An inline AOI (automated optical inspection) station on an SMT line.** 64×64×3 crops
centred on a 0603 chip-component footprint, four classes: `good`, `insufficient`, `bridge`,
`excess`. Everything is drawn analytically with soft (anti-aliased) masks — no image library,
no download.
 
Geometry (pixels in the 64×64 frame): `PAD_DX, PAD_HW, PAD_HH = 14.0, 6.5, 10.0`;
`BODY_HW, BODY_HH, CAP_W = 10.0, 8.5, 4.0`; `SCAN_ROW = 38`; `EDGE = 0.9` (AA softness).
Class shapes: insufficient = ellipse at `0.80/0.76` of the pad, offset `1.8` px outward;
excess = ellipse at `1.18/1.10`; bridge = `_rect(8.5, 1.7)` band at `cv=5.4`, alpha `0.92`.
Clutter (0–2 neighbouring footprints, a via at p=0.6, silkscreen text at p=0.5) is drawn
**before** the pads so it can be occluded.
 
### The regime split — this is the load-bearing design decision
 
```python
REGIME = {
  "narrow":    dict(rot=4.0,  shift=2.5, scale=(0.97,1.03), gain=(0.95,1.08),
                    grad=0.06, noise=(0.020,0.030)),
  "full":      dict(rot=15.0, shift=9.0, scale=(0.88,1.12), gain=(0.70,1.30),
                    grad=0.25, noise=(0.020,0.060)),
  "canonical": dict(rot=0, shift=0, scale=(1,1), gain=(1,1), grad=0,
                    noise=(0.004,0.004)),   # mechanism demos only
}
```
 
Train 480 + validation 240 are drawn `narrow` (seeds 7 and 108); test 800 is drawn `full`
(seed 209). So **validation comes from the same afternoon as training, and the test set is
production**. That single choice is what makes the augmentation section work and what gives
the deck its punchline.
 
Models: CNN `Conv(3→16)/pool → Conv(16→32)/pool → Conv(32→64)/pool → Conv(64→64) → GAP →
Dropout → Linear(64,4)`, 60 772 params. MLP `Flatten → 12288→256 → 256→128 → 128→4`,
3 179 396 params. Adam `lr=2e-3`, `CosineAnnealingLR(T_max=90)`, batch 32, **90 epochs**,
`CrossEntropyLoss` on raw logits.
 
## Headline numbers (deck and notebook agree exactly — both read `results.json`)
 
| model | params | train | valid (narrow) | **test (production)** |
|---|---|---|---|---|
| dense MLP | 3 179 396 | 0.992 | 0.958 | **0.3787** |
| CNN, no regularisation | 60 772 | 1.000 | 1.000 | 0.8425 |
| CNN + dropout(0.3) + L2(1e-4) | 60 772 | 0.998 | 1.000 | 0.8938 |
| **CNN + dropout + L2 + augmentation** | 60 772 | 0.994 | 1.000 | **0.9875** |
 
| quantity | value |
|---|---|
| CNN over MLP | **+46.4 points** with **52.3× fewer** parameters |
| dropout + L2 gain | +5.1 points |
| **augmentation gain** | **+14.5 points** (3× the regularisation gain) |
| 1-D gap kernel response | good +0.2599 · insufficient +0.2601 · excess +0.2600 · **bridge +0.0090** |
| pooling, 1-px shift, values bit-identical | raw 0.03 % · 2×2 31.5 % · **4×4 53.6 %** · 8×8 68.2 % |
| conv vs dense params (3→16 ch on 64×64) | 448 vs 805 371 904 → **1 797 705×** |
| best-model confusion (800 images) | good 199/200 · insufficient 200/200 · bridge 199/200 · excess 192/200 (8 → insufficient) |
| scan-row profile | pads ≈0.727 (0.645 for insufficient) · gap 0.138, but **0.680 for bridge** |
 
## Pedagogically load-bearing facts — do not "fix" these
 
- **The regime split is the whole lecture.** If training and test are drawn from the same
  regime the plain CNN reaches 99 % and there is nothing left for dropout or augmentation to
  demonstrate. Verified: with everything `full`, all four models exceed 0.99.
- **Large translation is the lever that defeats the MLP**, not rotation. An earlier draft used
  four 90° feeder orientations; that made the task hard for *both* architectures (a plain CNN
  is translation-equivariant but not rotation-equivariant), muddying the parameter-sharing
  lesson. `shift=9.0` with `rot=15.0` is the setting that separates them cleanly.
- **Dropout + L2 costing nothing and augmentation winning big is the honest result**, and it
  is the section's point: the failure here is a *distribution* gap, not a variance problem.
  Do not tune dropout until it "wins" — the lesson is that it should not.
- **Validation accuracy saturates at 1.000 for all three CNNs**, so it cannot select a
  checkpoint. Final epoch after cosine annealing is used for every run — deterministic, fair,
  and the saturation is itself taught as the warning sign. An earlier best-checkpoint rule
  made the comparison a lottery (plain CNN spuriously beat both regularised runs).
## Deck outline
 
1. Title (dark)
2. Four classes, and a trap built into the data — narrow vs full sample grid
3. A kernel is a matched filter — 1-D scan row + response + code box
4. The same idea with two indices — conv schematic + output-size table
5. Nine numbers make a feature detector — Sobel/Laplacian/blur on a bridge + code box
6. Pooling buys invariance, for no parameters — max/mean worked example + shift test
7. Channels, and the parameter argument — kernel-tensor schematic + 448 vs 805 M
8. Putting it together in torch.nn — architecture strip + full `nn.Sequential`
9. Choosing the loss, and the one bug everybody hits — wrong/right code box + pair boxes
10. Dropout and L2 — and what they cannot fix — dropout schematic + the ladder
11. Augmentation manufactures the missing data — augmented vs production rows
12. Validation passed everything. The line did not. — confusion matrix + val/test bars
13. Nobody specified these kernels — learned conv1 filters + feature maps
14. What to take away (dark) — workflow strip + 7 rules
## Notebook sections
 
Environment · the data (inline renderer, the regime table) · 1-D convolution (naive `conv1d`
vs `np.convolve`, the zero-sum gap kernel, cross-correlation) · 2-D convolution (naive
`conv2d` vs `scipy.signal.convolve2d`, padding modes, output-size table checked against
`nn.Conv2d`) · kernels as feature detectors · pooling (+ the shift-invariance measurement) ·
channels and parameter counting · loss functions (BCE/BCEWithLogits, NLL/CrossEntropy, the
softmax-twice block) · building the CNN (shape walk with a dummy batch, why GAP not Flatten) ·
regularisation (hand-built L2, dropout train/eval demo) · augmentation (pure-torch
`affine_grid`/`grid_sample`, with the `torchvision.transforms.v2` equivalent shown) · the four
experiments + results table + confusion matrix · what the network learned · transfer-learning
note · the 10-rule protocol · **8 exercises** · references.
 
## Notes for reuse
 
- Scripts live only in the session workspace: `pcb_common.py` (renderer), `train_all.py` (all
  analyses → `results.json` + `analysis_cache.pt`), `make_figs.py` (12 figures, loads the
  cache so layout iterations never retrain), `build_notebook.py`, `build_deck.js`.
- **`nbformat`/`nbclient` installed fine this time** (PyPI was returning 503 during the ch. 13
  session, which is why that lecture used a hand-rolled `nbtools.py`). Use the real kernel
  when it is available — `%%time` and IPython display work properly.
- **`torchvision` is not installed** and is not worth the download: the data is synthetic
  tensors, so augmentation is done with `F.affine_grid`/`F.grid_sample`, which is also more
  instructive. The `transforms.v2` spelling is shown in a markdown block.
- **Notebook namespace gotcha that cost a full run:** the 2-D convolution demo used
  `X` and `W` at module level, clobbering the global image width `W = 64`, so every later
  `render_joint` call died in `np.mgrid[0:H, 0:W]`. Renamed to `IMG_H`/`IMG_W` and
  `X_demo`/`W_demo`. Also `n1 = n2, c_in, c_out, m = 64, 3, 16, 3` is a chained-assignment
  trap — it binds the *tuple* to `n1`.
- **Figure-layout lesson, new this time:** matplotlib schematic cells drawn on `xlim/ylim
  0..1` are not square on the page. Use separate `cw` and `ch = cw * fig_w / fig_h` or the
  grid cells render as wide rectangles (fig 03).
- pptxgenjs gotcha, fourth lecture running: `LAYOUT_16x9` is 10 × 5.625 in — always
  `defineLayout({name, width: 13.333, height: 7.5})`. The `figureFit(s, name, yTop,
  yBottomMax, maxW)` helper from the ch. 13 deck was reused and is worth keeping; the one
  overflow this time was a figure+codeBox slide where the code box needs ~1.05 in below the
  figure, so `yBottomMax` must be ~5.85, not 6.3.
- Only **2 CPU cores** in this session; the four 90-epoch runs take ~4.5 min. If a future
  lecture needs more training, budget accordingly or cut epochs.
- Palette identical to the previous three decks (blue `#2a78d6`, orange `#eb6834`, aqua
  `#1baf7a`, yellow `#eda100`, violet `#4a3aa7`, red `#e34948`, ink `#0b0b0b`/`#52514e`,
  surface `#fcfcfb`), light content slides, dark title/summary (`#141413`, accent `#3987e5`).
## Natural follow-on lectures
 
Recurrent networks and sequence models (ch. 15) — the next chapter, and the mechanics from
ch. 13 carry over unchanged. Also available from this same dataset: object detection and
segmentation (the AOI station really wants a bounding box, not just a label); transfer
learning done properly, which is only sketched here; and cost-sensitive decision rules, seeded
as notebook exercise 8 (a missed bridge costs ~200× a false alarm).
 
# Slide Notes

## Slide 1
Opening slide. Two things to establish before any mathematics.

First, what a CNN is for. Chapter 13 gave us dense networks, which connect every input to every unit. For a 64 by 64 colour image that is 12 288 inputs, and a dense first layer prices itself out immediately. More importantly it throws away the one thing we know about images: nearby pixels belong together, and a defect looks the same wherever it appears. A convolutional layer builds both of those facts into the architecture.

Second, the running example. An AOI station sits at the end of the reflow oven and photographs every joint on every board. This is the single most deployed computer-vision task in manufacturing, and the four classes on this slide come straight out of IPC-A-610, the standard that defines what an acceptable solder joint is. Insufficient solder makes a weak joint that fails later on thermal cycling; a bridge is a short and the board does not work at all.

Say that the images are synthetic, rendered from a geometric model of the footprint, and say why: it runs offline in minutes, and we know the true generating process, which is what lets us make honest claims later about what a model should have been able to learn.

Question to open with: 'if you had 480 labelled images and a deadline, what would you try first?' Hold their answers — by slide 12 we will have measured four of them, and the ranking is not the one most people guess.


## Slide 2
Spend real time here — every later slide refers back to this picture.

Walk the four rows. Good: a concave fillet climbing the metallised end cap, with a bright specular streak because a concave surface reflects the ring light as a line. Insufficient: less solder, so bare gold pad is still visible and the glint is a small round spot. Bridge: the silver band running across the gap between the pads — that is a short. Excess: a convex blob spilling past the pad edge, with a round hot spot because a convex surface reflects as a point.

Now the left-right split, which is the real content of this slide. On the left, the labelled set: the part sits near the centre, the lighting is even, rotation is a few degrees. On the right, production: nine pixels of registration error, fifteen degrees of rotation, the lamp anywhere from 70 to 130 percent of nominal, and clutter from neighbouring footprints, vias and silkscreen intruding at the edges.

Say plainly that this is not a contrived difficulty. Labelled defect data always comes from a narrow slice, because somebody had to stand at a bench and confirm each label, and they did it on one afternoon on one line. This is the normal situation, not the awkward one.

Question to the room: 'we will hold out a validation set. Where should it come from?' The instinctive answer is 'split off 20 percent of the labelled images'. Let them say it; on slide 12 we will see what that costs.

Expect: 'why not just label more data?' Good instinct, and exercise 7 measures exactly that trade-off — how many production-like labels are worth as much as augmentation.


## Slide 3
The slide that makes 'convolution' concrete before any 2-D notation appears.

Left panel: the mean intensity along one scan row, averaged over 24 perfectly registered parts of each class. Read the numbers off it. Every class has bright solder shoulders at about 0.73. Three of the four have a dark gap at 0.14 — that is the black ceramic body showing between the pads. The bridge has a bright gap at 0.68, because solder has wicked across it.

So the detector writes itself: positive weights where we expect bright, negative where we expect dark. Right panel: three curves land on top of each other at plus 0.260 and the bridge collapses to plus 0.009. A single threshold at 0.134 separates them, on data the filter was never fitted to, because it was not fitted at all — we designed it.

The code box carries the two details worth writing down. First, zero-sum. Subtracting the mean makes the filter give exactly zero on any flat patch whatever its brightness, so it reports structure and ignores the lamp. That property is why the first layer of a trained CNN almost always discovers near-zero-sum kernels. Second, the reversal. Convolution flips the filter; cross-correlation does not. PyTorch implements cross-correlation and calls it convolution. It makes no difference when the weights are learned — you learn the flip — but it matters the moment you hand-build a kernel, as here.

Question to the room: 'this filter already solves one of our four classes. Why train a network at all?' Answers worth drawing out: it needs the part perfectly registered, it needs someone to know which row and which mechanism, and it does nothing for the other three classes. A CNN learns hundreds of these and learns where to apply them.


## Slide 4
The mechanics slide. Keep it brisk and make sure the formula lands.

Walk the picture left to right. One output pixel is one dot product between a 3 by 3 patch of the input and the 3 by 3 kernel, summed. Slide the kernel one step and you get the next output pixel. That is the entire operation. The two properties in the caption are the whole argument for the architecture: sparse connectivity means an output pixel sees only nine inputs, not all 4096; parameter sharing means the same nine weights are used at every position.

Then the formula: o equals floor of n plus 2p minus m over s, plus one. Work the three rows aloud. A 3 by 3 kernel with padding 1 and stride 1 gives 64 out of 64 — that is same padding, and it is what we will use everywhere. Padding 0 gives 62; do that eight times and you have lost 16 pixels to nothing. A 2 by 2 window with stride 2 gives 32 — that is the pooling layer, and the same formula covers it.

Mention the three padding modes by name: valid is p equals zero, same is chosen so the size is preserved, full is p equals m minus one and makes the output bigger. Full is a signal-processing thing; you will essentially never use it in a CNN.

Question to put to the room: 'a 5 by 5 kernel, stride 1 — what padding gives same?' Answer 2, from the formula, and in general m minus one over two for odd m. Follow up: 'what about an even kernel?' There is no symmetric answer, which is one reason kernels are almost always odd-sized.

Expect: 'why 3 by 3 rather than something bigger?' Two stacked 3 by 3 layers see a 5 by 5 region with fewer parameters and an extra nonlinearity in between. That is the VGG argument and it is why 3 by 3 won.


## Slide 5
A short slide that converts the abstraction into something they can see.

Point at each panel. The vertical Sobel lights up the left and right edges of the pads and of the component body, and is nearly blind to the horizontal ones. The horizontal Sobel does the opposite — notice that it is the one that picks out the bridge band, because the bridge is a horizontal structure in this orientation. The Laplacian outlines everything. The box blur smears the lot.

Now the arithmetic printed in the notebook: the first three kernels sum to zero, the box blur sums to one. That is the same property we designed into the 1-D kernel two slides ago. A zero-sum kernel measures difference, so doubling the lamp output doubles its response but never turns a non-edge into an edge; a kernel that sums to one measures brightness, which is the nuisance variable we are trying to ignore.

Flag the cross-correlation trap in the code box once more, because this is where it actually bites: F.conv2d does not flip your kernel, so a hand-built asymmetric kernel comes out sign-flipped relative to the textbook convolution. If you want the true convolution, call .flip(-1, -2) first.

Question to the room: 'the network has sixteen kernels in its first layer. What would you expect them to look like after training?' Take guesses and tell them we will look at the real answer on slide 13.

Expect: 'couldn't we just hand-design all the kernels?' People did, for thirty years — SIFT, HOG, Gabor banks. It works until the defect you care about is not the one your features describe. Learning them is what changed in 2012.


## Slide 6
Two things pooling gives you, and the second one is the one students under-rate.

The obvious one first: halving both spatial dimensions quarters the work of every layer above, and it is free because pooling has no parameters at all — no weights, no biases, nothing to train.

The important one is local translation invariance. Work the left panel by hand. Take the top-left 2 by 2 window, values 2, 3, 5, 7; max-pooling reports 7. Now imagine the part shifts and the 3 becomes a 4. The answer is still 7. The output only changes if the shift moves the winner out of its window.

Read the right panel out loud, because the number is the memorable part. Shift the entire image by one pixel: essentially none of the raw pixels are unchanged — 0.03 percent — but after a 4 by 4 max-pool, 53.6 percent of the values are bit-identical. The notebook also runs 2 by 2 and 8 by 8, which give 31.5 and 68.2 percent. Bigger windows, more invariance, less spatial precision. That is the trade-off, stated in numbers.

Question to the room: 'when would you NOT want translation invariance?' Good answers: measuring whether a component is offset from its pad, where the position IS the defect; any task where you need to report where something is, not just whether it is there. Segmentation and detection architectures handle this by keeping a high-resolution path.

Expect: 'my architecture has no pooling at all.' Correct — many modern nets use stride-2 convolutions instead, which are pooling with learnable weights. Springenberg's All Convolutional Net is the reference, and it is in the notebook.


## Slide 7
The slide that answers 'why not just use a bigger dense network?'

First the mechanics, left panel. Our images have three input channels, and the first layer produces sixteen output feature maps. So the layer holds sixteen kernels, each of which is 3 by 3 by 3 — it spans all three colour channels. Each kernel is convolved with the input, the three channel results are summed, one bias is added, and that produces exactly one output map. Sixteen kernels, sixteen maps. The kernel tensor is therefore rank four: height by width by input channels by output channels.

Note the colour point, because it is real here: gold pad, silver solder and green resist differ mostly in colour, not in brightness. A grayscale camera would make this job much harder, and the network will use those three channels.

Now the right panel, and read the numbers slowly. That conv layer holds 448 parameters — 3 times 3 times 3 times 16, plus 16 biases. A dense layer taking the same 64 by 64 by 3 input and producing the same 64 by 64 by 16 output would hold 805 million. That is a factor of about 1.8 million.

But make the second point too, because it matters more than the first: the saving is not the main prize. The conv layer encodes an assumption — that a feature means the same thing wherever it appears — which happens to be true of solder joints. A dense layer has to learn that separately for every pixel position, from labelled data you do not have. Slide 12 measures exactly what that costs.

Question: 'so is a conv layer just a dense layer with most weights set to zero and the rest tied together?' Yes, precisely — and that is a useful way to see it. It is a dense layer with a very strong prior baked in.


## Slide 8
The build slide. Most of them have written something like this; spend the time on the two design choices that are not obvious.

First, the shape discipline. Same padding on every convolution means the conv layers never change the spatial size, so all the downsampling happens at the pooling layers and you can read the shapes straight off the strip: 64, 32, 16, 8. Channels double as resolution halves, which is the standard shape of a CNN — you trade spatial detail for feature richness as you go up. Tell them to always run a dummy batch of ones through the model and print the shape after each layer before training anything; the notebook does it, and most CNN bugs are shape bugs.

Second, and this is the interesting one: global average pooling instead of flatten. Flattening 64 by 8 by 8 gives 4096 numbers, and a dense layer on top would hold about 16 000 parameters — a quarter of the whole network, every one of them position-specific, which is precisely the inductive bias we spent six slides avoiding. Global average pooling asks 'how strongly does this feature appear anywhere in the frame', costs zero parameters, and works for any input size. Exercise 5 has them swap it back and measure what happens.

Third, the last line: no Softmax. CrossEntropyLoss applies log-softmax internally. Adding a Softmax layer applies it twice, the gradients squash, training crawls, and nothing ever raises an error. It is the single most common silent bug in this chapter.

Question: 'why 16 channels first and not 64?' Cost. The first layer runs at full 64 by 64 resolution, so it is the most expensive place to put channels. Cheap where it is big, rich where it is small.


## Slide 9
A reference slide. Do not read the table aloud — point at it, make the two arguments, move on.

Argument one: the task chooses the loss, not taste. One output unit and a yes/no question means binary cross-entropy. One unit per class and mutually exclusive classes means categorical cross-entropy. Our four defect classes are mutually exclusive — a joint is not simultaneously bridged and insufficient — so CrossEntropyLoss is right. If a joint could carry several independent defect flags at once you would use four independent sigmoids and BCEWithLogitsLoss instead, and that is worth saying because multi-label inspection is common.

Argument two: prefer logits. The fused versions compute log-sum-exp in a numerically stable way. The split versions overflow once the model becomes confident, which is exactly when training is going well. The notebook shows the two forms agreeing to four decimals on a small example — that agreement is the point, because it means there is no reason to use the fragile one.

Then the code box, which is the slide everyone photographs. Applying Softmax before CrossEntropyLoss squashes the gradients and slows training to a crawl without ever raising an error, so it is invisible unless you know to look. Also note the target format: CrossEntropyLoss wants integer class indices, not one-hot vectors.

Question to the room: 'your model prints 25 percent accuracy on four classes and the loss sits at 1.386. What do you check first?' Answer: ln 4 is 1.386, so the model is outputting a uniform distribution and has learned nothing — check the loss wiring and the learning rate before touching the architecture.


## Slide 10
Two mechanisms and one honest limitation.

L2 first, in one sentence: add lambda times the sum of squared weights to the loss and the optimiser shrinks the weights toward zero. In PyTorch you pass weight_decay to the optimizer rather than building the penalty by hand; for SGD the two are provably equivalent, and for Adam they are subtly different, which is why AdamW exists.

Dropout, left panel. During training each unit is zeroed with probability p, so a different sub-network is trained on every mini-batch — with h hidden units that is 2 to the h possible sub-networks, sharing one set of weights. At inference all units are active. The reading worth remembering is the ensemble one: you are training a huge ensemble for the price of a single model, and averaging over it at test time.

Then the practical trap, and say it firmly: PyTorch uses inverse dropout, scaling the surviving activations by one over one minus p during training so that inference needs no scaling. That only works if the module knows which mode it is in. model.train() before the loop, model.eval() before evaluating. Forget it and your validation numbers are quietly wrong — and they will look plausible, which is what makes it dangerous.

Now the right panel, and be honest about it. Dropout plus L2 moves us from 84.2 to 89.4 percent on production data. That is a real five points and worth having. But look at the red line: all three of these models score 100 percent on validation. Regularisation is not the thing standing between us and a working inspector.

Question to the room, and let it hang: 'if the model is at 100 percent on validation and 84 on the line, is that overfitting?' The honest answer is no — not in the usual sense. It is a distribution problem, and that is the next slide.


## Slide 11
The payoff of the domain-gap story, and the most practically useful slide in the deck.

Top row: one labelled image, five random augmentations of it. Bottom row: five genuine production frames. Let them look for a moment, then make the point — the augmented images are starting to look like the bottom row. That is the entire mechanism. We could not collect production labels, so we synthesised the variation instead.

Now the design rule, which is the transferable part. A transformation is legitimate only if it reflects a symmetry the real process has. Flips: the footprint is symmetric and boards arrive either way round, so a flipped bridge is still a bridge. Rotation and translation: placement tolerance and board registration. Brightness: the lamp ages. Every one of those is a fact about the factory, not a trick.

And say what would be wrong, because that is what they will get wrong. A 90-degree rotation on a polarised component, where orientation is the defect you are looking for. A strong hue shift here, which would merge gold pad and silver solder — and 'bare gold visible' is exactly the cue that defines insufficient solder. Exercise 6 has them do that on purpose and watch which class collapses.

Then the rule with no exceptions: augment the training set only. Never the validation set, never the test set. The test set is supposed to be a picture of production; perturbing it makes your headline number meaningless.

Read the callout: 14.5 points, against 5.1 for dropout and L2. Ask why the difference, and steer them to the answer — dropout treats a variance problem, augmentation treats a coverage problem, and what we have is a coverage problem.

Expect: 'shouldn't we just collect better data?' Yes, if you can. Augmentation is what you do while you wait for it.


## Slide 12
The slide the whole lecture has been walking toward. Three lessons, and they do not have the same fix — make sure all three land separately.

Lesson one, the architecture. The dense MLP has 3.18 million parameters, fifty-two times the CNN's 60 772, and it gets 37.9 percent against the plain CNN's 84.2. It is not short of capacity — it fits the labelled set to 99 percent. It is short of the right inductive bias. Parameter sharing and pooling let the CNN recognise a bridge at a position it never saw in training; the MLP has to learn every position separately from 480 images, and it cannot.

Lesson two, and this is the one to make them uncomfortable. Look at the grey bars: 95.8, 100, 100, 100. Every single model passes validation. If you had shipped on the strength of that number you would have shipped the 37.9 percent model. The validation split came from the same afternoon as the training data, so it measures whether optimisation worked, not whether the model generalises. This is not a subtle statistical point — it is the most common way real projects fail.

Lesson three, the ladder. Regularisation adds 5.1 points, augmentation adds 14.5. Different problems, different fixes. Diagnosing which one you have is the difference between a fix and a month of tuning.

Left panel: the confusion matrix of the best model, 98.8 percent. Point out that the remaining errors are almost all excess predicted as insufficient — eight of 200 — which is the cosmetic pair. One bridge in 200 is missed, and that is the expensive one.

Question to close on: 'a missed bridge ships a shorted board; a false alarm costs thirty seconds at the rework bench. Is argmax the right decision rule?' No — exercise 8 has them build the cost-weighted rule and measure what accuracy they give up for it.


## Slide 13
The satisfying slide. Slow down — this is the one that makes the abstraction feel real.

Top row: the actual learned kernels of the first convolution layer, each 3 by 3 by 3, plotted with the three input channels as red, green and blue. They are not random. Several are clearly oriented — bright on one side, dark on the other, which is a Sobel-like edge detector discovered from data. Others are colour-opponent: gold against green, or silver against dark. That is the network figuring out for itself that pad, solder and resist differ mainly in hue, which is the observation we made by hand on slide 7.

Bottom row: what those kernels produce for a genuine production frame — a bridge, rotated and off-centre. Different maps light up on different structures: the outline of the body, the solder-to-board boundary, the bridge band itself. Compare them with the hand-built Sobel outputs on slide 5 and the resemblance is obvious. The network rediscovered the classical feature detectors without being told they exist.

Then state the hierarchy claim properly, because this is the evidence for it: layer one finds edges and colour contrast; layer two combines those into corners and bars; by layer four, at 8 by 8 resolution and 64 channels, individual units are responding to things like 'fillet present' and 'gap bridged'. Nobody designed any of it — the only supervision was 480 class labels.

Question to the room: 'if the first layer of every image network learns roughly these same detectors, what does that suggest?' Steer them to transfer learning — take a backbone pretrained on millions of natural images and you start with these for free. Section 13 of the notebook has the recipe and the three caveats that matter on a factory floor.

Expect: 'can we visualise the later layers this way?' Not directly — a layer-three kernel is 3 by 3 by 64 and has no natural picture. That is what activation maximisation and the attribution literature are for.


## Slide 14
Close by walking the strip once, left to right, naming the step and the thing that goes wrong at it: images as tensors — channel order and dtype; conv, ReLU and pool — same padding, and shapes you did not check; the head — a wide dense layer where global average pooling belonged; the loss — softmax applied twice; regularisation — the wrong one for the problem you actually have; evaluation — a test set that does not look like production.

Then pick three of the seven rules rather than reading all of them. Rule 2 saves them the most time this week. Rule 6 is the one that makes them better engineers, because underfitting and a domain gap look identical from the validation set and call for opposite fixes. Rule 7 is the one that makes their results trustworthy, and it is the lesson of this entire deck: every model we trained passed validation, and three of the four would have failed on the line.

Point them at the notebook. Seven minutes on a laptop CPU, no downloads, every number on these slides printed in it. Exercises 1 and 3 build the mechanical intuition — convolution by hand, and receptive-field arithmetic verified empirically. Exercise 7 is the one to argue about in the next session: how many production-like labels are worth as much as augmentation? That is the real trade-off an inspection engineer faces, labelling time against modelling effort.

Preview what comes next: everything here is a fixed-size image in, one label out. Sequence models relax the fixed-size part, and detection and segmentation relax the one-label part — but the convolution, the padding arithmetic and the feature hierarchy all carry over unchanged.

Final question to leave them with: 'your model is at 99 percent on the test set. What is the first thing you should check?' That the test set is not secretly a copy of the training set.


