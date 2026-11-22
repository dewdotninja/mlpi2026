# Chapter 10. CNNs for Visual Inspection

*Part III. Deep learning for robot signals and images*

> **Case study:** automated optical inspection (AOI) of SMT solder joints
> **Companion notebook:** [`lectures/lect09/supplement/mlpi26_Lect09_nbsup1.ipynb`](../../lectures/lect09/supplement/mlpi26_Lect09_nbsup1.ipynb)
> **Lab:** [`labs/lab09/mlpi26_Lect09_Lab.ipynb`](../../labs/lab09/mlpi26_Lect09_Lab.ipynb) (automated visual inspection with CNNs)
> **Built from:** Lecture 9 deck, Lecture 9 Supplement 1 and its speaker notes

The dense networks of Chapters 8 and 9 connect every input to every unit. For a 64 × 64 colour image that is 12 288 inputs, and a dense first layer prices itself out immediately. More importantly, it throws away the two things we know about images: nearby pixels belong together, and a defect looks the same wherever it appears. A **convolutional layer** builds both facts into the architecture. This chapter shows how, measures what it buys, and then shows something more important: a model that scores 100% on validation can still fail on the production line, and the fix for that is not the one most engineers reach for first.

## Learning objectives

After this chapter you should be able to:

1. Compute 1-D and 2-D convolutions by hand, explain the difference between convolution and cross-correlation, and predict output sizes from kernel, padding and stride.
2. Design a kernel as a matched filter, and explain why zero-sum kernels ignore illumination.
3. Explain what pooling buys (local translation invariance, at no parameter cost) and measure it.
4. Count the parameters of a convolutional layer and explain why parameter sharing matters more than the parameter saving.
5. Build a CNN in `torch.nn`, check its shapes, and justify global average pooling over flattening.
6. Choose between dropout and L2 regularisation on the one hand and data augmentation on the other, by diagnosing whether the problem is variance or coverage.
7. Hold out a test set that looks like deployment, and read a confusion matrix in terms of the cost of each error.
8. Set up transfer learning from a pretrained backbone and state its caveats on a factory floor.

---

## 10.1 The engineering problem: an AOI station on an SMT line

An AOI station sits after the reflow oven and photographs every solder joint on every board. It is the most widely deployed computer-vision task in electronics manufacturing. The case study classifies 64 × 64 RGB crops centred on a 0603 chip-component footprint into four classes taken from IPC-A-610, the standard that defines an acceptable solder joint:

| Class | What it looks like | Why it matters |
|---|---|---|
| `good` | A concave fillet climbing the metallised end cap; a bright specular *streak*, because a concave surface reflects the ring light as a line | Ships |
| `insufficient` | Less solder; bare gold pad visible; the glint is a small round spot | A weak joint that fails later under thermal cycling |
| `bridge` | A silver band across the gap between the pads | A short circuit; the board does not work |
| `excess` | A convex blob spilling past the pad edge; a round hot spot, because a convex surface reflects as a point | Mostly cosmetic, sometimes a short risk |

The images are rendered analytically from a geometric model of the footprint, with anti-aliased edges and clutter from neighbouring footprints, vias and silkscreen text. Nothing is downloaded, the notebook runs offline in a few minutes, and because we know the generating process we can make honest claims about what a model should have been able to learn.

### 10.1.1 A trap built into the data

The labelled images and the production images come from different **regimes**:

| Regime | Rotation | Shift | Scale | Lamp gain | Used for |
|---|---|---|---|---|---|
| Narrow (the labelling bench) | ±4° | ±2.5 px | 0.97–1.03 | 0.95–1.08 | 480 training + 240 validation images |
| Full (production) | ±15° | ±9 px | 0.88–1.12 | 0.70–1.30 | 800 test images |

The labelled set comes from one careful afternoon: parts near the centre, even lighting, a few degrees of rotation. Production has nine pixels of registration error, fifteen degrees of rotation, a lamp anywhere from 70% to 130% of nominal, and clutter intruding at the edges.

This is not a contrived difficulty. Labelled defect data always comes from a narrow slice, because somebody had to stand at a bench and confirm each label. The natural move is to hold out 20% of the labelled images as validation. Section 10.8 shows what that costs.

## 10.2 Convolution in one dimension: a kernel is a matched filter

Before any 2-D notation, look at one scan row across the footprint, averaged over 24 perfectly registered parts of each class. Every class has bright solder shoulders at about 0.73 intensity. Three classes show a dark gap at 0.14, the black ceramic body between the pads. The bridge shows a bright gap at **0.68**, because solder has wicked across it.

A detector writes itself: positive weights where we expect bright, negative where we expect dark. Make the weights sum to zero by subtracting their mean, and slide the kernel along the row:

$$
y[i] = \sum_{k} x[i+k]\,w[k].
$$

The response at the gap centre is +0.2599, +0.2601 and +0.2600 for `good`, `insufficient` and `excess`, and **+0.0090** for `bridge`. A single threshold separates them on data the filter was never fitted to, because it was not fitted at all; it was designed.

Two details in that construction carry through the whole chapter.

**Zero-sum kernels ignore brightness.** A kernel whose weights sum to zero gives exactly zero on any flat patch, whatever its intensity. It reports structure and ignores the lamp. That is why the first layer of a trained CNN almost always discovers near-zero-sum kernels.

**Convolution flips; PyTorch does not.** Mathematical convolution reverses the kernel, y[i] = Σₖ x[i − k] w[k]. Cross-correlation, the formula above, does not. PyTorch's `conv1d` and `conv2d` compute cross-correlation and call it convolution. When weights are learned this makes no difference (the network learns the flip). When you hand-build an asymmetric kernel, as here, it does; call `.flip(-1)` first if you want the textbook convolution.

If one designed filter solves one class, why train a network? Because the filter needs the part perfectly registered, someone had to know which row and which mechanism to look at, and it does nothing for the other three classes. A CNN learns hundreds of such filters and learns where to apply them.

## 10.3 Convolution in two dimensions

In two dimensions, one output pixel is the dot product of an m × m patch of the input with an m × m kernel. Slide the kernel one step and you get the next output pixel. Two properties make this the right operation for images:

- **Sparse connectivity.** An output pixel sees only nine inputs (for a 3 × 3 kernel), not all 4096.
- **Parameter sharing.** The same nine weights are used at every position.

The output size along each spatial dimension is

$$
o = \left\lfloor \frac{n + 2p - m}{s} \right\rfloor + 1,
$$

for input size n, padding p, kernel size m and stride s.

| Input n | Kernel m | Padding p | Stride s | Output o | Name |
|---|---|---|---|---|---|
| 64 | 3 | 1 | 1 | 64 | "Same" padding |
| 64 | 3 | 0 | 1 | 62 | "Valid" padding |
| 64 | 2 | 0 | 2 | 32 | A 2 × 2 pooling window |

For an odd kernel, same padding is p = (m − 1)/2. Even kernels have no symmetric answer, which is one reason kernels are almost always odd. With valid padding eight layers would lose 16 pixels to nothing, so CNNs use same padding on convolutions and downsample deliberately. Why 3 × 3 rather than larger? Two stacked 3 × 3 layers see a 5 × 5 region with fewer parameters (18 versus 25 per channel pair) and an extra non-linearity in between. That is the VGG argument, and it is why 3 × 3 won.

### 10.3.1 Nine numbers make a feature detector

Applied to a bridge image, classical kernels each pick out one kind of structure. The vertical Sobel kernel lights up the left and right edges of the pads and body. The horizontal Sobel picks out the bridge band, which is horizontal in this orientation. The Laplacian outlines everything. A 3 × 3 box blur smears it all. The first three sum to zero and measure *difference*; the blur sums to one and measures *brightness*, the nuisance variable we are trying to ignore.

People hand-designed such features for thirty years (SIFT, HOG, Gabor filter banks). It works until the defect you care about is not the one your features describe. Learning the kernels from labelled data is what changed in 2012 (Krizhevsky, Sutskever and Hinton).

## 10.4 Pooling: invariance for free

A max-pooling layer replaces each k × k window by its maximum; average pooling uses the mean. Pooling has **no parameters**. It does two things.

The obvious one: halving each spatial dimension quarters the work of every layer above.

The under-rated one: **local translation invariance**. Take a 2 × 2 window holding 2, 3, 5 and 7. Max-pooling reports 7. If the part shifts slightly and the 3 becomes a 4, the answer is still 7. The output changes only if the shift moves the winner out of its window. The notebook measures this on whole images shifted by one pixel:

| Representation | Values bit-identical after a 1-pixel shift |
|---|---|
| Raw pixels | 0.03% |
| After 2 × 2 max-pool | 31.5% |
| After 4 × 4 max-pool | 53.6% |
| After 8 × 8 max-pool | 68.2% |

Bigger windows give more invariance and less spatial precision. When you need to know *where* something is, as when measuring whether a component is offset from its pad, invariance is the wrong property; detection and segmentation architectures keep a high-resolution path for that reason. Many modern networks also replace pooling with stride-2 convolutions, which are pooling with learnable weights (Springenberg et al., 2015).

## 10.5 Channels and the parameter argument

Colour images have three input channels, and a convolutional layer produces several output **feature maps**. A layer mapping 3 channels to 16 holds 16 kernels, each 3 × 3 × 3, spanning all input channels. Each kernel is applied to the input, the three channel results are summed, a bias is added, and the result is one output map. The weight tensor is rank four: output channels × input channels × height × width.

Colour is not decorative here. Gold pad, silver solder and green solder resist differ mainly in hue, not brightness. A greyscale camera would make the job much harder.

That 3 → 16 layer holds 3 × 3 × 3 × 16 + 16 = **448 parameters**. A dense layer mapping the same 64 × 64 × 3 input to the same 64 × 64 × 16 output would hold **805 371 904**, about 1.8 million times as many.

The saving is not the main prize. The convolutional layer encodes an assumption, that a feature means the same thing wherever it appears, which happens to be true of solder joints. A dense layer must learn that separately for every pixel position, from labelled data you do not have. One useful way to see a convolutional layer is as a dense layer with most weights fixed at zero and the rest tied together: a dense layer with a very strong prior. Section 10.8 measures what that prior is worth.

## 10.6 Building the CNN

```python
cnn = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),  nn.ReLU(), nn.MaxPool2d(2),   # 64 -> 32
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32 -> 16
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16 -> 8
    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(),                         # global average pooling
    nn.Dropout(0.3),
    nn.Linear(64, 4),                                              # raw logits: no Softmax
)                                                                  # 60 772 parameters
```

Three design choices deserve comment.

**Shape discipline.** Same padding means convolutions never change spatial size, so all downsampling happens at the pooling layers and you can read the shapes straight off: 64, 32, 16, 8. Channels double as resolution halves, the standard shape of a CNN, trading spatial detail for feature richness. The first layer runs at full resolution, so it is the most expensive place for channels: cheap where the maps are big, rich where they are small. Before training anything, push a dummy batch through and print the shape after every layer. Most CNN bugs are shape bugs.

**Global average pooling instead of flattening.** Flattening the final 64 × 8 × 8 maps gives 4096 numbers, and a dense layer on top would hold about 16 000 parameters, a quarter of the network, every one of them position-specific, which is exactly the inductive bias we have spent this chapter avoiding. Global average pooling (Lin, Chen and Yan, 2014) asks "how strongly does this feature appear anywhere in the frame?", costs no parameters, and works for any input size. Exercise 5 swaps it back and measures the damage.

**No softmax.** `nn.CrossEntropyLoss` applies log-softmax internally and expects integer class indices as targets, not one-hot vectors. Adding a softmax applies it twice, squashes the gradients, and slows training to a crawl without raising an error (Section 9.5). A related diagnostic: on four balanced classes, a loss stuck at ln 4 = 1.386 with 25% accuracy means the model outputs a uniform distribution and has learned nothing. Check the loss wiring and the learning rate before touching the architecture.

If a joint could carry several independent defect flags at once, the classes would not be mutually exclusive and you would use four independent logits with `BCEWithLogitsLoss` instead. Multi-label inspection is common, so ask the question explicitly.

For comparison, the notebook also trains a dense MLP (`Flatten → 12288 → 256 → 128 → 4`) with **3 179 396 parameters**, 52 times the CNN. All models use Adam at 2 × 10⁻³ with cosine annealing, batch size 32, 90 epochs, and cross-entropy on logits.

## 10.7 Regularisation and augmentation

### 10.7.1 L2 and dropout

L2 regularisation adds λ‖w‖² to the loss and shrinks weights toward zero; in PyTorch pass `weight_decay` to the optimiser, and prefer `AdamW` with adaptive optimisers. Dropout zeroes each unit with probability p during training, so every mini-batch trains a different sub-network. With h hidden units that is 2ʰ possible sub-networks sharing one set of weights, and inference averages over them: a large ensemble for the price of one model (Srivastava et al., 2014). PyTorch uses *inverted* dropout, scaling surviving activations by 1/(1 − p) during training so that inference needs no correction. That works only if the module knows its mode, so `model.train()` before the training loop and `model.eval()` before evaluation are not optional.

### 10.7.2 Data augmentation

Augmentation applies random, label-preserving transformations to training images each time they are drawn. The notebook implements it in pure PyTorch with `F.affine_grid` and `F.grid_sample` (random rotation, translation and scale) plus a random brightness gain; the equivalent `torchvision.transforms.v2` pipeline is a few lines of `RandomAffine` and `ColorJitter`.

The design rule is the transferable part: **a transformation is legitimate only if it reflects a symmetry the real process has.**

- Flips: the footprint is symmetric and boards arrive either way round, so a flipped bridge is still a bridge.
- Rotation and translation: placement tolerance and board registration.
- Brightness: the lamp ages.

Each of those is a fact about the factory, not a trick. Now the ones that would be wrong. A 90° rotation on a polarised component, where orientation *is* the defect. A strong hue shift here, which would merge gold pad and silver solder; "bare gold visible" is exactly the cue that defines `insufficient` (Exercise 6).

And one rule with no exceptions: **augment the training set only.** The test set is supposed to be a picture of production. Perturbing it makes the headline number meaningless.

## 10.8 Validation passed everything. The line did not.

Four models, the same 480 training images, the same 90 epochs:

| Model | Parameters | Train | Validation (narrow) | **Test (production)** |
|---|---|---|---|---|
| Dense MLP | 3 179 396 | 0.992 | 0.958 | **0.379** |
| CNN, no regularisation | 60 772 | 1.000 | 1.000 | 0.843 |
| CNN + dropout(0.3) + L2(10⁻⁴) | 60 772 | 0.998 | 1.000 | 0.894 |
| CNN + dropout + L2 + augmentation | 60 772 | 0.994 | 1.000 | **0.988** |

Three lessons, and they have three different fixes.

**Lesson one: architecture.** The MLP has 52 times the CNN's parameters and scores 37.9% on production against the plain CNN's 84.3%, a gap of **46.4 points**. It is not short of capacity; it fits the labelled set to 99%. It is short of the right inductive bias. Parameter sharing and pooling let the CNN recognise a bridge at a position it never saw in training. The MLP must learn every position separately from 480 images, and it cannot.

**Lesson two: validation lied.** Every model passes validation: 95.8%, 100%, 100%, 100%. Had you shipped on that number, you could have shipped the 37.9% model. The validation split came from the same afternoon as the training data, so it measured whether optimisation worked, not whether the model generalises to production. Validation accuracy saturated at 100% for all three CNNs, so it could not even select a checkpoint; the notebook uses the final epoch of every run for that reason. This is not a subtle statistical point. It is the most common way real projects fail.

**Lesson three: diagnose before you regularise.** Dropout and L2 add **5.1 points**. Augmentation adds **14.5 points**, nearly three times as much. Dropout treats a *variance* problem: too many parameters for the data. Augmentation treats a *coverage* problem: training data that does not span production. What we have is a coverage problem. If the model is at 100% on validation and 84% on the line, that is not overfitting in the usual sense. It is a distribution gap, and underfitting, overfitting and a distribution gap call for different fixes.

Should you just collect better data? Yes, if you can. Augmentation is what you do while you wait. Exercise 7 measures the exchange rate: how many production-regime labelled images are worth as much as augmentation.

### 10.8.1 Read the confusion matrix

The best model's 800 production images:

| True class | Correct | Main error |
|---|---|---|
| `good` | 199 / 200 | |
| `insufficient` | 200 / 200 | |
| `bridge` | 199 / 200 | One missed |
| `excess` | 192 / 200 | 8 predicted as `insufficient` |

The remaining errors are almost all `excess` predicted as `insufficient`, the cosmetic pair. One bridge in 200 is missed, and that is the expensive one: a missed bridge ships a shorted board, while a false alarm costs thirty seconds at the rework bench. Is `argmax` the right decision rule when errors cost so differently? No. Choose the class that minimises expected cost under the predicted probabilities (Exercise 8), the multi-class version of the threshold choice in Section 7.7.2. Note also that softmax probabilities from a network are often overconfident; Chapter 16 returns to calibration.

## 10.9 What the network learned

Nobody specified the first-layer kernels, yet they are not random. Plotted with their three input channels as red, green and blue, several are clearly oriented, bright on one side and dark on the other: Sobel-like edge detectors discovered from data. Others are colour-opponent, gold against green or silver against dark. The network found for itself that pad, solder and resist differ mainly in hue.

Applied to a rotated, off-centre production bridge, different feature maps respond to different structures: the outline of the component body, the boundary between solder and board, and the bridge band itself. The resemblance to the hand-built Sobel outputs of Section 10.3.1 is obvious.

This is the evidence for the feature-hierarchy picture of CNNs. Layer one finds edges and colour contrast. Layer two combines them into corners and bars. By layer four, at 8 × 8 resolution and 64 channels, units respond to things like "fillet present" and "gap bridged". The only supervision was 480 class labels. Later layers cannot be visualised this way, because a layer-three kernel is 3 × 3 × 64 and has no natural picture; activation maximisation and attribution methods such as Grad-CAM (Chapter 17) address that.

## 10.10 Transfer learning

If the first layer of every image network learns roughly the same detectors, you should not have to learn them again. On a real inspection project with a few hundred labelled images, you would almost certainly start from a **backbone pretrained** on a large natural-image corpus:

```python
from torchvision.models import resnet18, ResNet18_Weights

backbone = resnet18(weights=ResNet18_Weights.DEFAULT)    # downloads weights
for p in backbone.parameters():                           # 1. freeze the features
    p.requires_grad = False
backbone.fc = nn.Linear(backbone.fc.in_features, 4)       # 2. new, trainable head
# 3. train the head; then unfreeze the last block and fine-tune at ~10x lower learning rate
```

The recipe has two stages. First train only the new head on top of frozen features; this is fast and hard to overfit. Then unfreeze the last block or two and fine-tune with a learning rate about ten times smaller, so that the pretrained weights move gently.

Three caveats matter on a factory floor.

1. **The input contract.** The pretrained normalisation statistics and expected input size are part of the model. Feed it what it expects; `ResNet18_Weights.DEFAULT.transforms()` returns them.
2. **Domain shift.** A frozen backbone was trained on images lit by the sun, not a ring light, and on objects, not solder fillets. Fine-tuning the last block usually pays.
3. **Size and latency.** ResNet-18 has about 11 million parameters, more than 180 times the network built here. On a line inspecting one board a second, with dozens of joints per board, that is a real cost. Smaller backbones (MobileNet, EfficientNet) and the quantisation of Chapter 16 exist for this reason.

Self-supervised backbones such as DINOv2 and vision transformers (Chapter 12) extend the same idea: features learned from unlabelled images transfer well to inspection tasks with few labels. None of this changes the mechanics of this chapter; you simply start from better initial kernels.

A real AOI station often needs more than a class label. It needs a bounding box around the defect, or a pixel mask of the solder region. Detection and segmentation relax the "one image, one label" assumption of this chapter while keeping its convolution, padding arithmetic and feature hierarchy unchanged.

## 10.11 The decisive experiment

The decisive experiment for a vision model is to **test it on images that look like deployment, not like the labelling bench.** Here that meant a test set drawn from the full production regime, and it exposed a 58-point gap for the MLP and a 16-point gap for the plain CNN that validation hid completely. On a real line the equivalent is a held-out set from a different shift, a different lamp, or a different board revision, labelled once and kept aside. If your model scores 99% on the test set, the first thing to check is that the test set is not secretly a copy of the training distribution.

## Summary

- A convolutional kernel is a matched filter. Zero-sum kernels respond to structure and ignore illumination. PyTorch computes cross-correlation.
- Output size is ⌊(n + 2p − m)/s⌋ + 1. Use same padding on convolutions and downsample deliberately.
- Pooling has no parameters and buys local translation invariance: after a 4 × 4 max-pool, 53.6% of values survive a 1-pixel shift unchanged, against 0.03% of raw pixels.
- Parameter sharing is a prior, not just a saving. A 60 772-parameter CNN beat a 3.2-million-parameter MLP by 46 points on production images.
- Use global average pooling instead of a wide dense head, and never put a softmax before `CrossEntropyLoss`.
- Dropout and L2 fix variance; augmentation fixes coverage. Here augmentation added 14.5 points, nearly three times the regularisation gain. Augment only with transformations the process really has, and only the training set.
- A validation split drawn from the same batch as training cannot tell you whether a model will work. Hold out data that looks like deployment.
- Read the confusion matrix and weigh errors by their cost. Start real projects from a pretrained backbone and fine-tune.

## Exercises

1. **Convolution by hand.** Take x = [3, 2, 1, 7, 1, 2, 5, 4] and w = [½, ¾, 1, ¼]. Compute y = x ∗ w with p = 0 and s = 2 on paper, then check with `conv1d`. Repeat without flipping the filter and say which of the two `F.conv1d` computes.
2. **Design a kernel.** Build a 1-D matched filter that fires on `insufficient` rather than `bridge`; the cue is bare gold visible between the solder and the component end cap. Report the separation you achieve and explain what makes your kernel robust to lamp brightness.
3. **Receptive field.** The network is conv3 → pool2 → conv3 → pool2 → conv3 → pool2 → conv3. How many input pixels does one unit of the last feature map depend on? Verify empirically by perturbing a single input pixel and finding which outputs change.
4. **Pooling, pushed.** Rerun the shift-invariance measurement for shifts of 1 to 8 pixels and pool sizes 2, 4 and 8, and plot the surface. At what shift does 4 × 4 pooling stop helping, and how does that relate to the pool size?
5. **Replace the head.** Swap global average pooling for `nn.Flatten()` plus `nn.Linear`. Report the parameter count and production accuracy. Did the extra parameters buy anything? Explain the sign of the change.
6. **Augmentation that hurts.** Add a random 180° rotation to the augmentation and retrain; is accuracy unchanged? Then add a strong random hue shift, which is not label-preserving here, and explain the result in terms of the class you destroyed.
7. **Close the gap the other way.** Instead of augmenting, draw 120 of the 480 training images from the full regime and retrain without augmentation. How many production-like images are worth as much as augmentation?
8. **Cost-sensitive decisions.** A missed bridge costs 200 times a false alarm. Replace `argmax` with the class that minimises expected cost under the softmax probabilities and report the new confusion matrix. How much accuracy did you give up, and was it worth it?
9. **Transfer learning.** If you have network access, replace the CNN with a frozen ResNet-18 and a new head, then fine-tune its last block. Upsample the 64 × 64 crops as the backbone expects. Compare production accuracy, training time and single-image latency with the 60 772-parameter CNN.

## Further reading

- Y. LeCun, L. Bottou, Y. Bengio and P. Haffner, "Gradient-based learning applied to document recognition," *Proceedings of the IEEE* 86(11) (1998) 2278–2324.
- A. Krizhevsky, I. Sutskever and G. Hinton, "ImageNet classification with deep convolutional neural networks," *NeurIPS* 2012.
- K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," *ICLR* 2015.
- K. He, X. Zhang, S. Ren and J. Sun, "Deep residual learning for image recognition," *CVPR* 2016.
- M. Lin, Q. Chen and S. Yan, "Network in network," *ICLR* 2014. The origin of global average pooling.
- J. T. Springenberg, A. Dosovitskiy, T. Brox and M. Riedmiller, "Striving for simplicity: the all convolutional net," *ICLR Workshop* 2015.
- V. Dumoulin and F. Visin, "A guide to convolution arithmetic for deep learning," arXiv:1603.07285, 2016.
- N. Srivastava et al., "Dropout: a simple way to prevent neural networks from overfitting," *JMLR* 15 (2014) 1929–1958.
- C. Shorten and T. M. Khoshgoftaar, "A survey on image data augmentation for deep learning," *Journal of Big Data* 6 (2019) 60.
- J. Yosinski, J. Clune, Y. Bengio and H. Lipson, "How transferable are features in deep neural networks?" *NeurIPS* 2014.
- IPC-A-610H, *Acceptability of Electronic Assemblies*, 2020.
