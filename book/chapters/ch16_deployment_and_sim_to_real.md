# Chapter 16. Deployment and Sim-to-Real

*Part V. From notebook to factory*

> **Case study:** a surface-defect classifier trained on a simulator and deployed to a factory camera
> **Lab:** [`labs/lab13/mlpi26_Lect13_Lab.ipynb`](../../labs/lab13/mlpi26_Lect13_Lab.ipynb) (the whole chapter, measured; about three minutes on a CPU)
> **Built from:** Lecture 13 deck, Lab 13, and the domain-randomisation experiment of Lecture 12 Supplement 1

Every model in this book has been judged on a held-out test set drawn from the same pile of data as its training set. That is correct practice, and Chapter 7 spent a whole chapter on doing it honestly. It is also the only thing we have ever measured. A deployed model meets data produced by a *different physical process*: another camera, another lamp, a lens, a machine that has worn since the data was collected. Nothing in a random train/test split has ever tested that.

This chapter builds a defect classifier that is **perfect in simulation and useless on the factory floor**, and then works out what to do about it. It shows that the obvious fix (more data) does nothing, that domain randomisation helps exactly as far as it reaches, and that the model cannot tell you it is broken while a one-line input statistic can. It then ships the model: export, latency against a real cycle-time budget, quantisation, and a ROS 2 node with the lessons built in.

## Learning objectives

After this chapter you should be able to:

1. Measure a sim-to-real gap, and show whether it is a data-quantity problem or a distribution problem.
2. Implement domain randomisation and quantify how much of the gap it closes.
3. Explain why softmax confidence cannot detect domain shift, and what uncertainty methods can and cannot add.
4. Build a label-free input monitor and decide what to track in production.
5. Export a PyTorch model with `torch.export`, measure latency at the unit the machine cares about, and decide whether to optimise at all.
6. Design a deployment that gates, logs, falls back and escalates to a human.

---

## 16.1 The last mile

| | Development | Deployment |
|---|---|---|
| Where the data comes from | One pool, split at random | A different physical process |
| What the test set is | A fair sample of the training distribution | Not available until it is too late |
| What we have measured | Generalisation to new samples | Nothing |

Chapter 7 showed how grouped and date-based splits can expose a shift before deployment does. Three kinds of shift matter on a factory floor:

- **Covariate shift.** The inputs change and the labelling rule does not: a new lamp, a dirty lens, a camera remounted a few degrees off.
- **Label (prior) shift.** The class balance changes: a new supplier sends more pitted stock.
- **Concept drift.** The relationship itself changes: a tighter tolerance makes yesterday's "ok" today's "scratch".

This chapter's case study is covariate shift, the most common and the easiest to underestimate.

## 16.2 Case study: perfect in simulation, useless on the floor

### 16.2.1 Two domains

The simulator renders 32 × 32 patches of a machined surface under perfect, even lighting with a sharp, perfectly square camera. There are four classes: **ok**, **scratch**, **pit** and **crack**. Simulated data is free, perfectly labelled and unlimited, which is exactly why people train on it.

The factory camera differs in five ways:

| Anticipated | Never thought of |
|---|---|
| Uneven lighting | Defocus blur |
| Gain and black-level offset | A few degrees of mount angle |
| Sensor noise | |

Hold on to that split. Which corruptions you anticipated decides the result of the whole chapter.

### 16.2.2 The gap

A small CNN of the kind you would put on an inspection station (14 340 parameters), trained on 2500 clean simulated images:

| Test set | Accuracy |
|---|---|
| Simulated | **1.000** |
| Real (factory camera) | **0.267** |
| Chance, four classes | 0.250 |

You would have reported a perfect validation score to your supervisor.

Accuracy alone does not say *how* it fails. On real images the model predicted "crack" for **584 of 600** parts: it collapsed onto one class. The lighting gradient and blur shifted every image far enough that one class's features dominate everywhere. Remember that a collapsed prediction distribution **is visible without a single label**; it comes back in Section 16.4.

### 16.2.3 The obvious fix that does not work

Simulated data is free, so collect four times as much:

| Training set | Simulated test | Real test |
|---|---|---|
| Clean simulation, 2500 images | 1.000 | 0.267 |
| Clean simulation, 10 000 images | 1.000 | **0.278** |

Four times the data bought nothing, because the problem was never a shortage of examples. Every one of those 10 000 images came from the same simulator, and the simulator is what is wrong. **More data from the wrong distribution is still the wrong distribution.** This is the most expensive mistake in applied machine learning: months of collection that could not possibly have helped.

## 16.3 Domain randomisation

If you cannot make the simulator match reality, make it *refuse to commit* to any one appearance. Give every training image a different lamp angle, gain, noise level, focus and mount angle. No single appearance is reliable any more, so the only stable signal left is the shape of the defect itself. If the real camera falls inside the range you trained over, reality becomes one more sample (Tobin et al., 2017).

Lab 13 compares two versions. **Narrow** randomisation varies lighting, gain and noise: the three anticipated corruptions. **Wide** adds the two we never thought of, blur and rotation.

| Training data | Real-test accuracy |
|---|---|
| Clean simulation | 0.267 |
| Narrow randomisation | 0.497 |
| **Wide randomisation** | **0.870** |

The gap closes in proportion to coverage. Narrow randomisation fixes the part of the shift it covers and leaves the rest; wide randomisation recovers most of the loss and restores a sensible prediction distribution. Note the cost: wide randomisation never reaches the simulated score. Tolerating blur and rotation costs accuracy on clean images, a trade almost always worth taking.

Chapter 14 found the same law in a control loop. A PPO quadrotor policy trained on the nominal simulator crashed in 100% of episodes with an 80 ms motor lag; randomising lag up to 80 ms brought that to 0%, at a cost of 1.7 points of return on the nominal quad. Outside the randomised range (100 ms of lag with a 1.45 × battery) it still crashed 26% of the time. **Randomisation helps exactly as far as it reaches.**

### 16.3.1 Before you believe that result

We built the factory camera ourselves, so we knew exactly which corruptions to add. A real project never hands you that.

| | |
|---|---|
| What we did | Randomised over blur and rotation because we knew the real camera had them |
| What you must do | Randomise wider than seems reasonable, over anything you cannot pin down |
| What never changes | Validate on real data before trusting any of it. Some of the gap will remain |

A simulator tuned to match your test set is just a training set with extra steps.

### 16.3.2 Other ways across the gap

Domain randomisation is the bluntest tool, and often the right first one. The others are worth knowing by name:

- **System identification.** Measure the real plant's parameters (motor time constant, camera gain curve) and set the simulator to match. It narrows the range you need to randomise.
- **Residual models.** Learn a correction from a small amount of real data and add it to the simulator. The Swift drone racer (Kaufmann et al., 2023) corrected its simulator with residual models of perception and dynamics recorded on the real track.
- **Fine-tuning on real data.** A few hundred labelled real images, used as in Chapter 10's transfer learning, often close the remaining gap.
- **History-conditioned policies.** Give a controller recent observations and actions so it can infer which plant it is driving (Chapter 14, Exercise 9).
- **Learned generators.** Train a generative model on a small set of real images and synthesise the rest. It has the same promise and the same trap: a generator trained on one camera reproduces that camera's quirks.

However you generate training data, the question is always the same: does its distribution cover the one you will meet?

## 16.4 Does the model know it is in trouble?

If confidence dropped when the model went out of its depth, deployment would be easy: gate on confidence and route the rest to a human. Test it with the model trained on clean simulation, on the real test set:

| | |
|---|---|
| Real predictions made with over 90% confidence | **79.5%** |
| Real predictions that are wrong | 73.3% |
| Mean confidence on the *wrong* predictions | **0.924** |

The model performs at chance and reports high confidence while doing so. It is confidently, catastrophically wrong. **A confidence threshold is not a safety mechanism against domain shift.**

### 16.4.1 Why confidence cannot see a domain shift

This is structural, not a quirk of this network. Softmax answers "which of my four classes fits best?". It never answers "does this input resemble anything I was trained on?". The four probabilities are forced to sum to one, so something always wins, however strange the input. Worse, push an input well outside the training data and one logit typically dominates, so confidence goes *up*, not down (Hein et al., 2019, prove this for ReLU networks). Any deployment that leans on a confidence threshold to catch domain shift is resting on nothing.

### 16.4.2 Monitor the input, not the model's opinion

The shift was obvious in the raw pixels. Build an accept band for a simple statistic, here per-patch contrast (standard deviation), from training data alone, and flag anything outside it:

| | Flagged by the contrast monitor |
|---|---|
| Real (factory) images | **99.8%** |
| In-domain images | 1.7% |

No labels, one number per image, computed in microseconds, and it works from day one.

| Approach | Detects the shift? | Needs labels? | Cost |
|---|---|---|---|
| Softmax confidence | **No**: 92% confident while wrong | No | One forward pass |
| Input statistics | **Yes**: flags nearly all of it | No | Microseconds |

Contrast is the right statistic *here* because we know the shift involves lighting and noise. A production monitor tracks several:

- **Input statistics**: per-image mean, contrast and sharpness, compared with their training distribution.
- **The prediction distribution**: the fraction of each class per hour. The collapse of Section 16.2.2 (584 of 600 "crack") shows up instantly.
- **The rate of the rarest class**, which is the first to vanish or explode.
- **Delayed ground truth** where it exists, such as downstream gauge results or customer returns, to catch concept drift that no input statistic will see.

For many features at once, use a two-sample test (Kolmogorov–Smirnov per feature, or a maximum mean discrepancy on embeddings) over a sliding window, with a threshold set so that the in-domain false-alarm rate is acceptable. Rabanser et al. (2019) compare these methods. The principle is general: **watch the data, not the model's opinion of itself.**

> **Sidebar: uncertainty quantification, and what it can and cannot do**
>
> Softmax confidence failed. Can a better uncertainty estimate do the job? Three methods are standard, and each is worth knowing for what it actually guarantees.
>
> - **Deep ensembles** (Lakshminarayanan et al., 2017). Train five networks from different random seeds and use their disagreement as uncertainty. Ensembles are the most reliable of the practical methods and degrade more gracefully under shift than a single network (Ovadia et al., 2019). They cost five times the training and inference.
> - **Monte Carlo dropout** (Gal and Ghahramani, 2016). Keep dropout switched on at test time, run several forward passes, and read the spread. It is nearly free if the network already uses dropout, but it usually underestimates uncertainty compared with an ensemble.
> - **Conformal prediction** (Vovk et al., 2005; Angelopoulos and Bates, 2023). Use a held-out calibration set to turn any model's scores into *prediction sets*, such as {scratch, crack}, that contain the true class with a guaranteed probability, say 95%. The guarantee is distribution-free, but it assumes the calibration data and the deployment data are **exchangeable**: drawn from the same distribution.
>
> Read the last assumption again. Under the very shift this chapter studies, conformal coverage is no longer guaranteed, and ensemble members trained on the same clean simulator can agree confidently on the same wrong answer, because they share the same blind spot. Uncertainty methods are valuable *in-distribution*: for deciding which parts to send to a human on a normal day, for calibrated scrap decisions, and for active learning. They complement an input monitor; they do not replace it. A useful combination on the line is an input gate for shift, then conformal sets for routine triage, with an ensemble's disagreement logged as one more monitored statistic.

## 16.5 Shipping it

### 16.5.1 Export

You will not ship a notebook. `torch.export` captures the model as a standalone program, so the robot needs no Python class definition:

```python
batch = torch.export.Dim("batch")
ep = torch.export.export(model, (torch.randn(2, 1, 32, 32),),
                         dynamic_shapes={"x": {0: batch}})
torch.export.save(ep, "defect_model.pt2")

loaded = torch.export.load("defect_model.pt2").module()
```

The 14 340-parameter CNN exports to a 110 KB program whose outputs match the eager model exactly (maximum difference 0.0).

**The trap that costs an afternoon:** export with an example batch of 1 and PyTorch specialises on a batch size of 1; the exported program then refuses every other batch size at runtime. Export with a representative batch and declare the batch dimension dynamic. (In general, check the exported program against the eager model on several batch sizes before you ship.)

### 16.5.2 Latency against a real budget

"0.3 ms" means nothing until you compare it with the machine's cycle time. The inspection station sees **120 parts per minute**, a 500 ms budget per part. The camera is 1280 × 1024, so one part is **1280 patches** of 32 × 32.

| | ms per patch | ms per part | Share of the 500 ms budget |
|---|---|---|---|
| One patch at a time | ≈ 0.34 | ≈ 440 | **≈ 90%** |
| Batched (32 patches) | ≈ 0.13 | ≈ 170 | ≈ 35% |

Per patch it looked fast. Per part it very nearly did not fit, leaving nothing for the camera, the conveyor or a bad day. Batching alone made it about 2.5 times faster, for free. Absolute timings vary by machine; the ratio is the stable part. **Always measure at the unit the machine actually cares about.** Report a tail percentile (p99), not just the mean: a line stops on the slowest part, not the average one.

### 16.5.3 Does quantisation earn its place?

Quantisation stores weights as 8-bit integers instead of 32-bit floats. The folklore says four times smaller and faster. Measure it:

| | Before | After dynamic int8 quantisation |
|---|---|---|
| The defect CNN: size | 64.4 KB | 64.6 KB |
| A 790k-parameter MLP: size | 3087 KB | 779 KB (4.0 × smaller) |
| The same MLP: latency | 0.07 ms | 0.09 ms |

On our CNN, quantisation does nothing: PyTorch's dynamic quantisation converts only `Linear` layers, and this network's only `Linear` is a 32 × 4 head, while nearly all the weights are in convolutions it never touches. On a large MLP it shrinks the file fourfold, but on this CPU the int8 kernels were the slower path. **Quantisation buys memory. It does not automatically buy speed.** Static quantisation with calibration, or a vendor runtime, can do better on convolutions, but only a measurement on the target hardware says whether it will.

Put the two results together. An engineer could have spent the sprint on int8 kernels, gained nothing on either axis, and shipped a model still performing at chance while reporting 92% confidence. **Measure before you optimise, pick the optimisation the measurement points to, and make sure the thing you are optimising is the thing that is actually broken.**

> **Sidebar: runtimes and accelerators**
>
> `torch.export` is PyTorch's own path to a portable graph. You will also meet:
>
> - **ONNX**, an open model format that most runtimes accept. Export with `torch.onnx.export` and run with **ONNX Runtime** on CPU, GPU or many embedded accelerators.
> - **TensorRT**, NVIDIA's optimising compiler for its GPUs and Jetson modules. It fuses layers and runs in FP16 or INT8 with a calibration set, and is where the quantisation speed-ups of the folklore usually come from.
> - **OpenVINO** for Intel CPUs and integrated GPUs, and **TensorFlow Lite / LiteRT** or vendor NPUs on microcontroller-class hardware.
>
> Each conversion is a place where numerics can change. Keep a fixed set of reference inputs and outputs, and check every converted model against the eager model before it reaches a machine. Treat the exported file like firmware: version it, record the training-data version and the monitor band with it, and keep the previous one ready to roll back.

## 16.6 On the robot: the integration pattern

In a ROS 2 system the model becomes a node that subscribes to a camera topic and publishes a verdict:

```python
class DefectInspector(Node):
    def __init__(self):
        super().__init__("defect_inspector")
        self.model = torch.export.load("defect_model.pt2").module()
        self.lo, self.hi = LO, HI                      # monitor band from TRAINING data
        self.sub = self.create_subscription(Image, "/camera/image_raw", self.on_image, 10)
        self.pub = self.create_publisher(String, "/inspection/verdict", 10)

    def on_image(self, msg):
        patches = to_patches(msg)                      # (N, 1, 32, 32) float32

        # 1. GATE FIRST: never classify an image you do not recognise
        contrast = patches.std(axis=(2, 3)).ravel()
        if np.mean((contrast < self.lo) | (contrast > self.hi)) > 0.10:
            self.pub.publish(String(data="REJECT: input out of distribution"))
            return

        # 2. Batch the whole part: the optimisation that mattered
        with torch.no_grad():
            pred = self.model(torch.from_numpy(patches)).argmax(1)

        self.pub.publish(String(data=summarise(pred)))
```

Three lessons are built in. The monitor runs **before** the model and can veto it. Patches are batched, because that was the optimisation that mattered. The exported program loads directly, with no model class on the robot.

What the sketch does not show, and what you should insist on in a real system:

- **Log** every input statistic and the prediction distribution, so that a collapse like 584 of 600 "crack" would be visible on a dashboard within minutes.
- **Keep a conventional fallback**, such as a fixed-threshold vision check, for when the gate fires.
- **Make sure a human sees the rejects**, and that their decisions flow back as labelled real data.
- **Run in shadow mode first**: the model's verdicts are logged next to the existing process for a period before they act on anything.
- **Separate the timing-critical path.** A learned perception node can sit outside the hard real-time loop; a learned controller inside it (Chapter 14) needs a watchdog and a safe fallback controller.

## 16.7 The decisive experiment

The decisive experiment for a deployment is **an evaluation on data from the deployment process**, not from the training process. Every number in this chapter that mattered came from one: the 0.267 that a perfect simulated score was hiding, the 0.278 that showed more data could not help, the 0.497 and 0.870 that measured how far randomisation reached, and the 92% confidence on wrong answers that ruled out a confidence gate. If you cannot get labelled real data before go-live, get *unlabelled* real data and run the input monitor on it. It will tell you whether you have a problem, even if it cannot tell you how big.

## Summary

- A held-out test set does not predict deployment. It is drawn from the training distribution; the factory is not.
- A CNN scoring 1.000 in simulation scored 0.267 on the factory camera and collapsed onto one class. Four times the simulated data moved that to 0.278: more data from the wrong distribution is still wrong.
- Domain randomisation helps exactly as far as it reaches: narrow 0.497, wide 0.870. You are covering corruptions, not adding magic, and in a real project you will not know which ones to cover.
- Confidence is not a safety mechanism: 79.5% of real predictions were over 90% confident, at chance accuracy. Ensembles, MC dropout and conformal prediction help in-distribution, but their guarantees assume the shift away.
- Monitor the data: one scalar per image flagged 99.8% of the shift with no labels.
- Measure before you optimise. Batching bought 2.5 ×; quantisation bought nothing; and the real fault was never speed.

## Exercises

1. **Which corruption hurts most?** Apply each of the five factory corruptions on its own to the simulated test set and measure the clean model's accuracy. Rank them. Does the ranking predict how much narrow randomisation helped?
2. **Randomise wider.** Double every randomisation range in `randomise_wide`. What happens to simulated and real accuracy? Find the width that maximises real accuracy, and explain why you could not do this search in a real project.
3. **A few real labels.** Fine-tune the wide-randomised model on 20, 50 and 200 labelled real patches. How many close the remaining gap?
4. **A second monitor statistic.** Add a sharpness statistic (the variance of a Laplacian filter). Which corruption does it catch that contrast misses? Build a combined monitor and report its in-domain false-alarm rate.
5. **Ensembles under shift.** Train five clean-simulation models from different seeds. On real images, how often do all five agree on a wrong answer? Does ensemble disagreement separate the two domains better or worse than the contrast monitor?
6. **Conformal sets, in and out of distribution.** Calibrate split-conformal prediction sets at 90% coverage on simulated validation data. Measure coverage and average set size on the simulated test and on the real test. Explain the result.
7. **Latency distribution.** Time 1000 parts at batch sizes 1, 8, 32 and 128. Plot the mean and the 99th percentile. Which batch size would you ship, and why not the fastest mean?
8. **Ready for Monday.** You are told this inspection system goes live on Monday on a line where a missed crack reaches a customer. List three things you would require before agreeing, and for each say which result from this chapter motivates it.

## Further reading

- J. Tobin et al., "Domain randomization for transferring deep neural networks from simulation to the real world," IROS 2017.
- J. Quiñonero-Candela, M. Sugiyama, A. Schwaighofer and N. D. Lawrence (eds.), *Dataset Shift in Machine Learning*, MIT Press, 2009.
- S. Rabanser, S. Günnemann and Z. C. Lipton, "Failing loudly: an empirical study of methods for detecting dataset shift," NeurIPS 2019.
- M. Hein, M. Andriushchenko and J. Bitterwolf, "Why ReLU networks yield high-confidence predictions far away from the training data and how to mitigate the problem," CVPR 2019.
- D. Hendrycks and K. Gimpel, "A baseline for detecting misclassified and out-of-distribution examples in neural networks," ICLR 2017.
- B. Lakshminarayanan, A. Pritzel and C. Blundell, "Simple and scalable predictive uncertainty estimation using deep ensembles," NeurIPS 2017.
- Y. Gal and Z. Ghahramani, "Dropout as a Bayesian approximation," ICML 2016.
- Y. Ovadia et al., "Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift," NeurIPS 2019.
- A. N. Angelopoulos and S. Bates, "Conformal prediction: a gentle introduction," *Foundations and Trends in Machine Learning* 16(4), 2023.
- D. Sculley et al., "Hidden technical debt in machine learning systems," NeurIPS 2015.
- E. Kaufmann et al., "Champion-level drone racing using deep reinforcement learning," *Nature* 620, 2023.
- PyTorch `torch.export` documentation; ONNX Runtime <https://onnxruntime.ai/>; NVIDIA TensorRT documentation; ROS 2 documentation <https://docs.ros.org/>.
