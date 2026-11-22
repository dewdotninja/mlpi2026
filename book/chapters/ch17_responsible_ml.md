# Chapter 17. Responsible Machine Learning: Knowing When to Trust a Model

*Part V. From notebook to factory*

> **Case study:** a predictive-maintenance model that scores 0.979 and should never be deployed
> **Companion script:** [`lectures/lect14/lect14_leakage_demo.py`](../../lectures/lect14/lect14_leakage_demo.py) (needs only NumPy, scikit-learn and Matplotlib; runs in under a minute; `--correlated` adds the limits-of-explanation experiment)
> **Built from:** Lecture 14 deck and the leakage demo

By now you can take a sensor stream and build something that works. This chapter is about the harder question: knowing whether it does. Every capstone presenter in this course was asked the same question, and every reader should be ready for it: **"How would you know if your model were wrong?"** If a project cannot answer it, the rest of the result does not matter.

Responsible ML, for a control engineer, is not a compliance exercise or an ethics module bolted on at the end. It is the engineering discipline you already apply to everything else in the loop. You would not ship a controller without a stability margin, a failure analysis and a defined safe state. A learned component is no different, except that its failure modes are less obvious and its confidence is not a safety signal. The questions are the same: when does it fail, how would I know, what happens next, and who is accountable?

The uncomfortable part is this: **a model can be accurate, well tested and still unsafe, because accuracy is measured on data you chose.**

## Learning objectives

After this chapter you should be able to:

1. Use explainability as a debugging tool, and say what an explanation is and is not evidence of.
2. Find a leaked feature with permutation importance, and explain why no accuracy metric could have found it.
3. Choose among permutation importance, partial dependence, SHAP, saliency and Grad-CAM, occlusion, surrogates and ablation.
4. Recognise the limits of an explanation: correlated inputs, importance without causation, and persuasive plausibility.
5. Run a project past the six failure modes measured in this book.
6. Design a system that is safe with a model that is sometimes wrong: bounded authority, a reachable safe state, an input gate.
7. Specify what to log from day one, and answer the seven go-live questions.

---

## 17.1 Every chapter taught the same lesson twice

Each case study in this book ended with a decisive experiment. Look at what they were actually testing:

| Chapter | The experiment | What it showed |
|---|---|---|
| 7 Honest evaluation | Change the split, not the model | Reseeding alone moved the AUC by 0.123; a single split is partly a measurement of luck |
| 8 Neural networks | Step outside the training envelope | Whether the network learned the arm's dynamics or only the data cloud |
| 9 Training craft | Give the failing model the missing feature | Whether capacity, optimisation or sensing was the limit |
| 10 CNNs | Test on production images, not the labelling bench | A 58-point gap for the MLP and 16 points for the plain CNN that validation hid |
| 11 Sequence models | Hide the part of the sequence you think matters | 98.2% fell to 50.7%: the long-range dependency was real |
| 12 Transformers | Occlude patches, singly and in groups | Attention maps missed distributed evidence and an asymmetric rule |
| 13 RL foundations | Add wheel slip | Uncertainty changed which route was optimal, and forcing the old route crashed 22% of runs |
| 14 Deep RL | Evaluate on a robustness grid | Two policies equal on return: one crashed 100% with 80 ms of motor lag, the other 0% |
| 16 Deployment | Swap the camera | Validation accuracy of 1.000 predicted nothing: 0.267 on the floor |

Not one of them was a test of accuracy. Every one was an attempt to **catch the model being right for the wrong reason**. That instinct is the thing to take away from this book, and it is also the whole of responsible ML, applied honestly.

## 17.2 Explainability: what it is for

The usual framing is "regulators want it". The engineering reason is better: **explainability is the fastest way to catch a model that is right for the wrong reason.**

| What it is | What it is not |
|---|---|
| A set of techniques that say which inputs a model's output actually depends on | A causal account of the process |
| A debugging tool that finds shortcuts, leakage and brittle features that accuracy hides | A guarantee of correctness |
| Evidence about the *model* | A substitute for testing on data from the deployment distribution |

An explanation tells you what the model used, not what is true of the world.

## 17.3 Case study: a model that scores 0.979 and is worthless

A machine shop wants to predict spindle failure from its maintenance database: 4000 machine-months, a failure rate of 0.347, so the majority-class baseline is 0.653. Five of the six columns are genuine physical signals: vibration RMS, temperature, hours since service, load and spindle age. The sixth, `work_order_raised`, is a clerical field. The maintenance system sets it once a technician has already been dispatched.

It is in the historical database. It correlates almost perfectly with failure (in the synthetic data it is raised for 97% of failures and 2% of healthy machines). And at the moment you actually need a prediction, it is empty, because nobody has been sent yet. The failure causes the work order, not the other way round.

### 17.3.1 The model anyone would ship

A 300-tree random forest on all six features, with an honest stratified 70/30 split:

| | Test accuracy |
|---|---|
| Majority-class baseline | 0.653 |
| Random forest, all six features | **0.979** |

That would pass every review you have ever sat in. Decide now whether you would deploy it.

### 17.3.2 What is it actually using?

**Permutation importance** (Breiman, 2001) asks a simple question: shuffle one column of the test set, breaking its relationship with the target, and how much accuracy do you lose? Nothing is retrained.

```python
from sklearn.inspection import permutation_importance

pi = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=0)
for i in pi.importances_mean.argsort()[::-1]:
    print(f"{names[i]:>22}  {pi.importances_mean[i]:+.4f} ± {pi.importances_std[i]:.4f}")
```

| Feature | Accuracy lost when shuffled |
|---|---|
| `work_order_raised` | **+0.379** |
| `vibration_rms` | +0.006 |
| `temperature` | +0.005 |
| `hours_since_service` | +0.004 |
| `spindle_age` | +0.001 |
| `load` | +0.000 |

The leak is about **66 times** more important than the best physical sensor. The model ignored the physics almost entirely. It learned to read the answer.

### 17.3.3 Which one would you ship?

Drop the leaked column and refit:

| | Accuracy | At prediction time |
|---|---|---|
| Model A, all six features | **0.979** | Cannot make a prediction when it matters |
| Model B, physical sensors only | 0.839 | Works |

Model B is worse on paper and 0.187 above the baseline, so the sensors really do carry signal. Its importances are also physically sensible: vibration (+0.098), then temperature (+0.080), then hours since service (+0.048), then load (+0.023) and spindle age (+0.004). That ordering is itself a free sanity check. **Ship B.**

Three things to keep:

1. **This is not overfitting.** Model A generalises perfectly to held-out historical data, and the train/test split is completely honest. The split is fine; the feature is not.
2. **No accuracy-based metric could have caught this**, on any split, because the leak is present in all the historical data. Only looking at *what the model uses* exposed it.
3. **A model that scores worse can be the better model.** The difference is invisible to every metric on the validation report.

The general test for leakage is a question about time, not statistics: *for each input, at the moment the prediction is needed, does this value exist yet, and was it produced without knowledge of the outcome?* Chapter 7 met the same failure through preprocessing fitted on the whole dataset, and Chapter 11 through windows split by sample instead of by sequence, and targets that leak the future through an off-by-one horizon.

## 17.4 The explainability toolkit

| Technique | What it tells you | Use it when |
|---|---|---|
| Permutation importance | How much performance depends on each input | Any model, tabular features: start here |
| Partial dependence and ICE curves | The shape of the relationship, not just its size | Checking that a trend is physically sensible |
| SHAP values | Each input's contribution to *one* prediction | Explaining individual decisions; auditing subgroups |
| Saliency, Grad-CAM, attention maps | Which pixels or time steps the output used | Images and sequences (Chapters 10 to 12) |
| Occlusion | What happens when a region is blanked | Testing a saliency or attention map (Chapter 12) |
| Surrogate models | A simple model that mimics a complex one | Explaining a decision to a non-specialist |
| Ablation | What happens when a component or input group is removed | Deciding which part earns its keep |

You have already used several of these. The hide-the-first-55% test of Chapter 11 and the shuffle test of Chapter 12 are ablations; the patch occlusion of Chapter 12 is occlusion. You just called them experiments.

### 17.4.1 Partial dependence

Permutation importance says *how much* a feature matters; partial dependence says *how*. Sweep one feature across its range, holding each row's other features fixed, and average the predictions. For the honest maintenance model, failure probability should rise with vibration, temperature and hours since service. A curve that falls, or a step at a suspiciously round number, is a question to take to the people who own the data.

```python
from sklearn.inspection import PartialDependenceDisplay
PartialDependenceDisplay.from_estimator(model_b, X_test_b, ["vibration_rms", "hours_since_service"],
                                        kind="both")          # average curve plus per-row ICE lines
```

### 17.4.2 SHAP

SHAP (Lundberg and Lee, 2017) attributes one prediction to its inputs using Shapley values from cooperative game theory: each feature's average marginal contribution over all orders in which features could be added. The attributions sum to the difference between this prediction and the average prediction, which makes them easy to read. For tree ensembles, `shap.TreeExplainer` computes them exactly and quickly.

```python
import shap
explainer = shap.TreeExplainer(model_a)
sv = explainer.shap_values(X_test)           # one row of attributions per prediction
shap.summary_plot(sv, X_test, feature_names=names)
```

On the leaky model the summary plot puts `work_order_raised` on top for nearly every row, which is the same diagnosis as permutation importance, reached per prediction. SHAP is most useful where permutation importance is weakest: explaining *one* decision ("why was machine 14 flagged?") and comparing subgroups (does the model rely on different features for old spindles than new ones?). It inherits the correlated-feature problem of Section 17.5 and adds its own: "fixing" a feature at a background value can create machine states that never occur.

### 17.4.3 Saliency and Grad-CAM for images

For a CNN, a **saliency map** is the gradient of the class score with respect to the input pixels. Raw gradients are noisy. **Grad-CAM** (Selvaraju et al., 2017) is coarser and more robust: weight the last convolutional layer's feature maps by the average gradient of the class score with respect to each map, sum, and keep the positive part.

```python
feats, grads = {}, {}
layer = model.features[-1]                                  # last conv block
layer.register_forward_hook(lambda m, i, o: feats.update(a=o))
layer.register_full_backward_hook(lambda m, gi, go: grads.update(g=go[0]))

score = model(x)[0, cls]; model.zero_grad(); score.backward()
w = grads["g"].mean(dim=(2, 3), keepdim=True)               # one weight per channel
cam = torch.relu((w * feats["a"]).sum(1))                   # (1, H', W'), upsample to the image
```

On the solder-joint inspection CNN of Chapter 10, a trustworthy Grad-CAM for "insufficient solder" lights the fillet, not the board edge, the silkscreen or a corner the lamp always brightens. If it lights a fixture, you have a shortcut. Exercise 4 asks you to check. Then **test the map with occlusion**, exactly as Chapter 12 tested attention: blank the highlighted region and confirm the prediction changes; blank an unhighlighted region and confirm it does not. Saliency methods have been shown to produce plausible maps even for randomised networks (Adebayo et al., 2018), so a map that is never tested by intervention is decoration.

## 17.5 The limits of an explanation

Explanations are evidence, not proof. Three failure modes are worth knowing before you put one in a report.

**Correlated inputs share credit.** If two inputs carry the same information, shuffling one leaves the other to carry the signal, so permutation importance can show *both* as unimportant. The leakage demo measures this. Replace the single work-order column with k independent, 95%-accurate copies of the same event, as happens when a second maintenance system logs it too. Nothing changes about what the model relies on:

| Copies of the leak | Accuracy | Importance of each copy | Best physical sensor |
|---|---|---|---|
| 1 | 0.976 | +0.353 | 0.068 |
| 2 | 0.978 | +0.101, +0.087 | 0.015 |
| 4 | 0.979 | +0.006, +0.003, +0.002, +0.000 | 0.006 |

Read the accuracy column first: it does not move. The model is just as dependent on the work-order signal with four copies as with one. Now read across. With one column the leak is unmissable. With four, the largest copy has sunk to the level of an ordinary sensor, and an engineer scanning the table would conclude that nothing matters very much. The explanation did not become wrong; it answered a different question. It reports what happens when you remove *one* column, and the answer is "nothing", because the others still carry the signal. **When you suspect a group, drop the whole group and refit.** Grouped permutation importance (permute the group together) and conditional importance are the formal versions.

**Importance is not causation.** A feature can matter to the model and have no causal role in the process. The work order is the extreme case: it is caused by failure. Acting on an important feature as if it were a lever ("reduce work orders to reduce failures") is how projects produce confident nonsense.

**A plausible explanation is persuasive.** A saliency map that highlights roughly the right region is reassuring, and is not evidence that the model is using it for the right reason. Chapter 12 showed an attention map that looked unremarkable while the model applied an asymmetric rule only group occlusion could reveal.

Use explanations to look for trouble, not to argue there is none. **An explanation that confirms what you hoped deserves more suspicion, not less.**

## 17.6 Failure modes: the checklist

Every one of these appeared in a case study in this book, measured rather than described. Run your own project past the list.

| Failure mode | What it is | Where you measured it |
|---|---|---|
| **Leakage** | Information in training that will not exist at prediction time | Chapter 7 (preprocessing), Chapter 11 (window splits and horizons), this chapter (0.979 from a clerical field) |
| **Shortcut learning** | A cue that works in your data and not in the world | Chapter 10 (the MLP collapsed on production images), Chapter 12 (the asymmetric finish rule) |
| **Distribution shift** | Deployment data from a different process | Chapter 16 (1.000 in simulation, 0.267 on the floor) |
| **Overconfidence** | High stated confidence while wrong | Chapter 16 (mean confidence 0.924 on wrong answers, at chance accuracy) |
| **The wrong objective** | Optimising something that is not the goal | Chapters 13 and 14 (the reward is the entire specification; DQN scored well and chattered 15× more) |
| **Optimising the wrong thing** | Effort spent where there was no bottleneck | Chapter 16 (int8 quantisation bought nothing; batching bought 2.5×) |

You did not only read about these failures. You measured every one of them, which is the best reason you will recognise them again.

## 17.7 Safety: the model is one component in a loop

A model does not act. A system acts. Safety is a property of the system, so **design the system to be safe with a model that is sometimes wrong.**

- **Bound the authority.** Limit what the model is allowed to command: rate limits, range limits, and a conventional controller or interlock that owns the actual actuation. A learned feed-forward term (Chapter 8) gets a clamp; a learned policy (Chapter 14) runs inside a safety filter.
- **Define the safe state.** Decide what the system does when the model is unavailable or distrusted: stop the conveyor, route the part to manual inspection, hold position. It must be reachable *without* the model.
- **Gate the input.** Run the distribution monitor before inference (Chapter 16). A model that is asked nothing cannot answer wrongly.

Ask the control-engineering question: *what is the worst thing this component can command, and what stops it?* If the answer is "the model is accurate", you have no answer.

Existing functional-safety practice applies. IEC 61508 and ISO 13849 for machinery, and ISO 26262 and ISO 21448 (SOTIF, safety of the intended functionality) for vehicles, are written around components whose behaviour can be specified and verified, which a learned model's often cannot. The practical consequence today is architectural: keep the learned component out of the safety function, or wrap it in something that is verifiable. Standards specifically for machine learning in safety-related systems are still emerging; check the current versions that apply to your sector rather than relying on this paragraph.

## 17.8 Monitoring: what to log from day one

In production you have no labels. Everything you can watch is on the input and output side, which, as Chapter 16 showed, is often enough.

| Log this | And it catches |
|---|---|
| Input statistics (mean, spread, sharpness) | A new camera, lamp, batch or sensor: Chapter 16 flagged 99.8% of the shift this way |
| The prediction distribution | Collapse onto one class, long before anyone reports a bad part |
| The rate of the rarest class | A defect type quietly disappearing from the output |
| The gate or reject rate | The model being asked questions outside its training range |
| Latency per part (mean and p99) | A system drifting toward its cycle-time budget |
| Delayed ground truth (gauge results, returns) | Concept drift that no input statistic can see |

And log a **sample of the inputs themselves**. When something does go wrong, the single most valuable thing you can have is the data that caused it.

Decide in advance what each signal triggers. A shift in input statistics might pause automatic decisions and alert an engineer; a sustained change in the rare-class rate might trigger relabelling and retraining. A retraining trigger with no named owner does not exist.

## 17.9 Before go-live: seven questions

1. What distribution was this trained on, and how does it differ from the deployment one?
2. What does the model use? Have you looked, or are you assuming?
3. What happens when it is wrong, and who or what catches it?
4. What is the safe state, and can it be reached without the model?
5. What is logged, and who reads it?
6. What is the retraining trigger, and who owns it?
7. Would you sign your name under it?

The last one is not rhetorical. It is the only question on the list that cannot be delegated.

### 17.9.1 Writing it down

Two lightweight documents make these answers durable. A **model card** (Mitchell et al., 2019) records what a model is for, what it was trained and evaluated on, its measured performance including by subgroup and condition, and its known limits. A **datasheet for the dataset** (Gebru et al., 2021) records how the data was collected, by whom, under what conditions, and what it does not cover. For a factory model, "subgroup" means machine, shift, product variant, camera and season. The card for the honest maintenance model would state, among other things, that `work_order_raised` was excluded and why; that line alone would save the next engineer the discovery.

## 17.10 Where this book stops

Fourteen chapters buy a working foundation and an honest instinct. Immediately beyond them:

- **Methods**: generative models for synthetic data, graph neural networks for multi-robot systems and kinematic chains, Bayesian and ensemble uncertainty, model-based RL.
- **Scale and engineering**: distributed training, data versioning and ML pipelines, hardware accelerators on embedded targets.
- **Assurance**: formal verification of learned components, conformal prediction for calibrated sets and intervals, and safety standards for autonomy.

Of these, **conformal prediction** is the one to read first (Chapter 16's sidebar and Angelopoulos and Bates, 2023). It gives a set or interval with a guaranteed coverage rate under exchangeability: the honest answer to a confidently wrong model *on the distribution it was calibrated on*, combined with an input monitor for when the distribution changes.

## 17.11 The decisive experiment

The decisive experiment for responsible deployment is to **try to break your own result before someone else does**. Concretely: list every input and ask whether it exists at prediction time; compute permutation importance, by group as well as singly, and check that the ordering is physically sensible; test any map you intend to show with an intervention; evaluate on data from the deployment process; and write down the worst thing the model can command and what stops it. On the maintenance model, one permutation-importance table turned a 0.979 that would have passed every review into a model that could not make a single real prediction.

## Summary

- A number on a slide is a claim, not a fact. Ask what distribution it was measured on and what would have made it come out differently.
- Look at what the model uses. A 0.979-accurate model had learned only a clerical field, 66 times more important than any sensor; the honest model scored 0.839 and works.
- Explanations are evidence, not proof. Four correlated copies of the leak made it look unimportant while accuracy did not move. Drop groups, not columns, and test maps by intervention.
- Six failure modes recur: leakage, shortcuts, distribution shift, overconfidence, the wrong objective and optimising the wrong thing. This book measured each one.
- Confidence is not a safety signal. Gate on the input, not on the model's opinion of itself.
- Design the system to be safe with a model that is wrong: bounded authority, a reachable safe state, an input gate, and logs someone reads.

## Exercises

1. **Weaken the leak.** In `make_data`, raise a work order for only 60% of failures instead of 97%. Does permutation importance still flag it? At what rate does the leak stop being visible, and does test accuracy warn you?
2. **Fix the diagnosis.** Run with `--correlated`, then drop *all* work-order columns and refit. What accuracy do you get, and what does it tell you that the importance table could not? Implement grouped permutation importance and confirm it recovers the leak.
3. **Redundant sensors.** Delete `vibration_rms` from the honest model. Does `temperature` become more important? Explain the result in terms of what the two sensors measure.
4. **Grad-CAM audit.** Compute Grad-CAM for the Chapter 10 solder-joint CNN on 20 correctly classified and 20 misclassified production images. For each misclassification, say whether the map points to a shortcut, then confirm or refute it with occlusion.
5. **SHAP versus permutation.** Compute SHAP values for Model A and Model B. Do the global rankings agree with permutation importance? Find one machine whose individual explanation differs from the global ranking, and explain why.
6. **Partial dependence as a sanity check.** Plot partial dependence of Model B's failure probability on each sensor. Which curve would you question if you saw it in a review, and what would you ask the data owner?
7. **Find your leak.** Invent a leak for your own project dataset: a column available in historical data but not at prediction time. If you cannot think of one, check again; most real datasets have at least one.
8. **Bound the authority.** For the quadrotor PPO policy of Chapter 14, design a safety filter: limits on tilt command, rate of change, and a fallback to the cascaded PD controller. Evaluate crash rate on the robustness grid with and without it.
9. **Write the card.** Write a one-page model card for the domain-randomised defect classifier of Chapter 16, including its measured performance in and out of distribution, the monitor band, the safe state and the retraining owner.

## Further reading

- L. Breiman, "Random forests," *Machine Learning* 45, 5–32, 2001 (permutation importance).
- C. Molnar, *Interpretable Machine Learning*, 2nd ed., 2022, <https://christophm.github.io/interpretable-ml-book/>.
- S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," NeurIPS 2017 (SHAP).
- R. R. Selvaraju et al., "Grad-CAM: visual explanations from deep networks via gradient-based localization," ICCV 2017.
- J. Adebayo et al., "Sanity checks for saliency maps," NeurIPS 2018.
- R. Geirhos et al., "Shortcut learning in deep neural networks," *Nature Machine Intelligence* 2, 665–673, 2020.
- S. Kaufman, S. Rosset, C. Perlich and O. Stitelman, "Leakage in data mining: formulation, detection, and avoidance," *ACM TKDD* 6(4), 2012.
- G. Hooker, L. Mentch and S. Zhou, "Unrestricted permutation forces extrapolation: variable importance requires at least one more model," *Statistics and Computing* 31, 2021.
- M. Mitchell et al., "Model cards for model reporting," FAT* 2019.
- T. Gebru et al., "Datasheets for datasets," *Communications of the ACM* 64(12), 2021.
- D. Amodei et al., "Concrete problems in AI safety," arXiv:1606.06565, 2016.
- D. Sculley et al., "Hidden technical debt in machine learning systems," NeurIPS 2015.
- A. N. Angelopoulos and S. Bates, "Conformal prediction: a gentle introduction," *Foundations and Trends in Machine Learning* 16(4), 2023.
