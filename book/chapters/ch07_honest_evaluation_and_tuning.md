# Chapter 7. Honest Evaluation and Tuning

*Part II. Classical ML on sensor data*

> **Case study:** predicting robot grasp failures on a bin-picking cell
> **Companion notebook:** [`lectures/lect06/supplement/mlpi26_lect06_nbsup1.ipynb`](../../lectures/lect06/supplement/mlpi26_lect06_nbsup1.ipynb)
> **Lab:** [`labs/lab06/mlpi26_Lect06_Lab.ipynb`](../../labs/lab06/mlpi26_Lect06_Lab.ipynb) (evaluation and tuning on the Chapter 3 fault dataset)
> **Built from:** Lecture 6 deck, Lecture 6 Supplement 1

Every model in the previous chapters reported a score computed on one train/test split. That was enough to compare ideas in a classroom. It is not enough to justify a decision on a production cell, where a fault detector decides when to stop a line and a grasp predictor decides whether a robot commits to a pick. This chapter is about producing numbers you can defend: numbers that come with an error bar, that were not quietly inflated by the way you prepared the data, and that measure the mistakes that actually cost money.

The chapter introduces no new model families. It introduces a protocol. By the end you will be able to take any scikit-learn estimator from Chapters 3 to 6 and say, with evidence, how good it is, whether more data would help, which hyperparameters it should use, and where its decision threshold belongs.

## Learning objectives

After this chapter you should be able to:

1. Explain why a single train/test split gives an unreliable performance estimate, and replace it with **k-fold cross-validation**, stratified when the positive class is rare.
2. Read a **learning curve** and a **validation curve** and say whether a model is underfitting, overfitting, or short of data.
3. Tune hyperparameters with **`GridSearchCV`** and **`RandomizedSearchCV`**, and choose between them given a compute budget.
4. Use **precision**, **recall**, **F1** and **average precision** instead of accuracy on imbalanced problems.
5. Read **ROC** and **precision–recall** curves, interpret **AUC**, and treat the decision threshold as a design parameter.
6. Assemble all of the above into an evaluation protocol in which the test set is opened exactly once.

---

## 7.1 The engineering problem: will this grasp fail?

A production cell uses a two-finger parallel gripper to pick parts out of a bin. Most picks work. Some do not: the part slips, tips out of the fingers, or is never lifted. A failed pick costs a re-plan and a cycle, and sometimes a damaged part. The cell has logged **1400 pick attempts**. For each attempt it recorded what it knew *before* the pick and whether the pick failed. The question is whether we can predict, before committing, that a grasp is going to fail, so that the planner can choose a different one.

The ten features are all available before the gripper closes:

| Feature | Meaning | Source |
|---|---|---|
| `object_mass_g` | Part mass | Part database / vision |
| `object_width_mm` | Grasp width across the part | Vision |
| `surface_friction` | Estimated friction coefficient | Material lookup |
| `approach_angle_deg` | Tilt of the gripper axis versus the ideal approach | Planner |
| `grasp_offset_mm` | Distance of the grasp point from the centroid | Planner |
| `grip_force_N` | Commanded closing force | Controller |
| `bin_clutter` | Number of parts touching this one | Vision |
| `ft_variance` | Force/torque variance while the jaws close | Gripper sensor |
| `gripper_temp_C` | Finger temperature | Gripper sensor |
| `cycle_count_k` | Thousands of cycles on the current finger pads | Maintenance log |

The dataset is synthetic but physically motivated, so the notebook runs anywhere with no download. Its generator encodes four mechanisms a gripper engineer would recognise. A grasp tends to fail when the grip force cannot hold the weight through friction, when the grasp point is far from the centroid (creating a moment about the fingertips), when the approach is tilted, and when the part is far from the width the fingertips were shaped for. Two of these effects are deliberately non-linear: the width penalty is quadratic, and tilt and offset interact. Pad wear lowers the effective friction over time. Section 7.4 will show that these non-linearities are exactly what separates a linear model from a better one.

The target `grasp_failed` is 1 when the pick failed. We make the *rare, expensive* outcome the positive class on purpose. Precision and recall are always computed with respect to the positive class, so "positive" should mean "the thing we are trying to catch".

### 7.1.1 The first trap: imbalance

Only **252 of 1400 picks (18.0%) fail**. A model that predicts "success" for every pick is therefore **82% accurate**. That figure sounds respectable in a report and is worthless on the cell, because it never flags anything and so never prevents a single dropped part. On imbalanced problems, accuracy mostly measures the class balance. Keep the number 0.18 in mind; it will reappear as the no-skill baseline for precision.

### 7.1.2 Lock the test set away first

Before any plot, any model or any statistic, split off a test set and stop looking at it:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)
# train: 1050 picks, failure rate 0.180
# test:   350 picks, failure rate 0.180   <- locked away
```

`stratify=y` keeps the failure rate identical in both halves. Everything from here until Section 7.8 uses only `X_train`. Every plot, hyperparameter and model comparison is decided by cross-validation inside the training set.

> **Rule.** The test set is a one-shot resource. Each time you look at it and change something in response, it becomes a little more like a second validation set, and the number it gives you becomes a little more optimistic.

## 7.2 Why one split is not enough

How much does a test score depend on the luck of the split? We can measure it directly. The notebook fits the same scaled RBF support vector machine (`C=10`, `gamma=0.01`) on **60 different random 75/25 splits** of the same data and records the test ROC AUC each time. (AUC is defined in Section 7.7; for now, read it as a score between 0.5 and 1.0 where higher is better.)

| Statistic | Test ROC AUC |
|---|---|
| Worst split | 0.799 |
| Best split | 0.922 |
| Spread | **0.123** |

Same model, same data, same preprocessing: only the random seed changed. A spread of 0.123 is larger than the gap between a good model and a mediocre one on this problem. Two engineers running identical code with different seeds could reach opposite conclusions about which model to deploy. Any comparison based on one split is partly a measurement of luck.

The cure is to average over several splits and to report how much they disagree.

## 7.3 k-fold cross-validation

Split the training data into *k* equal blocks, or *folds*. Run *k* rounds. In round *i*, train on every fold except fold *i* and validate on fold *i*. Every sample is used for validation exactly once and for training *k* − 1 times. The cross-validation estimate is the mean of the *k* validation scores, and their standard deviation tells you how far to trust that mean.

```
round 1   [VALID][train][train][train][train]
round 2   [train][VALID][train][train][train]
round 3   [train][train][VALID][train][train]
round 4   [train][train][train][VALID][train]
round 5   [train][train][train][train][VALID]
                                        ->  score = mean ± std of 5 validation scores
```

**Choosing k.** Five and ten are the usual values. A larger *k* trains each round on more data, so the estimate is less pessimistic, but it costs *k* model fits per candidate. *k* = *n* is leave-one-out, which is expensive and has high variance. For datasets of a few thousand rows, *k* = 5 is a good default, and this book uses it throughout.

**Stratification.** With a rare positive class, plain `KFold` lets the failure rate drift between folds, adding noise you did not ask for. `StratifiedKFold` pins each fold to the overall rate. On the grasp data:

| Splitter | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Spread |
|---|---|---|---|---|---|---|
| `KFold` | 0.186 | 0.168 | 0.179 | 0.186 | 0.182 | 0.018 |
| `StratifiedKFold` | 0.182 | 0.182 | 0.179 | 0.179 | 0.179 | 0.004 |

For classification, always stratify.

### 7.3.1 Cross-validation in practice

`cross_val_score` runs the whole loop: clone the estimator, fit, score, repeat.

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
fold_scores = cross_val_score(svm_pipeline(C=10, gamma=0.01),
                              X_train, y_train, cv=cv5, scoring="roc_auc")
# fold scores: [0.85  0.872 0.837 0.905 0.857]
# 5-fold estimate: 0.864 +/- 0.023
```

The report line is **0.864 ± 0.023**, not "0.905" (the best fold) and not "0.864" alone. The ± part is what stops you from over-reading a 0.005 difference between two models.

`cross_validate` returns several metrics at once, plus training scores, which already hint at the diagnostics of Section 7.4:

| Metric | Train | Validation | Validation std |
|---|---|---|---|
| ROC AUC | 0.900 | 0.864 | 0.023 |
| Average precision | 0.752 | 0.649 | 0.032 |
| Precision | 0.887 | 0.770 | 0.091 |
| Recall | 0.317 | 0.270 | 0.074 |
| F1 | 0.467 | 0.395 | 0.096 |

Notice how much noisier the threshold-dependent metrics (precision, recall, F1) are than the ranking metrics (ROC AUC, average precision). With only about 38 failures in each validation fold, a handful of borderline picks moves recall by several points.

### 7.3.2 The leak that inflates every score

`StandardScaler` learns a mean and a standard deviation from data. If you scale the whole training set once and then cross-validate, every validation fold has already contributed to the scaler. Information has crossed the fold boundary and the score comes out optimistic. The fix is mechanical: put every fitted step inside a `Pipeline`, so that cross-validation refits it on each fold's training part only.

```python
# WRONG - the scaler saw the validation folds
X_scaled = StandardScaler().fit_transform(X_train)
cross_val_score(SVC(), X_scaled, y_train, cv=cv5)

# RIGHT - the scaler is refitted inside every fold
cross_val_score(Pipeline([("scaler", StandardScaler()), ("svc", SVC())]),
                X_train, y_train, cv=cv5)
```

For a scaler on 1000 rows the bias is small, because a mean and a standard deviation barely change when you add a fifth of the data. For a feature selector, an imputer that learns from the target, or an oversampler, the bias can be large (Exercise 2 measures it). The rule is simple enough to apply without thinking:

> **Leakage rule.** If it has a `.fit()` method (scaler, imputer, PCA, feature selector, resampler), it belongs inside the `Pipeline`.

### 7.3.3 When rows are not independent

Cross-validation assumes rows are exchangeable. On real cells they often are not. Picks of the same part type, samples from the same shift, or windows cut from the same machine run are near-duplicates. If near-duplicates land on both sides of a fold boundary, the model is validated on data it has effectively already seen, and the score describes how well it remembers rather than how well it generalises to a new part. Split by group with `GroupKFold` or `StratifiedGroupKFold`, using the variable that defines "new" in deployment: part type, machine, batch or day. Exercise 7 shows how much the estimate drops when you do. Chapter 11 returns to the same problem for time series, where the group is time itself, and Chapter 17 treats it as one of the recurring failure modes of applied ML.

## 7.4 Learning curves: is it the data or the model?

A **learning curve** plots training and validation score against the number of training examples. It answers a question no single score can answer: would logging more picks help?

| What you see | Diagnosis | What to do |
|---|---|---|
| Both curves low, close together, flat | Underfitting (high bias) | More expressive model, better features |
| Training high, validation much lower, gap persists | Overfitting (high variance) | More data, regularise, reduce capacity |
| Curves converging, still rising | Still data-limited | Collect more data |

The notebook draws the curves for two deliberately bad models on the same data (Figure 7.1).

*Figure 7.1. Learning curves for an underfitting linear model (left) and an overfitting RBF SVM with γ = 3.0 (right). Generated in the companion notebook, Section 4.*

| Model | Final training AUC | Final validation AUC |
|---|---|---|
| Logistic regression (linear) | 0.808 | 0.797 |
| RBF SVM, γ = 3.0 | 1.000 | 0.697 |

Read the two panels as two different prescriptions.

- **Underfitting.** The gap is closed and the validation curve is flat at 0.797. Logging another 5000 picks would change nothing. A linear model cannot represent the quadratic width penalty or the tilt-times-offset interaction. You need a model that can bend, or features that encode the physics directly, such as the ratio of required to commanded force and the product of mass and offset. Exercise 8 shows how far features alone can take the linear model.
- **Overfitting.** A persistent gap of 0.303 AUC. Training AUC is a perfect 1.0, because with a very large γ each training point influences only its own neighbourhood and the model memorises. More data would help, but turning γ down is free. That is the job of the next section.

> **Diagnose before you tune.** A learning curve takes a few seconds and can save weeks of data collection that would not have helped.

## 7.5 Validation curves: one knob at a time

A **validation curve** plots training and validation score against one hyperparameter, with everything else fixed. It tells you which way to move a knob and where the interesting range is, which is how you choose the bounds of the search in Section 7.6.

An RBF SVM has two knobs.

- **γ** sets the reach of a training point in the kernel

  $$k(\mathbf{x},\mathbf{x}') = \exp\!\left(-\gamma\,\lVert\mathbf{x}-\mathbf{x}'\rVert^2\right).$$

  Small γ gives each point a wide, smooth influence and the model behaves almost linearly. Large γ makes the influence local and the model memorises.
- **C** is the price of a margin violation. Small C gives a soft, heavily regularised boundary. Large C makes the model try to classify every training point correctly.

Both are searched on a logarithmic scale because what matters is the order of magnitude.

*Figure 7.2. Validation curves for γ (with C = 10) and C (with γ = 0.01). Generated in the companion notebook, Section 5.*

The γ panel is the bias–variance trade-off in one picture. As γ grows, training AUC climbs to 1.0 while validation AUC rises, peaks and collapses. The peak sits at **γ ≈ 0.019 (validation AUC 0.862)**. The C sweep peaks at **C ≈ 23 (validation AUC 0.864)**. Always choose the peak of the *validation* curve; the training curve only ever says "more capacity".

> **Caution.** A validation curve moves one knob with the others frozen. C and γ interact: a larger γ can be compensated by a smaller C, and vice versa. The single-knob optimum is therefore not necessarily the joint optimum. Use validation curves to find the range, then search the knobs jointly.

## 7.6 Searching hyperparameters

### 7.6.1 Grid search

`GridSearchCV` evaluates every combination of the values you list, scores each with cross-validation inside the training set, and refits the winner on the whole training set.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {"svc__C":     [0.1, 1, 10, 100],
              "svc__gamma": [0.001, 0.003, 0.01, 0.03, 0.1]}

grid = GridSearchCV(svm_pipeline(), param_grid, scoring="roc_auc",
                    cv=cv5, n_jobs=-1, refit=True)
grid.fit(X_train, y_train)
# combinations : 20
# model fits   : 100
# best params  : {'svc__C': 100, 'svc__gamma': 0.003}
# best CV AUC  : 0.8647
```

The `svc__C` name follows the convention `<step name>__<parameter name>`, which routes a parameter to a step inside the pipeline. The pipeline is what keeps the scaler from seeing each validation fold.

The cost is the product of the number of levels times *k*: 4 × 5 × 5 = 100 fits. Add a third knob with five values and you are at 500. Grid search is exhaustive and reproducible, and its cost grows exponentially with the number of parameters. That is the whole trade-off.

The linear baseline scores a cross-validated AUC of about 0.80 to 0.82 depending on the folds, so the tuned SVM's 0.865 is a real improvement, consistent with the non-linear physics built into the data.

When you read the score surface as a heat map (Figure 7.3), form two habits.

1. **Is the winner on the edge of the grid?** If so, the grid was placed badly. Here the best C is 100, the largest value tried, so an extended grid is worth a look (Exercise 4 does this with random search).
2. **How flat is the surface?** If a dozen cells are within one fold standard deviation of the best, "the best" is mostly noise. Prefer the simplest, most regularised setting on that plateau rather than the literal arg-max.

*Figure 7.3. Mean cross-validated ROC AUC at each grid point. Generated in the companion notebook, Section 6.*

### 7.6.2 Random search

`RandomizedSearchCV` draws `n_iter` combinations from distributions you specify. You set the compute budget directly instead of inheriting it from the grid size.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

param_dist = {"svc__C": loguniform(1e-1, 1e3),
              "svc__gamma": loguniform(1e-4, 1e0)}
rand = RandomizedSearchCV(svm_pipeline(), param_dist, n_iter=20,
                          scoring="roc_auc", cv=cv5, n_jobs=-1, random_state=0)
rand.fit(X_train, y_train)
# best params : C = 6.677, gamma = 0.0188
# best CV AUC : 0.8650   (grid: 0.8647, same 100-fit budget)
```

Why does random search work so well? In most problems only one or two hyperparameters really matter, and you do not know in advance which. A grid with four levels per knob only ever tries four distinct values of the important one, however many points it evaluates; every other evaluation repeats one of those values with a different setting of a knob that does not matter. Twenty random draws try twenty distinct values of every knob (Bergstra and Bengio, 2012). Use `loguniform` for scale parameters so that the samples spread evenly over orders of magnitude.

Notice that the two searches found different winners (C = 100, γ = 0.003 versus C = 6.7, γ = 0.019) with the same score. That is the C–γ trade-off from Section 7.5, and it is another sign that the top of the surface is a ridge, not a peak.

| | Grid search | Random search |
|---|---|---|
| Budget | Fixed by the grid, ∏ nᵢ × k | You choose `n_iter` × k |
| Distinct values per knob | nᵢ | `n_iter` |
| Continuous parameters | Discretised by hand | Sampled from a distribution |
| ≤ 3 knobs, known ranges | Good choice | Fine |
| ≥ 4 knobs, or unknown ranges | Scales badly | Good choice |

A practical recipe: run random search wide first to find the promising region, then a small grid inside it to polish. For expensive models such as the networks of Part III, sequential model-based methods (Bayesian optimisation, or the successive-halving searches `HalvingGridSearchCV` and `HalvingRandomSearchCV`) spend the budget more cleverly still; the principle of evaluating by cross-validation inside the training set does not change.

### 7.6.3 The winner's score is optimistic

With twenty candidates compared on the same folds, the winner is partly the candidate that was lucky on those folds. `best_score_` is therefore a slightly optimistic estimate of the winner's true performance. The bias grows with the number of candidates and shrinks with the amount of data. There are two remedies. The cheap one is the held-out test set of Section 7.8. The thorough one is **nested cross-validation**: an outer loop of folds, each of which runs the entire search on its training part and scores the winner on its held-out part. Nested CV estimates the performance of the *whole procedure*, search included, and is the honest choice when the dataset is too small to spare a test set.

## 7.7 Metrics that match the cost of a mistake

### 7.7.1 The confusion matrix

Fix a model (the tuned SVM) and look at what kind of mistakes it makes. Every binary prediction falls into one of four cells:

| | Predicted success | Predicted failure |
|---|---|---|
| **Actually succeeded** | TN: correct, pick proceeds | **FP**: false alarm, a needless re-plan |
| **Actually failed** | **FN**: missed failure, the part is dropped | TP: caught, the planner re-plans |

From these,

$$
\text{precision} = \frac{TP}{TP+FP},\qquad
\text{recall} = \frac{TP}{TP+FN},\qquad
F_1 = 2\,\frac{\text{precision}\cdot\text{recall}}{\text{precision}+\text{recall}}.
$$

In the language of the cell, **precision** is the fraction of flagged grasps that really would have failed; low precision means crying wolf and throwing away good cycles. **Recall** is the fraction of real failures that were caught; low recall means dropped parts still get through. **F1** is their harmonic mean, which punishes a model that wins one by sacrificing the other. None of the three counts true negatives, which is exactly why 1148 easy successes cannot inflate them.

The cross-validated predictions of the tuned SVM on the training set (the test set is still closed) give:

| | Predicted success | Predicted failure |
|---|---|---|
| **Actually succeeded** | 837 | 24 |
| **Actually failed** | 123 | 66 |

Accuracy is 0.860, which looks fine. Precision is 66/(66+24) = **0.733** and recall is 66/(66+123) = **0.349**. The model is cautious: when it flags a grasp it is usually right, but it catches only about a third of failures. Whether that is acceptable is not a statistical question.

### 7.7.2 The threshold is a design decision

`predict()` applies a fixed cut-off: decision score above 0 for an SVM, probability above 0.5 for most other classifiers. Nothing forces you to keep it. Sliding the threshold trades precision against recall along a curve, and where you sit on that curve should follow from the cost of each mistake. On this cell a dropped part costs far more than a needless re-plan, so you would deliberately move the threshold to buy recall.

*Figure 7.4. Cross-validated precision–recall curve of the tuned SVM, with the default operating point marked (left), and precision and recall against the decision threshold (right). Generated in the companion notebook, Section 8.1.*

The notebook asks what it would cost to catch 70% of failures: the threshold must move from 0 to **−0.732**, and precision falls from 0.73 to **0.54**. In other words, to catch twice as many failures you accept that roughly every second flag is a false alarm. If a re-plan costs a second and a dropped part costs a minute, that is an excellent trade.

The **precision–recall curve** shows every operating point at once, and **average precision (AP)** summarises it in one number: **0.655** for the tuned SVM. The no-skill baseline for AP is the positive rate, **0.18**, not 0.5. An AP of 0.655 is therefore about 3.6 times better than chance.

When you can attach costs to the two mistakes, choose the threshold by minimising expected cost directly:

$$
\text{cost}(t) = c_{FN}\,FN(t) + c_{FP}\,FP(t),
$$

evaluated on cross-validated scores. Exercise 5 does this with c_FN = 12 and c_FP = 1. This is the single most useful habit in the chapter: it converts a modelling result into an engineering decision that a plant manager can audit.

> **State the trade-off.** F1 is convenient when you need one number to compare models. For a deployment, state the operating point explicitly: "at this threshold we catch 70% of failures and 54% of our flags are real".

### 7.7.3 ROC curves and AUC

The **ROC curve** is the other view of the same threshold sweep. It plots the true positive rate (recall) against the false positive rate

$$
\text{TPR} = \frac{TP}{TP+FN},\qquad \text{FPR} = \frac{FP}{FP+TN}.
$$

The bottom-left corner means "flag nothing", the top-right means "flag everything", and the diagonal is random guessing. A perfect classifier reaches the top-left corner. The **area under the curve (AUC)** has a clean interpretation: it is the probability that a randomly chosen failing grasp receives a higher score than a randomly chosen successful one. An AUC of 0.5 is a coin flip and 1.0 is a perfect ranking.

AUC is threshold-free. It measures how well the model *ranks*, not how well a particular cut-off classifies. That makes it a good metric for selecting models (it is the `scoring=` argument used throughout this chapter) and a poor one for deploying them, because the cell must eventually commit to a threshold.

**ROC versus PR under imbalance.** The FPR denominator is the large negative class, so hundreds of extra false alarms move the ROC curve only slightly. ROC can look flattering when positives are rare. The PR curve divides by predicted and actual positives, so it stays honest (Saito and Rehmsmeier, 2015). Report both: ROC AUC to compare models, average precision to judge usefulness.

### 7.7.4 Three models, six metrics

The notebook puts the do-nothing model next to the two real ones, all evaluated with cross-validated predictions on the training set:

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | Avg. precision |
|---|---|---|---|---|---|---|
| Always "success" | 0.820 | 0.000 | 0.000 | 0.000 | 0.500 | 0.180 |
| Linear baseline | 0.841 | 0.638 | 0.270 | 0.379 | 0.796 | 0.542 |
| Tuned RBF SVM | 0.860 | 0.733 | 0.349 | 0.473 | 0.863 | 0.655 |

Read the first row. The do-nothing model scores 0.82 accuracy, within four points of the best model, and zero on precision, recall and F1, with an AUC of exactly 0.5. Every metric except accuracy correctly reports that this model is worthless. That single row is the argument for never reporting accuracy alone on an imbalanced problem.

## 7.8 The final evaluation: open the test set once

Everything so far used only `X_train`. The model was selected, tuned, diagnosed and inspected without the test set being read. Now, and only now, we score it once.

| Metric | CV estimate (training set) | Held-out test |
|---|---|---|
| ROC AUC | 0.865 | 0.866 |
| Precision | 0.733 | 0.781 |
| Recall | 0.349 | 0.397 |
| F1 | 0.473 | 0.526 |

The test AUC of 0.866 matches the cross-validated estimate almost exactly. That agreement is the reward for the discipline: the CV estimate was honest, so the test set confirmed it rather than surprising us. The threshold-dependent metrics land slightly higher on the test set. With only 63 failures in the test set, a difference of three or four caught failures moves recall by five points, which is well inside the fold-to-fold spread we measured in Section 7.3.1. If you now went back, changed a hyperparameter and re-scored on the test set, the number would start drifting optimistic. Don't.

### 7.8.1 The protocol in one picture

```
all logged picks
      |
      +--> hold out a TEST set  (stratified, or grouped; locked away)
      |
      +--> TRAINING set
              |
              +--> k-fold CV  ---> learning curves      (is it data or the model?)
              |                --> validation curves    (which way does this knob go?)
              |                --> grid / random search (pick the hyperparameters)
              |                --> precision / recall   (what kind of mistakes? which threshold?)
              |
              +--> refit the winner on the whole training set
                        |
                        v
                 score ONCE on the test set   <-- the number you report
```

## 7.9 The decisive experiment

Each chapter in this book ends its case study with one experiment designed to catch the model being right for the wrong reason. For evaluation the decisive experiment is to **change the split, not the model**. Rerun your final number with a different random seed, with grouped folds keyed on part type, and with a split by date. If the score holds under all three, it describes the model. If it moves by more than the fold standard deviation, it describes your split, and you have found either a leak or a distribution shift before the cell found it for you. The 0.123-AUC spread in Section 7.2 is what the first of these checks looks like when it fails.

## Summary

- One split gives you a number; cross-validation gives you evidence. Report **mean ± std** across folds, here 0.864 ± 0.023 rather than any single lucky split.
- Stratify classification folds, and split by group (part type, machine, day) when rows are not independent.
- Every step with a `.fit()` goes inside the `Pipeline`, or preprocessing leaks across the fold boundary.
- Diagnose before tuning: a learning curve tells you whether to collect data or change the model; a validation curve tells you where the useful range of a knob is.
- Grid search for two or three knobs with known ranges; random search when there are more knobs or the ranges are unknown. Treat the winner's CV score as slightly optimistic.
- Accuracy is almost never the right metric on an imbalanced problem. Use precision, recall, F1 and average precision, and remember that the no-skill AP is the positive rate.
- The decision threshold is a design parameter. Choose it from the cost of each kind of mistake and state the resulting operating point.
- Open the test set exactly once, at the end.

## Exercises

1. **Fold count.** Re-run the cross-validation of Section 7.3.1 with *k* = 3, 5, 10 and 20. Plot the mean validation AUC and its standard deviation against *k*. What happens to the estimate, its spread, and the run time?
2. **Leakage, measured.** Fit a `StandardScaler` on all of `X_train`, then cross-validate a bare `SVC` on the scaled array, and compare with the pipeline version. How large is the optimistic bias, and why is it small here? Repeat with `SelectKBest(k=3)` in place of the scaler and watch the bias grow.
3. **A different model family.** Replace the SVM with `RandomForestClassifier` and tune `n_estimators`, `max_depth` and `min_samples_leaf` with `RandomizedSearchCV` (`n_iter=30`). Does it beat the tuned SVM on ROC AUC? On average precision? Does it need the scaler?
4. **Budget experiment.** Run `RandomizedSearchCV` with `n_iter` = 5, 10, 20, 40 and 80, each with ten different `random_state` values. Plot the mean and spread of `best_score_` against the budget. Where do the returns flatten out? Extend the C range to 10⁴ and check whether the grid's edge winner from Section 7.6.1 was a real limitation.
5. **Choose an operating point.** Assume a missed failure costs 12 units and a false alarm costs 1. Sweep the threshold over cross-validated `decision_function` outputs, compute the total cost, and find the minimising threshold. How far is it from 0, and what precision and recall does it give?
6. **Imbalance.** Re-run the grid search with `SVC(class_weight="balanced")`. Which metrics improve, which get worse, and does ROC AUC move at all? Explain the pattern in terms of what `class_weight` changes.
7. **Grouped data.** Suppose the 1400 picks come from only 40 part types, and rows from the same part type are highly similar. Add a synthetic `part_id` column and compare `StratifiedKFold` with `StratifiedGroupKFold(groups=part_id)`. Which gives the lower score, and which one tells the truth about a new part type?
8. **Feature engineering versus tuning.** Add two physics features, `required_force / grip_force` and `mass × grasp_offset`, and refit the linear baseline. How much of the gap to the tuned SVM does feature engineering close on its own?
9. **Nested cross-validation.** Wrap the grid search of Section 7.6.1 in an outer `StratifiedKFold(5)` with `cross_val_score`. Compare the nested estimate with `grid.best_score_`. How large is the optimism of the inner search on this dataset?

## Further reading

- scikit-learn User Guide, §3 "Model selection and evaluation": <https://scikit-learn.org/stable/model_selection.html>
- J. Bergstra and Y. Bengio, "Random search for hyper-parameter optimization," *JMLR* 13 (2012) 281–305.
- T. Fawcett, "An introduction to ROC analysis," *Pattern Recognition Letters* 27 (2006) 861–874.
- T. Saito and M. Rehmsmeier, "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets," *PLoS ONE* 10(3) (2015).
- G. Varoquaux, "Cross-validation failure: small sample sizes lead to large error bars," *NeuroImage* 180 (2018) 68–77.
- S. Varma and R. Simon, "Bias in error estimation when using cross-validation for model selection," *BMC Bioinformatics* 7 (2006) 91. The standard reference for nested CV.
- J. Mahler et al., "Dex-Net 2.0: Deep learning to plan robust grasps with synthetic point clouds and analytic grasp metrics," *RSS* 2017. The grasping side of the case study.
