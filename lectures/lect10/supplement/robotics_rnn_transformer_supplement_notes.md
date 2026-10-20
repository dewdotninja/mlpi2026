# Robotics supplements to ch. 15 (RNN) and ch. 16 (transformers) — 2 decks + 2 notebooks (generated 2026-10-19)
 
Supplementary pair to the CNC-cell lectures on ch. 15 and ch. 16 (Raschka, Liu & Mirjalili).
User asked for different, robotics examples; chose **mobile robot IMU** as the scenario and asked the
transformer pair to emphasise **long-range memory, interpretability and multi-step forecasting**.
 
## Deliverables
 
- `mlpi26_Lect10_supplement2.pptx` — 12 slides (dark title/summary + 10 light), notes 1 050–1 220 chars.
- `mlpi26_Lect10_nbsup2.ipynb` — 34 cells, executed, 0 errors / 0 stderr, ~5.5 min on 2 cores.
- `mlpi26_Lect11_supplement2.pptx` — 11 slides, notes 1 030–1 500 chars.
- `mlpi26_Lect11_nbsup2.ipynb` — 32 cells, executed, trains 12 models, ~16 min on 2 cores
  (`SWEEP_K = [4]` cuts it to ~10 min).
All QA'd via LibreOffice render (959.98 × 540 pt). Every deck number is read from `results_a.json` /
`results_b.json`; the notebooks reproduce them exactly (same seeds, same code via `module_body()` inlining).
## Lecture 10 Supplement 2 — differential-drive rover, 10 Hz, runs 8–20 s (lengths multiples of 10)
 
Channels v_cmd, w_cmd, v_left, v_right, gyro_z, acc_x, acc_z. Terrain concrete/grass/gravel via acc_z
(white / AR(0.55) / impulses 0.06 per step). Slip episodes (0.6/1.6/2.4 per 10 s) add 0.10–0.35 m/s to
one or both encoders. **Per-run gyro bias N(0, 0.04)** — this is what makes sensor fusion necessary.
900 / 300 / 600 runs (seeds 7 / 108 / 209). Hidden 48, Adam 3e-3 cosine, batch 64, clip 1.0,
gates opened (+1 bias on LSTM forget / GRU update slice).
 
| result | value |
|---|---|
| heading RMSE at run end: wheels / gyro / RNN / LSTM / GRU | 0.66 / 0.59 / 1.33 / 0.21 / 0.13 rad |
| gradient reach lag 100, init: RNN / LSTM / GRU | 1.2e-24 / 9.6e-08 / 2.5e-09 |
| trained LSTM reach lag 10 / 190 | 0.35 / 0.25 (RNN erratic, 1.40 at 190) |
| LSTM cell units tracking ψ | 19/48 with \|r\|>0.9, best r=0.974, its forget gate mean 0.76 (not ≈1!) |
| terrain: RNN / LSTM / GRU / 5 hand features + logreg | 0.795 / 0.818 / 0.963 / 0.972 |
| slip F1: rule / causal LSTM / BiLSTM | 0.665 / 0.972 / 0.968 |
| packing: h_n error padded / packed; CPU time packed vs bucketed | 0.163 / 6e-08; 0.854s vs 0.098s |
 
Deck A: title · one rover three questions · shapes · dead reckoning = RNN · pad/pack/bucket · heading
fusion result · x–y path · gradient reach · inside the LSTM · terrain belief vs time · slip causal vs bi · summary.
 
## Lecture 11 Supplement 2 — warehouse patrol rover
 
**Loop closure** (`make_loop_batch`, vectorised): T=128 samples @10 Hz, K earlier floor features with
4-D fingerprints N(0,1)×1.6 bump, query at the end = re-observation (σ 0.15) or new. Fresh 6000 runs per
epoch, 20 epochs, test 1000 (seed 209). Shared conv stem (k8, s4 → 31 tokens), LSTM 96 vs 2 Pre-LN blocks d48.
 
| K | LSTM | transformer | oracle |
|---|---|---|---|
| 1 | 0.901 | 0.993 | 0.997 |
| 2 | 0.716 | 0.969 | 0.995 |
| 4 | 0.592 | 0.946 | 0.991 |
| 8 | 0.552 | 0.923 | 0.985 |
 
AUC by gap at K=4 is flat for both → capacity, not distance, is what defeats the LSTM at 31 tokens.
Attention (last layer, positives): match 0.28; argmax on match 0.88.
Occlusion: P 0.91 → 0.12 removing match, 0.91 removing another.
 
**Patrol forecasting**: fixed lap 60 steps @5 Hz, random route per run, ±4 % lap gain, ctx 150 → horizon 40,
direct forecaster with 40 blank future slots + flag. 2000 train runs, 30 epochs.
Mean RMSE: persistence 0.388, LSTM 0.267, transformer 0.040,
copy-one-lap 0.050. Layer 1 attention peaks at lag 60
(0.24 mass near 1 lap, 0.20 near 2); layer 2 mostly attends
to future slots (0.40). No-PE: forecast identical across horizons (spread 0.0).
 
**Honest boundary**: terrain GRU 0.963 vs transformer 0.970 (4.6× params).
Compare with the *best* RNN — vs the LSTM (0.818) the transformer would look like a big win.
 
Deck B: title · bottleneck · attention as lookup · loop task · capacity sweep · attention+occlusion ·
patrol task · error vs horizon · lap-period attention + no-PE · costs/when to keep RNN · summary.
 
## Load-bearing choices — do not "fix"
 
- **Gyro bias** is what makes heading a fusion task; without it gyro integration nearly wins.
- **Loop closure: continuous fingerprints and K>1.** With K=1 the LSTM is fine (0.90); capacity is the lesson.
  Fresh data each epoch is required — with a fixed 4000 runs the transformer overfit to 0.76.
- **Forecasting must be direct (blank future slots), fixed lap period.** Teacher-forced next-step
  training never learned to copy (both models ≈0.25–0.30 RMSE). Variable period is left as exercise 2.
- The vanilla RNN *improving* its gradient reach after training but still failing is real; do not hide it.
- Terrain hand features tie the GRU; the LSTM is notably worse (gravel↔concrete). Keep as a lesson.
## Reuse notes / gotchas
 
- **Packed LSTM on CPU is 5–9× slower than unpacked.** Bucket by length (lengths multiples of 10) —
  `Buckets` class in `seq_common.py`. Packing is taught with a 4-run demo only.
- Two python jobs on 2 cores roughly triple wall time; run trainings sequentially.
- `pkill -f "until ..."` from bash kills the calling shell itself (matches its own command line).
- `Block.keep=True` to store attention; `need_weights=False` otherwise (faster).
- Scripts (session workspace only): rover_common.py, seq_common.py, patrol_common.py, b_models.py,
  train_a.py, train_b.py, make_figs_a/b.py, build_nb_a/b.py, build_deck_a/b.js, deckkit.js, nbkit.py, figstyle.py.
- figstyle `save()` must pass `facecolor=fig.get_facecolor()` or dark workflow strips get a light border.
 
# Lecture 10 Supplement 2 Slide Notes

## Slide 1
This is a supplement to the chapter 15 lecture. The main lecture used a CNC turning cell; here the same ideas come back on a completely different machine, a small two-wheeled rover, so that students see every concept twice in two settings.

The arc of the session: we start by showing that robotics already uses a recurrent network every day, dead reckoning, and then ask what a learned recurrent network can add. Three tasks on one dataset give us the three things to compare: heading (memory over the whole run), terrain (evidence accumulation) and wheel slip (per-step labelling).

Two promises to make at the start. First, every number on these slides is computed by the notebook; nothing is invented. Second, every network will be compared to the physics baseline an engineer would write first. Sometimes the network wins by a lot, sometimes it ties a five-feature logistic regression, and both outcomes are worth teaching.

Lecture 11 supplement 2, the transformer pair, picks up exactly where this deck ends: the point where a fixed-size hidden state runs out.



## Slide 2
The job of this slide is to make the data feel physical before any model appears.

Left panel: the true path of one test run. The robot never measures this; it is what we want to reconstruct. Orange dots mark the steps where a wheel was slipping.

Right panel, top to bottom: the gyro and the yaw rate computed from the two wheels. They agree almost everywhere, and the places where they disagree are exactly the orange slip bands. Middle: the wheel speeds against the command; during slip one wheel reports far more speed than the body has. Bottom: vertical vibration, which carries the terrain signature.

Point at the three questions in the callout and ask the room: which of these needs memory of the whole run, which needs only the last few steps, and which needs to collect evidence? Heading needs the whole run, because heading is an integral. Slip is mostly local. Terrain needs accumulated evidence. That classification is the outline of the lecture.

Slip shares: 8.5 % of steps on concrete, 21.0 % on grass, 31.7 % on gravel. 900 training runs, 600 test runs, all generated by a short simulator inside the notebook.



## Slide 3
This slide repeats the chapter's sequence-shape taxonomy on the rover tasks.

Left: many-to-one. Terrain classification reads one output after the last time step. Middle: many-to-many, synchronised. Both slip detection and heading estimation produce an output at every step. Right: sequence-to-sequence, where the outputs are future values; we leave that for Supplement B, where forecasting a patrol robot is one of the tasks.

The second bullet carries the important nuance. Shape and memory are separate decisions. Slip and heading have the same shape, but slip at step t can be decided from steps t−2 to t, whereas heading at step t depends on every gyro sample since the start. So a model that is excellent at slip can be terrible at heading. We will see exactly this: the vanilla RNN does well on slip-like tasks and fails on heading.

Question for the room: is there a robotics task that is many-to-one but needs very long memory? Good answers: did the robot return to its starting point, did it pass through a forbidden zone at any time. That is the loop-closure task in Supplement B.



## Slide 4
This is the slide that should make recurrent networks click for engineers.

Left of the figure: the familiar dead-reckoning loop drawn as an unrolled RNN. Each box is the heading, each arrow between boxes multiplies by one, each arrow from below adds Δt times the gyro reading. It is literally an RNN with one hidden unit, a recurrent weight of 1, an input weight of Δt and no nonlinearity. The code box shows the loop; it agrees with numpy's cumulative sum to float precision.

Right of the figure: what happens if the recurrent weight is not exactly 1. An impulse decays as rho to the k. With rho 0.95 the half-life is 13.5 steps, with 0.8 it is 3.1 steps. A tanh RNN that wants to integrate must keep its effective gain at 1 for hundreds of steps, which is exactly the knife edge between vanishing and exploding gradients from the chapter.

The bottom of the code box is a reminder that the same check works for a real nn.RNN: writing tanh of W x plus W h plus b in a loop reproduces PyTorch to 1.5×10⁻7.

Ask: what other robotics estimators are recurrences? Kalman filters, complementary filters, PID integrators. All of them carry a state forward. That is the whole idea.



## Slide 5
Every practical sequence dataset is ragged; here runs last 80 to 200 steps. This slide gives students a decision rule instead of a ritual.

Left schematic: four runs of length 40, 30, 20 and 10 padded to 40. 37.5 % of that batch is padding. Right schematic: the alternative, grouping runs of equal length so a batch never contains padding.

The three boxes are measured in the notebook against running each sequence on its own. With plain padding, a forward LSTM's output at the last real step is exactly right, error zero, because of causality: the padding comes later. But the final hidden state h_n has kept stepping through zeros, error 0.16, and a bidirectional layer reads the padding first in its backward direction, error 0.10. Packing fixes both to float precision.

The catch is speed. On a CPU a packed LSTM ran at 0.85 s per batch, the plain one at 0.10 s. Our run lengths are multiples of 10 steps, so there are only 13 lengths and bucketing is free. The notebook trains every model that way.

Common student question: is bucketing a bias? Only if length correlates with the label and buckets are visited in order; we shuffle the batch order each epoch.



## Slide 6
This is the main result of the RNN section. Read the bars first, then the curves.

Right panel, final heading error after a full run: wheel odometry 0.66 rad, gyro integration 0.59, vanilla RNN 1.33, LSTM 0.21, GRU 0.13. All three networks have the same input, the same hidden size of 48, the same 30 epochs.

Left panel: error as a function of time into the run. The gyro curve rises linearly, which is the signature of a constant bias integrated over time. The wheel curve rises in steps, one step per slip episode. The vanilla RNN rises fastest of all; it has not learned to integrate. The LSTM and GRU stay low over the whole run.

Why can the gated networks beat both sensors? Because they can do what an engineer would do with a Kalman filter: while the wheels agree with the gyro, the difference between them is the gyro bias; during slip, ignore the wheels. Nobody told the network this. It found it from the loss.

Question: is a 0.13 rad error good? It is about 7 degrees after up to 20 s with a biased gyro and slipping wheels, against a heading spread of 2.29 rad. For a short run that is a usable estimator.



## Slide 7
Same models, one test run, shown the way a robotics student will recognise.

Left: the dead-reckoned path. Black is the true path, never measured. The purple dashed path from wheel odometry turns too far at each slip and spirals away; the yellow gyro path bends steadily because of the bias; the red vanilla RNN path is wrong in its own way; the aqua GRU path sits on top of the truth.

Right: the heading traces behind those paths, with the slip episodes shaded orange. Watch the wheel trace jump at each orange band and the gyro trace drift slowly away between them. The GRU follows the truth through both.

The end-point errors in the callout are for this one run only; it was picked as a long run with plenty of slip and turning, so it is a demanding case, not an average. The averages are on the previous slide.

This is also the right moment to say what a network does not give you: no covariance, no guarantee, no physical interpretability by construction. A Kalman filter gives you those. Exercise 8 asks students to build the filter and compare; it is an honest fight.



## Slide 8
This slide turns the BPTT story from the chapter into a measurement on our own task.

What is plotted: the size of the gradient of the final heading estimate with respect to the gyro reading k steps earlier, averaged over 64 test runs of length 200 and normalised to its value at lag zero. A perfect integrator would give a flat line at 1, the dotted line: a gyro reading from 19 seconds ago should matter as much as one from now.

Left panel, untrained models. All three collapse: the vanilla RNN is at 1.2×10⁻24 by lag 100, the GRU at 2.5×10⁻9, the LSTM at 9.6×10⁻8. So even an LSTM does not start out with long memory; it starts out with a path along which long memory can be learned.

Right panel, after training. The LSTM and GRU curves are close to flat: early gyro readings still count. The vanilla RNN's curve is erratic, and at the longest lags it even exceeds 1 (1.40 at lag 190). That is the other face of the same problem: with only W_hh and tanh it can forget or amplify, but not hold.

The practical takeaway: when a sequence model underperforms, plot this curve before tuning anything else. It tells you directly how far back the model can learn from.



## Slide 9
Students often treat an LSTM as a black box. This slide opens it, using only the equations from the chapter.

The notebook writes the LSTM step by hand: input, forget, cell and output gates, c equals f times c plus i times g, h equals o times tanh of c. It matches PyTorch's own LSTM to 6.0×10⁻7, so we can trust the hand-computed cell states.

Left panel: three test runs. The black line is the true heading; the dashed coloured line is cell unit 4, rescaled by a straight-line fit. They lie on top of each other, correlation 0.974.

Right panel: all 48 cell units sorted by how well they correlate with heading. 19 of them exceed 0.9. The heading is stored in a distributed way, which is typical.

Now the honest surprise in the callout. The textbook picture says a memory unit should have its forget gate near 1, so the cell is a pure integrator. Here the best unit's forget gate averages 0.76, with a minimum of 0.54. The network keeps the heading alive through the loop from c to h and back into the candidate g. Gradient descent finds a memory mechanism, not necessarily the one we would design. Good discussion question: does that matter, if the gradient reach on the previous slide is flat?



## Slide 10
Terrain is the classic many-to-one task: one label per run, read off the last hidden state.

Left panel: we take the trained GRU and simply feed it the first k steps of each test run. Because the model is causal, that is exactly the decision it would make online after k steps. Accuracy rises from 36.8 % after half a second to 91.7 % after eight seconds, and the mean confidence rises with it. The hidden state behaves like a belief that accumulates evidence. Chance is one third.

The yellow line is the honest baseline: five features an engineer would compute from vibration, namely standard deviation, lag-one autocorrelation, kurtosis, speed-normalised standard deviation and mean absolute difference, with logistic regression. It matches or beats the GRU at most horizons and reaches 97.2 % on full runs.

Right panel: the GRU confusion matrix. Most errors put gravel and concrete together; they differ only in occasional impulses, which have to be counted.

Teach this as a positive result: knowing when not to use a recurrent network is part of the skill. The LSTM at 81.8 % and the vanilla RNN at 79.5 % show that even among RNNs the gap is large on a task like this.



## Slide 11
Slip detection is the synchronised many-to-many task, and it is where we settle the bidirectional question.

Left panel: one test run. Grey is the disagreement between the wheel yaw rate and the gyro; orange bands are true slip. The solid blue line is the causal LSTM's slip probability, the dashed purple line the bidirectional one. Both switch on at the start of each band and off at the end.

Right panel, per-step F1 on all 600 test runs: hand rule 0.665, causal LSTM 0.972, bidirectional 0.968. The rule flags a step when wheels and gyro disagree by more than 0.25 rad/s or the wheels read 0.12 m/s above the command. Its recall is only 0.578: a single threshold cannot adapt to speed and turning, and the network can.

The bidirectional model does not win, and it costs 25025 parameters against 12545 and twice the training time. That is because the evidence, wheels disagreeing with the gyro, is present at the moment of slip. The causal model's median onset latency is 0 steps; it missed 3 of 988 episodes.

Question for the room: invent a fault where the future would help. Answer: anything only recognisable by its aftermath, such as a bump identified by the recovery that follows. Exercise 5 builds exactly that.



## Slide 12
Close by walking the workflow strip left to right, because it is the order in which to make decisions on a new sequence problem.

Pick the shape from where the answers are needed. Ask what the state must hold: a running integral, an accumulating belief, or just the last few steps. Open the gates so the long path exists at all. Bucket or pack so batching does not quietly corrupt the final state. Check the gradient reach. And beat the physics baseline, or report honestly that you did not.

The seven rules repeat the measured evidence: the heading task where gating made the difference between 1.33 and 0.13 rad, the padding measurements, the bidirectional model that did not help, and the terrain task where five hand features tied the best network.

The bridge to Supplement B: every model here carried the past in a vector of 48 numbers. That was enough for one heading angle. Ask the room what happens when the robot must remember several places, or a whole lap of its route, to answer. A fixed-size state starts to run out, and that is precisely where attention helps. The notebook contains eight exercises; exercises 2, 5 and 8 are the best ones to assign.


# Lecture 11 Supplement 2 Slide Notes

## Slide 1
This is the second half of the robotics supplement pair, and it answers the question Supplement A ended on: what happens when a fixed-size hidden state is no longer enough?

The main chapter 16 lecture used the CNC cell, where the LSTM and the transformer tied on accuracy and the transformer won on speed. That was honest but it left students asking where transformers actually earn their reputation. This deck builds two robot tasks designed around that question.

Task one, loop closure: has the robot crossed this floor feature before in this run? Task two, forecasting: where will a patrol robot be over the next eight seconds, given that it repeats its route? Both need the model to look something up in the past, which is exactly what attention does.

The selected emphases for this pair were long-range memory, interpretability and multi-step forecasting, so each gets its own measured result. And we finish with a task where the LSTM is just as good, because knowing when not to use a transformer is part of the lesson.



## Slide 2
The job of this slide is to state the hypothesis we will test for the next ten slides.

Top row: an LSTM reading a run. The two aqua boxes are things the robot will later need, say two floor features it crossed. The orange box at the end is the question. Everything about those aqua boxes has to survive inside the orange arrows, a vector of 96 numbers, through every intermediate step, and the network has to decide to store them before it knows they will matter.

Bottom row: self-attention. The query at the end draws a direct arrow to every earlier token. The thick blue arrows go to the tokens that resemble it. Nothing had to be compressed in advance, and the distance is always one step.

The two limits in the callout are separate, and we will be able to tell them apart in the experiment. Capacity: how many things fit in the state. Path length: how far they must travel. The price of attention is that every query is compared with every key, which is T squared work per head; we come back to that cost at the end.

Ask the room: in Supplement A, which limit did the heading task hit? Neither, really: heading is one number, and gated RNNs handled it well. So we need tasks with more to remember.



## Slide 3
Before any training, make attention concrete with four numbers per place.

Left panel: a robot has stored a four-number IMU fingerprint for each of four places, the grey and aqua bars. These are the keys, and here also the values. The orange bars on the right are the fingerprint it is feeling right now, the query: a slightly noisy copy of place 3.

Right panel: the dot product of the query with each key, divided by the square root of the dimension, through a softmax. Place 3 receives weight 0.83; the others share the rest. That is a lookup table where the address is the content itself, and where the answer is soft, so gradients flow through it.

The code box is the entire mechanism: three lines. The notebook checks the hand computation against PyTorch's scaled dot product attention kernel, which is what runs inside every transformer layer.

Two points students often miss. One: in a trained transformer the keys, queries and values are learned linear projections of the tokens, so the network learns which features to compare. Two: nothing here knows about order. Position has to be added separately, and we will see a case where leaving it out makes the model useless.



## Slide 4
This slide introduces the long-range memory task.

Top panel: the three accelerometer channels of one test run. Bottom: the gyro. The grey bands are the four floor features the rover crossed; the aqua band is the one that matches the query; the orange band at the end is the query itself, crossed 12.0 s after its match. Look carefully and you can see that the aqua burst and the orange burst have the same shape across channels, while the grey ones differ.

Why this design: in Supplement A every task could be solved by carrying a small summary forward. Here the model has to keep K continuous fingerprints without knowing which one will be asked about, then compare. That separates capacity from detection.

Both models share the same front end, a one-dimensional convolution with kernel 8 and stride 4, turning 128 samples into 31 tokens. The only difference is the layer that mixes tokens over time: a 96-unit LSTM with its forget gate opened, or two Pre-LN transformer blocks with four heads. That is the fair comparison.

Ask: what would a SLAM engineer do here without learning? Store a descriptor at every keyframe and run nearest-neighbour search at query time. Keep that answer in mind; attention will turn out to learn exactly that.



## Slide 5
This is the headline result for long-range memory.

Left panel, accuracy against the number of places per run. Orange is the LSTM: 0.901 at K equal 1, 0.716 at 2, 0.592 at 4, 0.552 at 8. Blue is the transformer: 0.993, 0.969, 0.946, 0.923. The dotted line is an oracle that is handed the clean fingerprints and uses the best nearest-neighbour threshold: 0.997 at K equal 1, 0.985 at 8. The transformer tracks the ceiling; the LSTM does not.

Right panel, the subtle point. At K equal 4 we split the positive test runs by how long before the query the matching feature was crossed and compute AUC per bin. If path length were the problem, the LSTM would be good at short gaps and bad at long ones. It is mediocre everywhere. So with 31 tokens, capacity is the binding limit: four 4-D fingerprints have to share one state.

Fairness questions students will raise. Training: both saw 20 epochs of 6000 fresh runs. Size: the LSTM has 61169 parameters, the transformer 41521. Exercise 3 gives the LSTM four times the hidden size and more epochs; it is worth assigning, so students see how much of the gap a bigger state closes.



## Slide 6
This slide teaches the interpretability lesson properly: an attention map is evidence, not proof.

Left panel, the example positive run from earlier. The grey line is the query token's attention in layer 1, the blue line in layer 2, plotted against time. Layer 2 puts a sharp peak exactly on the aqua band, the matching feature, and small bumps on the other features. Right panel, a negative run where the query is a new place: attention spreads over all the features, because none of them matches. In effect the model compares the query against every stored feature, which is what the nearest-neighbour SLAM engineer would do.

The numbers in the bullets are averaged over all positive test runs: 28 % of the last-layer attention on the match, 29 % shared by the other three features, 22 % on plain background and 21 % on the query's own tokens.

Now the occlusion test. If the model really used the matching feature, removing it should destroy the positive prediction. It does: the mean probability falls from 0.91 to 0.12, and 85 % of positives flip to "new place". Removing a different feature leaves the probability at 0.91.

Make the general point explicitly. Jain and Wallace showed in 2019 that attention weights can be changed without changing predictions. So the rule is: read the map, then test it. In the chapter 16 CNC lecture, the same test was needed to discover that attention and occlusion disagreed for some patches.



## Slide 7
Second task, chosen because forecasting is where long context matters most in robotics practice.

Left: the dead-reckoned path of one run. Grey is the two and a half laps of history the model sees; orange is the part it must forecast. The route is a random sequence of straights and turns, regenerated for every run, so a model cannot memorise it in its weights. Each lap also varies by plus or minus 4 percent in speed, and the measurements are noisy.

Right: the yaw rate. The yellow trace is what the model sees, the black trace the hidden future, and the arrow shows one lap of 60 steps. The dotted vertical lines mark points exactly one, two and three laps before the end of the forecast.

The point to make: a perfect forecaster for the next eight seconds only has to look at what happened one lap earlier. A transformer can do that directly, by attending from a future slot to the same phase of the previous lap. An LSTM has to have carried a whole lap of the route, 60 steps of a two-channel signal, in its state.

Why a direct forecaster with blank future slots rather than autoregressive rollout: it keeps the two models' inputs identical and avoids error feedback, so the comparison is clean. Exercise 7 asks students to build the autoregressive version.



## Slide 8
This is the forecasting result. Read the left panel by horizon, then the example on the right.

Left: root mean square error over 500 test runs as a function of how far ahead the forecast is. Grey dashed, persistence: hold the last value. It is the best at the first step, 0.067, and then rises to 0.441. Orange, the LSTM: 0.108 at the first step and about 0.289 from one second onwards. Blue, the transformer: 0.071, 0.039, 0.043, essentially flat. Aqua dotted, a reference that is told the lap period and copies the previous lap: about 0.050.

The transformer beating copy-one-lap is worth a sentence. It sees two earlier laps and averages them, which cancels part of the measurement noise. It found a better rule than the obvious one.

Right: one run. The LSTM, dashed orange, produces a smooth curve that hugs the mean and misses every turn; the transformer, blue, follows the turns.

Both models trained 30 epochs on 2000 runs. The LSTM has 113474 parameters, the transformer 50594; training took 180 s and 218 s on two CPU cores.



## Slide 9
This slide closes the interpretability theme with the most striking picture in the deck.

Left: layer 1's attention for one test run. Rows are the future slots being forecast, columns are every position the model could look at. The orange dashed diagonals mark exactly one lap and two laps back. The bright stripes sit on them: each future slot looks back at the same phase of the previous two laps.

Right: the same thing averaged over 200 runs and plotted against lag. Layer 1, blue, has peaks at one and two laps: 24 % of its mass within two steps of one lap and 20 % near two laps. Layer 2, grey, does something else: 40 % of its attention goes to the other future slots, smoothing the forecast across the horizon. So the division of labour is: layer 1 fetches the right phase of earlier laps, layer 2 cleans up. The model was never given the period; it discovered the copy-the-previous-lap rule from the loss.

The second bullet is the positional-encoding lesson in its simplest form. In this design the future slots all receive the same input. Self-attention is permutation-equivariant, so without a positional embedding those slots cannot be told apart and every horizon gets the same forecast. The notebook measures the spread: exactly 0. Position is part of the input, not a detail.

Good discussion question: what would change if the patrol timer varied from lap to lap? Then a fixed positional offset would not work, and the model would have to find the matching phase by content. Exercise 2.



## Slide 10
Finish the technical part with the honest boundary of the result.

Left panel, accuracy side by side on three tasks. Terrain classification from Supplement A, against the best recurrent model there, the GRU: 0.963 against 0.970. One caveat to say out loud: Supplement A's LSTM only reached 0.818 on this task, so a transformer would look like a big win if you compared it with the wrong recurrent baseline. Always compare with the best one. Loop closure with one place: 0.901 against 0.993. Loop closure with 4 places: 0.592 against 0.946. The pattern is the lesson: the advantage grows with how much has to be looked up in the past, and nearly vanishes when the answer is a running summary such as vibration statistics.

Right panel, the cost. The work in an attention layer grows with the square of the sequence length; an LSTM's grows linearly but cannot be parallelised over time. The vertical lines mark where our two tasks sit: 31 tokens for loop closure after the convolutional stem, 190 for forecasting. At those lengths T squared is cheap. At ten thousand raw samples it is not, which is why patching and convolutional stems exist; the CNC lecture covered patch size in detail.

Practical rule for students: if you can name a summary statistic that answers the question, start with a GRU or even hand features. If the answer needs a lookup of something specific from the past, or several things, that is where attention is worth T squared.



## Slide 11
Walk the workflow strip as a decision procedure for a new sequence problem.

First, name what must be remembered, and count it: how many things, how far back. That one step predicts most of what we saw. Supplement A's heading was one number and gated RNNs were excellent. Loop closure with several places, and a whole lap of a route, were too much for a fixed state.

Then tokenise, so the sequence is hundreds of tokens rather than thousands of samples. Attend, with positions. Check the attention with occlusion before you explain anything with it. And pay the quadratic cost only where it wins.

The seven rules restate the measured evidence: 94.6 % against 59.2 % on loop closure, a 6.7-fold lower forecast error with the lap period discovered by attention, the occlusion test that confirmed the map, the flat forecast without positions, and the terrain task where the LSTM was as good for less.

For assignments, exercises 1, 2 and 3 in the notebook are the most instructive: run length versus capacity, a variable patrol timer, and giving the LSTM a fair chance with more capacity.




