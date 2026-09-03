# DBSCAN supplementary notebooks — CNC / Robot / Wind (generated 2026-08-30)
 
Three standalone companions to `DBSCAN_condition_monitoring.ipynb`. Each is executed
end-to-end with outputs and figures embedded, and each teaches a **different** lesson so they
can be assigned individually or as a set. All data is synthetic, seeded, and offline.
 
## Supplement 1 — `Supplement_1_CNC_Spindle.ipynb`
**Lesson: the features are the model.**
Milling on a 3-axis machining centre. Raw 20 kHz accelerometer windows (NS=4096) are
synthesised from milling physics, then reduced to six condition-monitoring features
(RMS, crest, kurtosis, spectral centroid, non-harmonic energy ratio, spindle load).
 
| result | value |
|---|---|
| params | StandardScaler, MinPts 12, eps 1.1 |
| clusters | 3 cutting regimes, ARI 1.000 |
| anomalies | recall 1.000, precision 0.889 (45 alarms, 40 real) |
| silhouette | 0.375 — deliberately discussed: elongated clusters, do not tune eps to it |
| stability plateau | eps 0.7–1.4 all give the same answer |
| contrast ratio | raw waveform 1.12 → feature space 1.91 |
 
Key design point: chatter is modelled as **incipient** (tone at a structural mode ~1.75 kHz,
never on a tooth harmonic; RMS stays inside the healthy envelope). This makes the ablation
honest — dropping both frequency-domain features takes chatter recall to 0.00.
 
## Supplement 2 — `Supplement_2_Robot_Joint_Torques.ipynb`
**Lesson: your feature families determine which faults are detectable.**
6-axis pick-and-place robot, 502 cycles, three payload programs (1.5/5.0/9.0 kg). Rigid-body
torque model; 18 features = 6 joints × {carry_mean, peak, carry_std}.
 
| result | value |
|---|---|
| params | StandardScaler → PCA(4), MinPts 12, eps 1.1 |
| clusters | 3 programs, ARI 1.000 |
| anomalies | recall 1.000, precision 0.894 (47 alarms) |
| dead features | 9 of 18 (J1/J4/J6 are not gravity-loaded) — shown via between/within variance ratio |
| raw 18-D vs PCA-4 | 0.84 vs 0.89 precision at recall 1.00 |
 
Ablation at equal alarm budget: drop `carry_std` → dropped-part recall 1.00→0.25; drop `peak`
→ collision 1.00→0.64; drop `carry_mean` → all faults caught but only **one** program found.
Also contains a worked warning: cosine distance reports ARI 1.000 while false-alarming on 42 %
of part B and turning belt wear into a "normal" cluster — read the crosstab, not the score.
 
## Supplement 3 — `Supplement_3_Wind_Turbine_SCADA.ipynb`
**Lesson: varying density is a signal that you are clustering in the wrong space.**
2 MW turbine, 1400 ten-minute SCADA records, Weibull wind. Contains blade icing (fault),
anemometer fault (sensor fault) and grid curtailment (legitimate mode).
 
- **The failure:** local density falls 9.8× from 4–6 m/s to 16–20 m/s. No eps works —
  eps 0.10 catches 94 % of icing but flags 67 healthy high-wind records; eps 0.30 flags 15 and
  catches none.
- **HDBSCAN does not rescue it:** at a matched alarm budget it catches *fewer* icing events
  (0.88 vs 0.94). The healthy data is a 1-D curve, not blobs.
- **The fix:** cluster the residual against the manufacturer power curve —
  `power_deficit`, `rpm_ratio`, `pitch_dev`. Plain DBSCAN (eps 0.10, MinPts 10) then gives
  **4 clusters, silhouette 0.936**, all 34 icing events as noise, zero normal-production false
  alarms. Robust to a ±10 % error in the reference curve.
- **Curtailment and the anemometer fault each become their own cluster** — the notebook's
  closing point: read the cluster inventory, not just `labels_ == -1`.
## Provenance / regeneration
 
Built by `build_cnc.py`, `build_robot.py`, `build_wind.py` (nbformat) with a shared
`nbkit.py` helper; those generator scripts live only in the session workspace, not here.
Every prose claim was fact-checked against executed output by an independent reviewer pass;
13 mismatches were found and corrected before delivery.