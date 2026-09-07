# Fuzzy Fusion Extension — Simple Explanation

## 1. What was the original project?

The project recognizes two emotions:

- `0 = Sad`
- `1 = Happy`

It uses three types of data:

1. EEG signals
2. Audio
3. Video

Each modality has its own neural-network branch. Every branch produces a
128-dimensional feature representation.

Originally, the model joined all three representations:

```text
EEG features   (128)
Audio features (128)  -> Concatenate (384) -> MLP -> Sad/Happy
Video features (128)
```

This original model is still available and has not been replaced.

---

## 2. What problem are we trying to solve?

The quality of each modality can change from one sample to another.

For example:

- the audio may contain noise;
- the face may not be clearly visible;
- the EEG signal may contain interference;
- the three modalities may predict different emotions.

The original classifier does not directly show how much it trusts each
modality.

We added fuzzy logic to answer:

> For this particular sample, how much should we trust EEG, audio, and video?

---

## 3. What did we add?

We added four main parts:

1. One prediction head for each modality
2. Confidence calculation
3. Cross-modal disagreement calculation
4. A fuzzy inference system that produces reliability weights

The new flow is:

```text
EEG branch   -> EEG probability   -> EEG confidence
Audio branch -> Audio probability -> Audio confidence
Video branch -> Video probability -> Video confidence

Probabilities -> Cross-modal disagreement

Confidences + disagreement
            ↓
      Fuzzy inference
            ↓
EEG, Audio, Video reliability weights
            ↓
Weighted probability fusion
            ↓
        Sad or Happy
```

The neural-network branches remain unchanged.

---

## 4. Modality prediction heads

Previously, each branch returned only a 128-D representation.

We added a small linear classifier after each representation:

```text
128-D representation -> Linear layer -> 2 logits -> Softmax
```

This gives:

```text
[P(Sad), P(Happy)]
```

Example:

```text
EEG:   [0.90, 0.10]
Audio: [0.55, 0.45]
Video: [0.86, 0.14]
```

In this example, all modalities predict Sad, but audio is much less certain.

The three heads are trained while the existing EEG, audio, and video encoders
remain frozen.

---

## 5. Confidence

Confidence is the largest probability produced by a modality.

```python
confidence = max(probability)
```

Using the previous example:

```text
EEG confidence   = 0.90
Audio confidence = 0.55
Video confidence = 0.86
```

For two-class classification:

- confidence near `0.50` means uncertain;
- confidence near `1.00` means very certain.

Confidence does not always mean that a prediction is correct. A model can be
confident and wrong, so we test the system under noisy and conflicting inputs.

---

## 6. Cross-modal disagreement

Disagreement measures how different the three probability distributions are.

First, we compare each pair:

```text
EEG vs Audio
EEG vs Video
Audio vs Video
```

We use total-variation distance:

```text
TV(p, q) = 0.5 × sum of absolute probability differences
```

The final disagreement is the average of the three pairwise distances:

```text
disagreement =
    (TV(EEG, Audio) + TV(EEG, Video) + TV(Audio, Video)) / 3
```

Interpretation:

- `0` means the modalities completely agree;
- a larger value means stronger disagreement;
- the result is normalized between `0` and `1`.

---

## 7. Fuzzy membership levels

The fuzzy system converts exact numbers into simple linguistic levels:

```text
Low
Medium
High
```

We use these levels for:

- EEG confidence
- Audio confidence
- Video confidence
- Disagreement
- Each output reliability

The membership functions overlap. A confidence value can be partly Medium and
partly High. This creates a smooth transition instead of a hard cutoff.

For example:

```text
Confidence = 0.80

may belong partly to:
Medium confidence
and
High confidence
```

Triangular and trapezoidal membership functions are used.

---

## 8. Fuzzy rules

The rules describe how confidence and disagreement affect reliability.

Example 1:

```text
IF EEG confidence is HIGH
AND Audio confidence is LOW
AND Video confidence is HIGH
THEN EEG reliability is HIGH
AND Audio reliability is LOW
AND Video reliability is HIGH
```

Example 2:

```text
IF EEG confidence is LOW
AND Audio confidence is HIGH
AND Video confidence is HIGH
THEN EEG reliability is LOW
AND Audio reliability is HIGH
AND Video reliability is HIGH
```

Example 3:

```text
IF disagreement is HIGH
AND one modality is clearly more confident
THEN preserve the most confident modality
AND reduce the alternatives
```

The system contains 81 rules:

```text
3 EEG levels
× 3 Audio levels
× 3 Video levels
× 3 Disagreement levels
= 81 rules
```

Therefore, every possible linguistic input combination is covered.

---

## 9. Mamdani fuzzy inference

We use Mamdani fuzzy inference.

In simple terms, it performs these steps:

1. Convert confidence and disagreement values into fuzzy levels.
2. Check how strongly each rule applies.
3. Combine the outputs of all active rules.
4. Convert each fuzzy output into one numerical reliability value.

The final conversion uses centroid defuzzification, which finds the center of
the combined output membership area.

This produces three values:

```text
EEG reliability
Audio reliability
Video reliability
```

---

## 10. Weight normalization

The three reliability values are normalized so they add up to one.

```text
sum = EEG reliability + Audio reliability + Video reliability

EEG weight   = EEG reliability / sum
Audio weight = Audio reliability / sum
Video weight = Video reliability / sum
```

Example:

```text
EEG weight   = 0.43
Audio weight = 0.14
Video weight = 0.43
```

The low-confidence audio prediction now has less influence.

---

## 11. Final emotion prediction

The final probability is a weighted combination:

```text
Final probability =
    EEG weight   × EEG probability
  + Audio weight × Audio probability
  + Video weight × Video probability
```

The class with the largest final probability is selected.

Example:

```text
Final probability = [0.87, 0.13]
Prediction = Sad
```

---

## 12. Why use fuzzy logic?

Fuzzy logic provides:

- different weights for different samples;
- understandable Low/Medium/High reasoning;
- an explainable rule base;
- smooth decisions instead of hard thresholds;
- no change to the existing expert architecture;
- visible reliability weights for EEG, audio, and video.

It is especially useful to investigate situations where one modality becomes
uncertain or disagrees with the others.

---

## 13. What methods are compared?

We compare seven approaches:

1. Original multimodal MLP
2. EEG only
3. Audio only
4. Video only
5. Equal average of the three probabilities
6. Fixed weighted probability fusion
7. Fuzzy adaptive probability fusion

This comparison is necessary because fuzzy fusion should not be considered
better without experimental evidence.

---

## 14. How are the models trained?

### Step 1: Train the original model

The original three-branch network is trained using cross-entropy loss.

### Step 2: Freeze the existing model

The EEG, audio, and video expert branches are not changed.

### Step 3: Train the modality heads

Only the three small `128 -> 2` classifiers are trained.

### Step 4: Use validation data

Validation data is used to:

- select the best heads;
- inspect confidence;
- calculate fixed baseline weights.

### Step 5: Evaluate once on test data

The test set is kept separate until final evaluation.

---

## 15. Evaluation

For every method, we calculate:

- accuracy;
- precision;
- recall;
- F1-score;
- ROC-AUC;
- per-class performance;
- confusion matrix.

For fuzzy fusion, we also save:

- EEG, audio, and video confidence;
- disagreement;
- final fuzzy weights;
- examples showing why a modality received a lower weight.

---

## 16. Robustness experiments

We test the system under controlled problems:

```text
Normal input
Noisy audio
Degraded video
Noisy EEG
Dropped audio
Dropped video
Dropped EEG
Conflicting audio prediction
```

We observe:

1. Does the affected modality's confidence decrease?
2. Does its fuzzy weight decrease?
3. Does fuzzy fusion preserve accuracy better than simple fusion?

We report the measured result. We do not assume that fuzzy fusion will always
improve performance.

---

## 17. Important limitation

The system uses one overall disagreement value.

This value can tell us:

```text
"The modalities disagree."
```

But when equally confident modalities conflict, it may not tell us:

```text
"This exact modality is wrong."
```

A future version could add signal-quality measurements or each modality's
distance from the group consensus. These additions are not included in the
initial four-input fuzzy system.

---

## 18. Files added for fuzzy fusion

```text
src/models/modality_heads.py
    Independent EEG, audio, and video classifiers

src/models/fuzzy_fusion.py
    Weighted probability fusion

src/fuzzy/membership_functions.py
    Low, Medium, and High membership functions

src/fuzzy/fuzzy_rules.py
    Complete 81-rule base

src/fuzzy/fuzzy_inference.py
    Mamdani inference and centroid defuzzification

src/scripts/train_modality_heads.py
    Trains only the new modality heads

src/scripts/evaluate_fuzzy_fusion.py
    Baseline comparison, fuzzy evaluation, plots, and robustness tests

tests/test_fuzzy_inference.py
    Tests rule coverage, disagreement, and fuzzy-weight correctness
```

Small methods were also added to the original model to expose the existing
128-D features. Its original `forward()` behavior and pretrained checkpoint
format remain unchanged.

---

## 19. Commands

Train the original model:

```bash
python3 src/scripts/train_multimodal.py --data-root data
```

Train the modality heads:

```bash
python3 src/scripts/train_modality_heads.py --data-root data
```

Evaluate all methods:

```bash
python3 src/scripts/evaluate_fuzzy_fusion.py --data-root data
```

Run robustness experiments:

```bash
python3 src/scripts/evaluate_fuzzy_fusion.py \
  --data-root data \
  --robustness
```

---

## 20. Short explanation for a presentation

> Our original model has separate EEG, audio, and video experts. It combines
> their 128-dimensional features and predicts Sad or Happy. We kept this model
> unchanged as our baseline.
>
> We then added one small classifier to each expert so that every modality
> produces its own Sad/Happy probability and confidence. We also calculate how
> much the three probability distributions disagree.
>
> These four values—three confidences and one disagreement value—enter a
> Mamdani fuzzy inference system. The fuzzy rules describe the inputs as Low,
> Medium, or High and produce a reliability value for each modality. Centroid
> defuzzification converts these fuzzy outputs into numerical weights.
>
> Finally, we normalize the weights and combine the three probability
> distributions. This allows the model to trust different modalities for
> different samples. We compare it with the original model, individual
> modalities, equal averaging, and fixed weighting. We also test noisy,
> missing, and conflicting modalities to determine whether fuzzy fusion
> actually helps.

---

## 21. One-sentence summary

The fuzzy extension keeps the original neural experts unchanged and adds an
explainable, sample-specific decision layer that weights EEG, audio, and video
probabilities according to confidence and cross-modal disagreement.
