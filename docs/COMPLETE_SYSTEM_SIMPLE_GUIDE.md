# Complete EAV System — Simple Explanation

## 1. What does this project do?

This project detects whether a person is:

- **Sad**
- **Happy**

It studies three signals recorded at the same time:

1. **EEG** — brain activity
2. **Audio** — the person's voice
3. **Video** — the person's facial expressions

Using three signals can be more informative than using only one. For example,
the face may be unclear while the voice and EEG are useful.

---

## 2. Complete system flow

```text
EEG data   -> EEG expert   -> EEG features   -> EEG prediction
Audio data -> Audio expert -> Audio features -> Audio prediction
Video data -> Video expert -> Video features -> Video prediction

                     All three predictions
                              |
                              v
                Confidence and disagreement
                              |
                              v
                    Fuzzy reliability system
                              |
                              v
                  EEG, Audio and Video weights
                              |
                              v
                    Final Sad/Happy prediction
```

The original neural-network classifier is also preserved so that it can be
compared fairly with fuzzy fusion.

---

## 3. Input data

Each example represents synchronized EEG, audio, and video.

| Input | Shape | Simple meaning |
|---|---|---|
| EEG | `30 × 500` | 30 brain-signal channels |
| Audio | `80000` | Five seconds of audio at 16 kHz |
| Video | `25 × 3 × 56 × 56` | 25 color face images |
| Label | One number | `0=Sad`, `1=Happy` |

The system checks that all three modalities have matching samples and labels.

---

## 4. EEG expert

The EEG branch analyzes how brain signals change over time.

```text
EEG
 -> Convolution layers
 -> Bidirectional LSTM
 -> Self-attention
 -> 128-number representation
```

- Convolution layers find local signal patterns.
- The bidirectional LSTM understands the sequence in both directions.
- Self-attention focuses on the most informative moments.

---

## 5. Audio expert

The audio branch analyzes speech and voice patterns.

```text
Audio waveform
 -> MelSpectrogram
 -> CNN
 -> Bidirectional LSTM
 -> Self-attention
 -> 128-number representation
```

- The MelSpectrogram converts sound into a time-frequency image.
- The CNN finds useful voice-frequency patterns.
- The LSTM studies how speech changes with time.
- Attention focuses on important audio moments.

---

## 6. Video expert

The video branch analyzes facial expressions over time.

```text
25 face frames
 -> CNN for every frame
 -> Bidirectional LSTM
 -> Self-attention
 -> 128-number representation
```

- The CNN extracts facial features from each frame.
- The LSTM studies how the expression changes.
- Attention focuses on the most useful frames.

---

## 7. Original multimodal model

Each expert produces 128 features. The original model joins them:

```text
128 EEG + 128 Audio + 128 Video = 384 features
```

These features enter an MLP classifier:

```text
384 -> 256 -> 64 -> 2 classes
```

The output is:

```text
[Sad score, Happy score]
```

This original model was not removed or redesigned. It remains the main
baseline.

---

## 8. Separate modality predictions

For fuzzy fusion, each expert needs its own prediction.

Three small classifiers were added:

```text
EEG 128 features   -> EEG Sad/Happy probability
Audio 128 features -> Audio Sad/Happy probability
Video 128 features -> Video Sad/Happy probability
```

Example:

```text
EEG:   [0.90 Sad, 0.10 Happy]
Audio: [0.55 Sad, 0.45 Happy]
Video: [0.87 Sad, 0.13 Happy]
```

The original experts are frozen while these small classifiers are trained.

---

## 9. Confidence

Confidence is the largest probability from one modality.

```text
confidence = maximum probability
```

For the previous example:

```text
EEG confidence   = 0.90
Audio confidence = 0.55
Video confidence = 0.87
```

A value near `0.50` means uncertain. A value near `1.00` means very certain.

---

## 10. Disagreement

The system also checks whether the three experts agree.

It compares:

```text
EEG with Audio
EEG with Video
Audio with Video
```

The average difference becomes one disagreement value:

- low disagreement means similar predictions;
- high disagreement means conflicting predictions.

---

## 11. Fuzzy reliability system

The fuzzy system receives four values:

1. EEG confidence
2. Audio confidence
3. Video confidence
4. Disagreement

It describes each value as:

```text
Low, Medium or High
```

It then applies understandable rules.

Example:

```text
IF EEG confidence is HIGH
AND Audio confidence is LOW
AND Video confidence is HIGH
THEN EEG reliability is HIGH
AND Audio reliability is LOW
AND Video reliability is HIGH
```

The complete system has 81 rules, covering every combination of the four
fuzzy inputs.

---

## 12. Final reliability weights

Mamdani fuzzy inference combines all active rules. Centroid defuzzification
converts the fuzzy outputs into numbers.

The numbers are normalized so they add to one.

Example:

```text
EEG weight   = 0.43
Audio weight = 0.14
Video weight = 0.43
Total        = 1.00
```

Here, uncertain audio has less influence.

---

## 13. Final prediction

The system multiplies each probability by its reliability weight:

```text
Final probability =
    EEG weight   × EEG probability
  + Audio weight × Audio probability
  + Video weight × Video probability
```

The class with the larger final probability becomes the answer.

---

## 14. Training process

### Phase 1

Train the original EEG-Audio-Video network.

### Phase 2

Expose the three existing 128-feature representations.

### Phase 3

Freeze the original experts and train only the three small modality
classifiers.

### Phase 4

Use validation data to inspect confidence and select the best heads.

### Phase 5

Apply the fixed fuzzy system during evaluation.

### Phase 6

Compare fuzzy fusion with all baseline methods.

The test set is not used for selecting models, rules, or weights.

---

## 15. Methods compared

The project evaluates:

1. Original multimodal neural classifier
2. EEG-only classifier
3. Audio-only classifier
4. Video-only classifier
5. Equal average fusion
6. Fixed weighted fusion
7. Fuzzy adaptive fusion

This comparison shows whether fuzzy fusion actually provides an advantage.

---

## 16. Evaluation results produced

The evaluation script calculates:

- accuracy;
- precision;
- recall;
- F1-score;
- ROC-AUC;
- Sad and Happy class performance;
- confusion matrices;
- fuzzy reliability weights for example samples.

Results are saved as JSON, CSV, and images.

---

## 17. Robustness testing

The project can test difficult situations:

- noisy audio;
- low-quality video;
- noisy EEG;
- missing audio;
- missing video;
- missing EEG;
- a conflicting audio prediction.

For each condition, it checks:

- whether the affected modality receives less weight;
- whether final accuracy changes;
- whether fuzzy fusion performs better than simpler methods.

The project reports the measured outcome instead of automatically claiming
that fuzzy fusion is better.

---

## 18. Data safety and fairness

The data is divided into:

```text
Training data
Validation data
Test data
```

- Training data teaches the neural networks.
- Validation data selects models and fixed weights.
- Test data is used only for final evaluation.

The original saved checkpoint was created by an older script that used test
data during model selection. A final research result should therefore retrain
the original model using the corrected split.

---

## 19. Important limitation

The fuzzy system knows when the modalities disagree, but one overall
disagreement value cannot always identify which equally confident modality is
wrong.

Also, high confidence does not guarantee correctness. A neural network can be
confident on noisy input.

This is why robustness experiments and real test results are necessary.

---

## 20. How to run the system

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the original model:

```bash
python3 src/scripts/train_multimodal.py --data-root data
```

Train the three modality heads:

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

## 21. Simple presentation explanation

> Our system detects Sad and Happy emotions using synchronized EEG, audio, and
> video. Each modality has its own neural expert made from convolution,
> bidirectional LSTM, and attention layers. Each expert produces 128 features.
>
> The original model joins all 384 features and uses an MLP classifier. We keep
> this model unchanged as our baseline.
>
> We also add a small classifier to each expert so EEG, audio, and video can
> produce separate emotion probabilities. From these probabilities, we
> calculate three confidence values and one disagreement value.
>
> A Mamdani fuzzy system uses 81 understandable Low, Medium, and High rules to
> produce a reliability weight for each modality. The weights are normalized
> and used to combine the three probabilities.
>
> This makes fusion adaptive for each sample. If one modality is uncertain, it
> can receive less influence. We compare this approach with the original model,
> individual modalities, averaging, and fixed weights. We also test noisy,
> missing, and conflicting inputs before deciding whether fuzzy fusion improves
> the system.

---

## 22. One-line summary

The complete system uses three neural experts to understand EEG, voice, and
facial expressions, then uses explainable fuzzy logic to decide how much each
expert should influence the final Sad or Happy prediction.
