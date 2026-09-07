# EAV Emotion Recognition — Complete Project Documentation

## 1. Project Overview

EAV is a multimodal deep-learning project for binary emotion recognition from
three synchronized signals:

- **EEG:** electrical brain activity from 30 channels
- **Audio:** a five-second speech waveform
- **Video:** 25 facial frames sampled over the same interval

The model predicts:

| Model label | Emotion | Original EAV dataset label |
|---:|---|---:|
| `0` | Sad | `1` |
| `1` | Happy | `3` |

The central research question is:

> Can a transparent fuzzy decision layer improve multimodal emotion
> recognition by trusting reliable modalities and reducing the influence of
> uncertain or conflicting modalities on each sample?

The fuzzy system does **not** replace the neural network. The existing EEG,
audio, and video expert branches remain unchanged. It is an additional
decision-level fusion method evaluated against the original model and simpler
fusion baselines.

No performance improvement is assumed. Improvement must be demonstrated using
held-out test results and controlled robustness experiments.

---

## 2. System Objectives

The project has two complementary objectives:

1. Learn useful emotion representations independently from EEG, audio, and
   video.
2. Combine the three expert decisions adaptively and explainably.

The original system performs learned late fusion by concatenating the three
representations. The extension also supports probability-level fusion:

```text
Synchronized sample
├── EEG expert   ──> 128-D representation ──> EEG probability
├── Audio expert ──> 128-D representation ──> Audio probability
└── Video expert ──> 128-D representation ──> Video probability
                                                │
                                                ▼
                              confidence + disagreement
                                                │
                                                ▼
                                Mamdani fuzzy inference
                                                │
                                                ▼
                              EEG/Audio/Video reliability
                                                │
                                                ▼
                               weighted final probability
                                                │
                                                ▼
                                      Sad or Happy
```

---

## 3. Repository Organization

```text
EAV-Emotion-Recognition/
├── best_multimodal_model.pth
├── requirements.txt
├── README.md
├── docs/
│   ├── PROJECT_DOCUMENTATION.md
│   ├── Model_Topology.txt
│   ├── README.md
│   └── images/
├── pretrained/
│   └── ast-finetuned-audioset/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── multimodal_dataset.py
│   │   └── splits.py
│   ├── fuzzy/
│   │   ├── __init__.py
│   │   ├── membership_functions.py
│   │   ├── fuzzy_rules.py
│   │   └── fuzzy_inference.py
│   ├── models/
│   │   ├── multimodal_model.py
│   │   ├── modality_heads.py
│   │   ├── fuzzy_fusion.py
│   │   ├── CNN_torch/
│   │   └── Transformer_torch/
│   └── scripts/
│       ├── experiment_utils.py
│       ├── train_multimodal.py
│       ├── train_modality_heads.py
│       ├── evaluate_model.py
│       ├── evaluate_fuzzy_fusion.py
│       └── print_architecture.py
└── tests/
    └── test_fuzzy_inference.py
```

### Active and legacy code

The active binary multimodal pipeline uses:

- `src/models/multimodal_model.py`
- `src/models/modality_heads.py`
- `src/models/fuzzy_fusion.py`
- `src/data/`
- `src/fuzzy/`
- `src/scripts/`

The `CNN_torch/` and `Transformer_torch/` directories contain older
single-modality experiments from the broader EAV project. They are retained
for reference but are not part of the active fuzzy multimodal pipeline.

---

## 4. Dataset and Preprocessing Contract

The original EAV dataset contains 42 participants and five emotions. This
project filters the synchronized data to Sadness and Happiness.

### Expected model inputs

| Modality | Per-sample tensor | Meaning |
|---|---|---|
| EEG | `(30, 500)` | 30 channels, 500 temporal values |
| Audio | `(80000,)` | five seconds at 16 kHz |
| Video | `(25, 3, 56, 56)` | 25 RGB face frames |
| Label | scalar integer | `0=Sad`, `1=Happy` |

After batching:

```text
EEG:    (batch, 30, 500)
Audio:  (batch, 80000)
Video:  (batch, 25, 3, 56, 56)
Labels: (batch,)
```

### Supported preprocessed files

The preferred synchronized format is:

```text
data/Inputs/Multimodal/subject_01.npz
```

Each NPZ must contain:

```text
eeg
audio
video
labels
```

Separate modality pickles are also supported:

```text
data/Inputs/EEG/subject_01_eeg.pkl
data/Inputs/Audio/subject_01_aud.pkl
data/Inputs/Vision/subject_01_vis.pkl
```

A pickle may contain either:

```python
(data, labels)
```

or:

```python
(train_data, train_labels, test_data, test_labels)
```

The loader validates that:

- all modalities contain the same number of samples;
- labels align exactly across modalities;
- tensors match the expected architecture;
- no unsupported normalization or silent dataset conversion occurs.

The loader can transpose common stored formats:

- EEG `(N, 500, 30)` to `(N, 30, 500)`
- Video `(N, 25, 56, 56, 3)` to `(N, 25, 3, 56, 56)`
- Audio `(N, 80000, 1)` to `(N, 80000)`

It does not recreate missing raw EEG filtering, audio segmentation, or face
detection. Those preprocessing choices affect checkpoint compatibility and
must match the data used to train the original model.

---

## 5. Data Splitting and Leakage Prevention

Splitting is performed independently for each subject and class.

1. The final 30% of each subject/class remains the historical test partition.
2. The first 70% is the development partition.
3. The development partition is deterministically shuffled.
4. Twenty percent of development is used for validation.
5. The remainder is used for training.

In approximate percentages:

```text
Training:   56%
Validation: 14%
Test:       30%
```

The random seed is isolated by subject and class. Adding or removing one
subject does not change another subject's split.

### Test-set policy

The test set must not be used for:

- checkpoint selection;
- fuzzy membership tuning;
- rule design based on observed test behavior;
- confidence calibration;
- fixed fusion-weight selection;
- hyperparameter selection.

The original training script selected its best checkpoint using the same 30%
partition later treated as test data. Therefore, the supplied historical
`best_multimodal_model.pth` is useful for compatibility and demonstration, but
it does not represent a fully leakage-free final experiment. A clean research
report should retrain the baseline using the updated training script.

---

## 6. Original Neural Architecture

The active model class is:

```text
EAVMultimodalFusionNetwork
```

It contains three expert branches and one original late-fusion classifier.

### 6.1 EEG expert

Input:

```text
(B, 30, 500)
```

Processing:

```text
Conv1D: 30 -> 32, kernel 5
ReLU
MaxPool1D: temporal length 500 -> 250
Conv1D: 32 -> 64, kernel 3
ReLU
MaxPool1D: temporal length 250 -> 125
Permute to sequence: (B, 125, 64)
Bidirectional LSTM: hidden size 64 per direction
Self-attention over 125 time steps
Linear: 128 -> 128
ReLU
```

Output:

```text
EEG representation: (B, 128)
```

The convolutional layers extract local temporal patterns. The bidirectional
LSTM models past and future context within the segment. Self-attention forms a
weighted temporal summary instead of treating every time step equally.

### 6.2 Audio expert

Input:

```text
(B, 80000)
```

Processing:

```text
MelSpectrogram:
  sample rate = 16000
  FFT size = 1024
  hop length = 512
  mel bins = 64
Log transform
Conv2D: 1 -> 16 + BatchNorm + ReLU + MaxPool
Conv2D: 16 -> 32 + BatchNorm + ReLU + MaxPool
Conv2D: 32 -> 64 + BatchNorm + ReLU
Adaptive frequency pooling
Reshape to a temporal sequence
Bidirectional LSTM: hidden size 32 per direction
Self-attention
Linear: 64 -> 128
ReLU
```

Output:

```text
Audio representation: (B, 128)
```

The MelSpectrogram converts raw speech into a time-frequency representation.
The CNN learns local spectral patterns, while the LSTM and attention layers
model how those patterns evolve.

On Apple Silicon, FFT operations used by the spectrogram may not be supported
natively by MPS. The existing architecture computes only the spectrogram on
CPU and moves it back to the selected device. This behavior is preserved.

### 6.3 Video expert

Input:

```text
(B, 25, 3, 56, 56)
```

Processing:

```text
Flatten batch and frame dimensions
Per-frame Conv2D: 3 -> 16 + ReLU + MaxPool
Per-frame Conv2D: 16 -> 32 + ReLU + MaxPool
Per-frame Conv2D: 32 -> 64 + ReLU
Adaptive spatial pooling to 1x1
Restore frame sequence: (B, 25, 64)
Bidirectional LSTM: hidden size 64 per direction
Self-attention over frames
Linear: 128 -> 128
ReLU
```

Output:

```text
Video representation: (B, 128)
```

The CNN extracts facial information from each frame. The bidirectional LSTM
models the expression sequence, and attention emphasizes informative frames.

### 6.4 Original late-fusion classifier

The three representations are concatenated:

```text
128 + 128 + 128 = 384 dimensions
```

Classifier:

```text
Linear: 384 -> 256
BatchNorm
ReLU
Dropout: 0.5
Linear: 256 -> 64
BatchNorm
ReLU
Dropout: 0.3
Linear: 64 -> 2
```

This produces logits for Sad and Happy. Training uses cross-entropy loss.

---

## 7. Backward-Compatible Feature Exposure

The original `forward()` behavior remains:

```python
logits = model(eeg, audio, video)
```

Additional methods expose existing representations without adding parameters
to the original model:

```python
eeg_features, audio_features, video_features = model.encode_modalities(
    eeg, audio, video
)
```

```python
logits, features = model.forward_with_features(eeg, audio, video)
```

The original checkpoint is a raw 85-key PyTorch state dictionary built for:

```text
num_classes = 2
embedding_dim = 128
```

No new trainable module was inserted into
`EAVMultimodalFusionNetwork`. Therefore, the existing checkpoint remains
strictly loadable.

---

## 8. Independent Modality Prediction Heads

Fuzzy fusion requires one probability distribution per modality. A separate
`ModalityPredictionHeads` module provides:

```text
EEG 128-D   -> Linear(128, 2) -> EEG logits
Audio 128-D -> Linear(128, 2) -> Audio logits
Video 128-D -> Linear(128, 2) -> Video logits
```

Softmax converts each output to:

```text
[P(Sad), P(Happy)]
```

The heads are stored separately in `modality_heads.pth`. This avoids changing
the original checkpoint schema.

### Head-training strategy

During head training:

- the three expert encoders are frozen;
- the original multimodal classifier is frozen;
- the model remains in evaluation mode;
- only the three linear heads receive gradients;
- each head receives its own cross-entropy loss;
- the three losses are averaged;
- the best heads are selected using mean validation accuracy.

The head checkpoint also records:

- validation accuracy per modality;
- mean validation confidence per modality;
- 10-bin expected calibration error per modality;
- validation-derived fixed fusion weights;
- seed and label mapping;
- source baseline checkpoint path.

---

## 9. Fuzzy Inputs

The fuzzy inference system has four scalar inputs.

### 9.1 Modality confidence

For a modality probability distribution \(p_m\):

\[
C_m = \max_c p_m(c)
\]

For binary classification:

\[
C_m \in [0.5, 1].
\]

The three confidence inputs are:

```text
EEG confidence
Audio confidence
Video confidence
```

A confidence near `0.5` means the head is uncertain. A confidence near `1.0`
means its probability is strongly concentrated on one class.

Confidence is not guaranteed to equal correctness. Neural networks can be
confident and wrong, which is why validation confidence diagnostics and
robustness experiments are important.

### 9.2 Cross-modal disagreement

Disagreement is based on total-variation distance.

For two class distributions \(p\) and \(q\):

\[
TV(p,q)=\frac{1}{2}\sum_c |p(c)-q(c)|.
\]

The three-expert disagreement is:

\[
D =
\frac{
TV(p_{eeg},p_{audio}) +
TV(p_{eeg},p_{video}) +
TV(p_{audio},p_{video})
}{3}.
\]

Properties:

- `D = 0` when all distributions are identical;
- larger values indicate stronger cross-modal conflict;
- `D` is normalized to `[0,1]`;
- the complete 384-D latent vector never enters the fuzzy system.

---

## 10. Membership Functions

Each input uses the linguistic labels:

```text
Low
Medium
High
```

### 10.1 Confidence partition

Confidence is defined on `[0.5, 1.0]`.

| Label | Type | Breakpoints |
|---|---|---|
| Low | Trapezoidal | `(0.50, 0.50, 0.625, 0.75)` |
| Medium | Triangular | `(0.50, 0.75, 1.00)` |
| High | Trapezoidal | `(0.75, 0.875, 1.00, 1.00)` |

These sets overlap, allowing gradual transitions rather than abrupt
thresholds.

### 10.2 Disagreement partition

Disagreement is defined on `[0,1]`.

| Label | Type | Breakpoints |
|---|---|---|
| Low | Trapezoidal | `(0.00, 0.00, 0.25, 0.50)` |
| Medium | Triangular | `(0.25, 0.50, 0.75)` |
| High | Trapezoidal | `(0.50, 0.75, 1.00, 1.00)` |

### 10.3 Reliability-output partition

Each output weight uses the same `[0,1]` Low/Medium/High partition as
disagreement.

The crisp outputs are not hard-coded weights. They are produced from the
aggregated fuzzy output areas through centroid defuzzification.

---

## 11. Fuzzy Rule Base

The rule base covers every combination of:

```text
3 EEG confidence levels
× 3 audio confidence levels
× 3 video confidence levels
× 3 disagreement levels
= 81 rules
```

This exhaustive construction prevents uncovered regions.

Example:

```text
IF EEG confidence is HIGH
AND Audio confidence is LOW
AND Video confidence is HIGH
AND disagreement is LOW
THEN EEG reliability is HIGH
AND Audio reliability is LOW
AND Video reliability is HIGH
```

### Rule policy

When disagreement is Low or Medium:

- each modality's reliability follows its confidence level.

When disagreement is High:

- a unique most-confident modality retains its reliability level;
- alternatives are reduced by one linguistic level;
- if the most-confident position is tied, tied leaders are reduced by one
  level because the available inputs cannot determine which tied expert is
  wrong.

Examples:

```text
Confidences: EEG=HIGH, Audio=LOW, Video=MEDIUM
Disagreement: HIGH
Outputs: EEG=HIGH, Audio=LOW, Video=LOW
```

```text
Confidences: EEG=HIGH, Audio=HIGH, Video=LOW
Disagreement: HIGH
Outputs: EEG=MEDIUM, Audio=MEDIUM, Video=LOW
```

### Important theoretical limitation

One global disagreement scalar says that the experts conflict, but does not
identify which expert is wrong. If two or three modalities are equally
confident, the four-input FIS cannot reliably isolate a confident outlier.

Solving that limitation would require additional inputs such as per-modality
distance from consensus, signal-quality indicators, entropy, or learned
reliability. Those are intentionally outside the initial four-input design.

---

## 12. Mamdani Fuzzy Inference

For each sample:

1. Fuzzify the four crisp inputs.
2. Evaluate all 81 rules.
3. Use minimum for logical AND.
4. Use minimum to clip each output membership function.
5. Use maximum to aggregate rule outputs.
6. Defuzzify each modality output using its centroid.

For aggregated membership function \(\mu_m(z)\), the centroid is:

\[
r_m =
\frac{\int_0^1 z\mu_m(z)\,dz}
{\int_0^1 \mu_m(z)\,dz}.
\]

The numerical universe contains 1,001 equally spaced points from `0` to `1`.

The implementation is deterministic. It does not use random inference,
learned fuzzy rules, or test-derived thresholds.

---

## 13. Reliability Normalization and Final Fusion

The three defuzzified reliabilities are normalized using one denominator:

\[
S = r_{eeg}+r_{audio}+r_{video}
\]

\[
w_{eeg}=\frac{r_{eeg}}{S},\quad
w_{audio}=\frac{r_{audio}}{S},\quad
w_{video}=\frac{r_{video}}{S}.\
\]

The final probability is:

\[
p_{final} =
w_{eeg}p_{eeg} +
w_{audio}p_{audio} +
w_{video}p_{video}.
\]

Final prediction:

\[
\hat y = \arg\max_c p_{final}(c).
\]

Because normalized weights sum to one, the fused output remains a valid
two-class probability distribution.

### Illustrative behavior

For probabilities with strong EEG/video confidence and weak audio confidence:

```text
EEG confidence:   0.93
Audio confidence: 0.54
Video confidence: 0.89

Example fuzzy weights:
EEG:   approximately 0.43
Audio: approximately 0.14
Video: approximately 0.43
```

These values illustrate the deterministic rule behavior and are not reported
test-set performance.

---

## 14. Compared Methods

The evaluation pipeline compares seven methods on exactly the same samples.

### 14.1 Original multimodal classifier

```text
384-D concatenated representation -> original MLP
```

This is the preserved learned late-fusion baseline.

### 14.2 EEG-only classifier

Uses only the EEG head probability.

### 14.3 Audio-only classifier

Uses only the audio head probability.

### 14.4 Video-only classifier

Uses only the video head probability.

### 14.5 Simple average fusion

\[
p = \frac{p_{eeg}+p_{audio}+p_{video}}{3}.
\]

### 14.6 Fixed weighted fusion

The validation accuracy of each modality is normalized:

\[
w_m =
\frac{validation\ accuracy_m}
{\sum_j validation\ accuracy_j}.
\]

The resulting weights are fixed for all test samples.

### 14.7 Fuzzy adaptive fusion

The reliability weights are recomputed for every sample from confidence and
disagreement.

---

## 15. Training Workflow

### Phase 1 — Train the original model

Train the unchanged three-branch late-fusion model:

```bash
python3 src/scripts/train_multimodal.py \
  --data-root data \
  --output best_multimodal_model.pth
```

Default training configuration:

```text
Loss: CrossEntropyLoss
Optimizer: Adam
Learning rate: 1e-4
Weight decay: 1e-5
Epochs: 20
Batch size: 8
Checkpoint criterion: validation accuracy
```

### Phase 2 — Expose frozen representations

No separate command is required. The model API exposes all three existing
128-D representations.

### Phase 3 — Train modality heads

```bash
python3 src/scripts/train_modality_heads.py \
  --data-root data \
  --baseline-checkpoint best_multimodal_model.pth \
  --output modality_heads.pth
```

Only the three `Linear(128,2)` heads are optimized.

### Phase 4 — Inspect validation confidence

Head training records validation confidence and expected calibration error in
the head checkpoint. No test samples are used.

### Phase 5 — Apply fuzzy inference

Fuzzy inference is applied after all modality probabilities have been
calculated. It does not retrain the expert branches.

### Phase 6 — Compare methods

```bash
python3 src/scripts/evaluate_fuzzy_fusion.py \
  --data-root data \
  --baseline-checkpoint best_multimodal_model.pth \
  --heads-checkpoint modality_heads.pth
```

---

## 16. Evaluation Metrics and Artifacts

For every method, the evaluation computes:

- accuracy;
- macro precision;
- macro recall;
- macro F1-score;
- binary ROC-AUC where both classes are present;
- precision, recall, F1, and support for Sad;
- precision, recall, F1, and support for Happy;
- confusion matrix.

Generated under `docs/fuzzy_results/`:

```text
metrics.json
metrics.csv
confusion_matrices.png
fuzzy_weight_examples.json
fuzzy_weight_examples.png
robustness_summary.png  # when --robustness is enabled
```

The example-weight report records:

- sample index;
- true label;
- each modality confidence;
- disagreement;
- each normalized fuzzy weight.

Actual generated metrics must be used when discussing performance. The
documentation does not claim fuzzy fusion is better before those results
exist.

---

## 17. Robustness Experiments

Enable all controlled conditions:

```bash
python3 src/scripts/evaluate_fuzzy_fusion.py \
  --data-root data \
  --baseline-checkpoint best_multimodal_model.pth \
  --heads-checkpoint modality_heads.pth \
  --robustness
```

### Conditions

| Condition | Modification |
|---|---|
| Normal | Unmodified input |
| Poor audio | Gaussian noise scaled by each sample's audio standard deviation |
| Poor video | Downsample frames to `14×14`, then restore to `56×56` |
| Poor EEG | Gaussian noise scaled by each sample's EEG standard deviation |
| Drop audio | Replace audio input with zeros |
| Drop video | Replace video input with zeros |
| Drop EEG | Replace EEG input with zeros |
| Audio disagreement | Reverse Audio's `[P(Sad), P(Happy)]` distribution |

All stochastic corruption uses a fixed seed.

The robustness report compares method accuracy and the mean fuzzy weight for
each modality under every condition.

### Correct interpretation

An input corruption does not necessarily produce lower softmax confidence.
For example, a neural head may be confidently wrong on zeroed or noisy input.
Because the initial fuzzy system uses confidence and global disagreement, a
reduced weight is an experimental outcome—not a guaranteed behavior.

The robustness results should answer:

```text
Did audio confidence and weight decrease under poor audio?
Did video confidence and weight decrease under poor video?
Did EEG confidence and weight decrease under poor EEG?
Did adaptive fusion preserve accuracy better than simple alternatives?
```

---

## 18. Device and Reproducibility Behavior

Automatic device selection:

```text
CUDA, if available
otherwise MPS, if available
otherwise CPU
```

`PYTORCH_ENABLE_MPS_FALLBACK=1` is enabled for unsupported Apple Silicon
operations.

Reproducibility controls include:

- Python random seed;
- NumPy seed;
- PyTorch seed;
- CUDA seeds where available;
- deterministic DataLoader generator;
- deterministic split generation;
- deterministic fuzzy rules and defuzzification;
- single-process data loading (`num_workers=0`).

Some low-level GPU operations can still vary by PyTorch version and hardware.

---

## 19. Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set the project root on the Python path when necessary:

```bash
export PYTHONPATH="."
```

Main dependencies include:

- PyTorch and torchaudio;
- NumPy and SciPy;
- scikit-learn;
- Matplotlib and Seaborn;
- audio/video packages used by the wider repository.

---

## 20. Commands Reference

Train the original baseline:

```bash
python3 src/scripts/train_multimodal.py --data-root data
```

Train frozen-encoder modality heads:

```bash
python3 src/scripts/train_modality_heads.py --data-root data
```

Evaluate only the original classifier:

```bash
python3 src/scripts/evaluate_model.py --data-root data
```

Evaluate all fusion methods:

```bash
python3 src/scripts/evaluate_fuzzy_fusion.py --data-root data
```

Evaluate robustness:

```bash
python3 src/scripts/evaluate_fuzzy_fusion.py --data-root data --robustness
```

Print the neural architecture:

```bash
python3 src/scripts/print_architecture.py
```

Run fuzzy unit tests:

```bash
python3 -m unittest tests/test_fuzzy_inference.py
```

---

## 21. Testing

The fuzzy tests verify:

1. the rule base contains all 81 combinations;
2. total-variation disagreement behaves correctly;
3. fuzzy weights are normalized and deterministic;
4. symmetric experts receive symmetric weights;
5. invalid probability distributions are rejected.

These tests do not require the EAV dataset or pretrained model.

Full neural integration testing requires:

- PyTorch and torchaudio;
- synchronized preprocessed data;
- `best_multimodal_model.pth`;
- a trained `modality_heads.pth`.

---

## 22. Known Limitations

### Missing dataset in the repository

The EAV data is intentionally not included. `data/` is ignored because the
dataset is large and may be subject to separate distribution terms.

### Historical checkpoint leakage

The supplied checkpoint originated from a workflow that selected epochs on the
historical test partition. Retraining is required for a clean final claim.

### Confidence is an imperfect reliability estimate

Maximum softmax probability can be miscalibrated and may remain high under
distribution shift.

### Global disagreement cannot locate every outlier

The FIS knows that experts disagree but cannot always know which equally
confident expert is wrong.

### Binary scope

The active checkpoint and fuzzy evaluation cover only Sad and Happy. Extending
to all five EAV classes requires new classifiers, retraining, and revised
confidence/disagreement validation.

### Subject-dependent split

The current split includes samples from each participant in development and
test partitions. This measures within-cohort generalization, not strict
generalization to unseen subjects. A leave-one-subject-out or group-based
protocol would answer a different and often harder research question.

### Legacy code

Some historical single-modality scripts depend on loaders or external paths
not included in the active pipeline.

---

## 23. Research Questions for the Final Report

The completed experiments should address:

1. How strong is each modality by itself?
2. Does simple averaging outperform any individual modality?
3. Does validation-derived fixed weighting outperform equal averaging?
4. Does fuzzy adaptive fusion outperform fixed fusion under normal input?
5. Does fuzzy fusion become more useful under corrupted or missing modalities?
6. Are fuzzy weights aligned with observed modality quality?
7. Does maximum-softmax confidence remain calibrated under corruption?
8. Which failure cases are caused by confident but incorrect experts?
9. Are conclusions consistent across subjects and classes?

Results should include effect sizes and uncertainty where possible, rather than
only reporting the single best accuracy.

---

## 24. Summary

EAV combines three sequence-aware neural experts:

```text
EEG:   Conv1D + BiLSTM + self-attention
Audio: MelSpectrogram + CNN + BiLSTM + self-attention
Video: frame CNN + BiLSTM + self-attention
```

Each expert returns a 128-D representation. The original model concatenates
them into 384 dimensions and predicts Sad or Happy with an MLP.

The extension adds independent modality probabilities and a transparent
four-input Mamdani fuzzy inference system. Per-sample confidence and
cross-modal disagreement generate normalized EEG, audio, and video reliability
weights. These weights fuse the expert probabilities without replacing or
redesigning the original neural architecture.

The project is designed to test—not assume—whether explainable adaptive fusion
offers an advantage over the original classifier, unimodal experts, equal
averaging, and fixed weighting, especially when one modality is noisy,
degraded, absent, or contradictory.
