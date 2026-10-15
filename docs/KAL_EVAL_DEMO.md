# कल evaluation के लिए — synthetic demo (जुगाड़ू पर सही protocol)

यह **original EAV dataset download नहीं** करता। छोटा **synthetic** data बनता है — shape/labels वही जो project expect करता है (`Sad=1`, `Happy=3` → model में `0/1`).

## एक कमांड (सब कुछ)

Repository root पर:

```bash
cd "/Users/aarjavjain/Desktop/Projects /EAV-Emotion-Recognition "
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 src/scripts/run_synthetic_demo.py
```

लगभग **15–40 मिनट** (Mac पर MPS/CPU पर depend करता है)। खत्म होने पर terminal में **DEMO SUMMARY** दिखेगा।

## Teacher को क्या दिखाना है

1. **Live fuzzy logic** (data की जरूरत नहीं):
   ```bash
   python3 src/scripts/show_findings.py --data-root data/synthetic
   ```
2. **Trained model + numbers**:
   - `docs/synthetic_demo_results/summary_for_demo.txt`
   - `docs/synthetic_demo_results/metrics.json`
   - `docs/synthetic_demo_results/images/` — confusion matrix, ROC
   - `demo_checkpoints/synthetic_multimodal.pth`

3. **एक लाइन में project**:
   > EEG + Audio + Video experts → late fusion baseline; fuzzy layer हर sample पर confidence/disagreement से weights बदलता है।

## अगर time कम है

सिर्फ baseline (fuzzy skip):

```bash
python3 src/scripts/generate_synthetic_data.py --subjects 8 --samples-per-class 20
python3 src/scripts/train_multimodal.py --data-root data/synthetic \
  --output demo_checkpoints/synthetic_multimodal.pth --epochs 15
python3 src/scripts/evaluate_model.py --data-root data/synthetic \
  --checkpoint demo_checkpoints/synthetic_multimodal.pth \
  --output-dir docs/synthetic_demo_results/images
```

## ईमानदारी (बोलना जरूरी)

- Data **synthetic** है — real hospital/lab recording नहीं।
- Numbers **realistic range** में आते हैं (~80–92% test) क्योंकि classes के बीच learnable patterns डाले गए हैं + थोड़ा noise।
- Project का **code path वही** है जो real preprocessed EAV पर चलेगा।

## दोबारा चलाना (पहले से data है)

```bash
python3 src/scripts/run_synthetic_demo.py --skip-generate
```
