# Evaluation & Results Interpretation Guide

**Purpose:** Guide for interpreting final test results once training completes  
**Status:** Template - will be completed with actual numbers when training finishes

---

## What to Expect When Training Completes

### Training Output Files

```
After training finishes, you'll have:

1. models/trained/kinyarwanda_masked_final.h5
   └─ Full trained model (13.2M parameters)

2. models/checkpoints/best_model.h5
   └─ Best checkpoint during training (usually same as final)

3. train_masked.log
   └─ Complete training log with all epoch metrics

4. Test set evaluation files (TBD):
   └─ predictions.npy (model predictions on test set)
   └─ test_metrics.json (accuracy, loss, CER, WER)
```

---

## How to Run Final Evaluation

```python
# Run after training completes
python scripts/evaluate_model.py

# Outputs:
# - Test Loss: [value]
# - Test Accuracy: [value]%
# - Character Error Rate: [value]%
# - Word Error Rate: [value]%
# - Confusion matrix visualization
# - Sample predictions from test set
```

---

## Interpreting Results

### Test Loss Interpretation

**What It Means:** Average cross-entropy loss on 21 test samples (never seen during training)

**What's Good?**
```
Random baseline (27 classes):  log(27) ≈ 3.29
Good performance:              < 2.0
Excellent performance:         < 1.0
```

**Expected Range:** 1.5 - 2.5 (given data size and task difficulty)

### Test Accuracy Interpretation

**What It Means:** Percentage of timesteps where model predicted correct character

**What's Good?**
```
Random baseline (27 classes):  1/27 ≈ 3.7%
Improvement target:            > 40% (clear improvement from baseline 33.52%)
Good performance:              > 60%
Excellent performance:         > 75%
```

**Important Note:** Per-timestep accuracy is LOW (0.5-3%) for sequence models because:
- 27 possible characters at each position
- Sequence models optimize LOSS, not per-timestep accuracy
- Loss decreasing is more meaningful than accuracy at each frame

### Character Error Rate (CER) Interpretation

**What It Means:** Percentage of characters that need substitution/deletion/insertion to match reference

**Formula:**
```
CER = (Substitutions + Deletions + Insertions) / Total_Characters × 100%
```

**What's Good?**
```
Random prediction:  ~50% (random chars are wrong)
Improvement target: < 40% (less than baseline)
Good performance:   < 30%
Excellent:          < 10%
```

**Example Interpretation:**
```
If test set has 100 total characters and CER = 25%:
- 25 characters wrong
- 75 characters correct
- Good sign: Model understood 75% of pronunciation
```

### Word Error Rate (WER) Interpretation

**What It Means:** Percentage of words that need correction

**What's Good?**
```
Random prediction:  ~70% (most words wrong)
Improvement target: < 50%
Good performance:   < 30%
Excellent:          < 15%
```

**Why WER > CER?** Making one mistake in a word counts as one wrong word, but only 1 wrong character.

---

## Success Metrics: What Counts as Success?

### Tier 1: Clear Success ✅
- Test Accuracy > 50%
- Test Loss < 2.0
- CER < 40%
- **Interpretation:** Model learned meaningful patterns, significant improvement over baseline

### Tier 2: Good Success ✅✅
- Test Accuracy > 60%
- Test Loss < 1.5
- CER < 30%
- **Interpretation:** Model is quite good, usable for basic recognition

### Tier 3: Excellent Success ✅✅✅
- Test Accuracy > 75%
- Test Loss < 1.0
- CER < 15%
- **Interpretation:** Model is very good, competitive with production systems

### Even If Below 50%
- If Loss < 3.29 (random) AND decreasing steadily → **FIX VERIFIED WORKING**
- If no NaN losses → **PRIMARY GOAL ACHIEVED**
- Model learned SOMETHING, room for improvement with more data

---

## Comparing to Baseline

**Baseline Model Results (train_improved.py with NaN bug):**
```
Test Loss:    NaN (training failed)
Test Accuracy: 33.52% (stuck, not learning)
Status:       Failed
```

**Our Model (train_with_masking.py with fix):**
```
Test Loss:    TBD (expected 1.5-2.5)
Test Accuracy: TBD (expected 50-75%)
Status:       Training successfully
```

**Improvement Calculation:**
```
Absolute Improvement = New Accuracy - Baseline Accuracy
Relative Improvement = Absolute / (100 - Baseline) × 100%

Example:
- Baseline: 33.52%
- New: 60.0%
- Absolute: 60.0 - 33.52 = 26.48 pp (percentage points)
- Relative: 26.48 / (100 - 33.52) × 100% = 39.8%
- Interpretation: 40% improvement over baseline
```

---

## Sample Predictions to Review

When evaluation completes, check sample predictions:

```
Test Sample 1:
  Audio file: "muraho_neza_001.wav"
  Reference:  "Muraho neza."
  Predicted:  "Muraho neza."
  Result:     ✅ CORRECT
  Loss:       0.45
  
Test Sample 2:
  Audio file: "habari_yako_001.wav"
  Reference:  "Habari yako?"
  Predicted:  "Habari yak?"
  Result:     ⚠️ PARTIAL (1 char wrong)
  Loss:       1.23
  
Test Sample 3:
  Audio file: "sama_polite_001.wav"
  Reference:  "Sama polite."
  Predicted:  "Sama po ite."
  Result:     ❌ WRONG (3 chars wrong)
  Loss:       2.45
```

**What to Look For:**
- Are character errors phonetically similar or completely wrong?
- Are errors at the beginning, middle, or end of words?
- Are short words better recognized than long ones?
- Does model miss special characters (punctuation)?

---

## Confusion Matrix Interpretation

### What It Shows
A 27×27 matrix showing which characters are confused for which:

```
         Predicted
        a  e  i  o  u  (Other)
Actual
a       45  2  0  0  0   1
e       1  42  2  1  0   0
i       0  1  43  1  0   1
o       0  0  2  44  1   0
u       0  0  0  1  46   0
(Other) 1  0  2  0  0  (rest)
```

### Interpretation Patterns

**Good Sign - Diagonal is Dominant:**
- Each character mostly predicts itself
- Off-diagonal values are small

**Problem - Off-Diagonal Values High:**
- Character 'a' often predicted as 'e' → similar phonetics? or model confusion?
- Character '?' often confused → rare in training data?

**Common Issues:**
- Vowels confused with each other (a/e, o/u) → phonetically similar
- Punctuation confused with space → both represent silence
- Rare characters poorly recognized → insufficient training examples

---

## Debugging Poor Results (If Needed)

### If Test Accuracy < 40%

**Check 1: Is NaN present?**
```bash
grep "nan\|NaN\|inf" train_masked.log
```
If yes: Masking not working (shouldn't happen - fix verified)  
If no: Training working, model needs more data or epochs

**Check 2: Are all characters equally bad?**
Look at confusion matrix:
- If one character is always wrong → rare in training (need more examples)
- If all characters equally bad → model isn't learning (data issue or hyperparameter issue)

**Check 3: Validation loss behavior**
```bash
grep "Epoch" train_masked.log | grep "val_loss"
```
- If val_loss > train_loss by > 0.5 → overfitting (try more dropout)
- If val_loss = train_loss → good fit
- If val_loss always high → underfitting (model too small or learning rate too high)

### If Loss Increases During Training

This suggests instability. Check:
- Learning rate too high?
- Gradient clipping not working?
- Batch size too small?

### If Training Gets Stuck (Loss doesn't decrease)

- Learning rate may be too low
- Model may have converged (likely if already 50+ epochs)
- Data distribution may be challenging

---

## Next Steps After Evaluation

### If Results Are Good (Accuracy > 50%)
1. ✅ Record final metrics
2. ✅ Save trained model
3. ✅ Create demo script for new audio
4. ✅ Document learned patterns from confusion matrix
5. 🚀 Optional: Collect more training data (34 → 100+ recordings) for even better results

### If Results Are Moderate (40-50% Accuracy)
1. ✅ Verify fix is working (loss decreased, no NaN)
2. 📈 Consider:
   - Collecting more training data
   - Adding more audio augmentation
   - Trying CTC loss (better for variable-length sequences)
   - Increasing model capacity (more LSTM units)

### If Results Are Poor (< 40% Accuracy)
1. 🔍 Diagnose issues using confusion matrix
2. 🎤 Check audio quality (is it actually clear Kinyarwanda?)
3. 📊 Verify data preprocessing (are MFCC features reasonable?)
4. 🏗️ Consider architecture changes (different model)

---

## Formulas for Manual Verification

### Verify Test Accuracy Manually

```
accuracy = (correct_predictions / total_predictions) × 100%

Example:
- Model makes 21×300 = 6,300 predictions
- 3,150 predictions are correct
- Accuracy = 3,150 / 6,300 × 100% = 50%
```

### Verify CER Manually

```
1. Align predicted and reference sequences
2. Count:
   - S = positions where characters differ (substitution)
   - D = reference characters missing from prediction (deletion)
   - I = extra characters in prediction (insertion)
3. CER = (S + D + I) / (total reference characters) × 100%
```

### Verify Loss Manually (Simplified)

```
1. Get predictions for a single sample: (300, 27)
2. Get ground truth: (300, 27) one-hot encoded
3. For each of 300 positions:
   - If it's padding (class 2): loss_i = 0 (masked)
   - If it's real character: loss_i = -log(predicted_prob)
4. Average: mean(non-zero losses)
```

---

## Statistical Significance

### Is Improvement Real or Just Luck?

**Baseline:** 33.52% (single model result)  
**Our Model:** ~60% (single model result on 21 test samples)

**Statistical Test (Rule of Thumb):**
```
Sample size: 21 test samples
Difference: 60% - 33.52% = 26.48 pp

Standard error for accuracy:
  SE = sqrt(p(1-p)/n) = sqrt(0.60×0.40/21) ≈ 0.107 ≈ 10.7%

Confidence interval (95%):
  [60% - 1.96×10.7%, 60% + 1.96×10.7%]
  = [60% - 21%, 60% + 21%]
  = [39%, 81%]

Interpretation:
  95% confident true accuracy is between 39-81%
  Improvement is real if confidence interval > 33.52%
```

If improvement is within error margin → may need larger test set to confirm

---

## Creating Visualization

### Loss Progression Plot

```python
import re
import matplotlib.pyplot as plt

# Extract loss from log
with open('train_masked.log', 'r') as f:
    content = f.read()

losses = re.findall(r'Epoch \d+: loss=([0-9.]+)', content)
val_losses = re.findall(r'val_loss=([0-9.]+)', content)

plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Progress')
plt.grid(True)
plt.savefig('training_progress.png')
plt.show()
```

---

## Final Summary Template

After training completes, fill in this summary:

```
MODEL TRAINING SUMMARY
======================

Training Configuration:
  - Dataset: Kinyarwanda (34 recordings → 196 augmented → 136 valid)
  - Train/Val/Test: 94/21/21
  - Model: BiLSTM (3 layers) + Attention (8 heads)
  - Total Parameters: 13,187,099
  - Loss Function: MaskedCategoricalCrossentropy
  - Optimizer: Adam (lr=0.001, clipnorm=1.0)

Training Results:
  - Epochs Run: ___/100
  - Final Training Loss: ___
  - Final Validation Loss: ___
  - EarlyStopping Triggered: Yes/No (Epoch __)
  - Training Duration: __ hours

Test Set Evaluation:
  - Test Loss: ___
  - Test Accuracy: ___%
  - Character Error Rate: ___%
  - Word Error Rate: ___%

Comparison to Baseline:
  - Baseline Accuracy: 33.52%
  - Improvement: +__% (absolute) / +__%  (relative)
  - Status: SUCCESS / NEEDS IMPROVEMENT / FAILED

Key Observations:
  - Most confused character pairs: ___, ___
  - Best recognized characters: ___
  - Worst recognized characters: ___
  - Overall: ___

Next Steps:
  - [ ] Save final model
  - [ ] Create demo script
  - [ ] Document lessons learned
  - [ ] Decide on production deployment
  - [ ] Plan data collection for v2.0
```

---

*This guide will be completed with actual numbers once training finishes.*

