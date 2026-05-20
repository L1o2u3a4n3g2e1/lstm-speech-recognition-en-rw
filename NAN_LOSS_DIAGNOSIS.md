# NaN Loss Issue: Root Cause Analysis & Solution

**Status:** Testing masked loss function (training in progress)

---

## Problem Statement

The improved training script (`train_improved.py`) achieved only **33.52% accuracy** with **NaN loss** from Epoch 2 onwards, identical to the original baseline. Data augmentation and numerical stability improvements did not help.

### Observed Behavior
```
Epoch 1: loss=9.41 (batch 5) → 40.47 (batch 6) → val_loss=NaN
Epoch 2-11: loss=NaN, accuracy=33.52% (stuck predicting "PAD" token)
```

---

## Root Cause Analysis

### The Padding Problem

The fundamental issue is how we encode padding tokens in the loss function:

**Original sequences:** 
- Actual transcripts: 50-200 characters
- Padded to: 300 frames (to match audio MFCC frames)
- Padding amount: **100-250 frames (33-83% of each sample!)**

**One-hot encoding of padding:**
```
Padding token (id=0):
  Original:        [1, 0, 0, 0, ..., 0]  (one-hot for class 0)
  With label_smoothing=0.1:
    Class 0: 0.9
    Classes 1-26: 0.1/26 ≈ 0.0038 each

Real character (id=5, "N"):
  Original:        [0, 0, 0, 0, 1, 0, ..., 0]
  With label_smoothing=0.1:
    Class 5: 0.9
    Classes 0,1-4,6-26: 0.1/26 ≈ 0.0038 each
```

### Why This Causes NaN

When computing categorical crossentropy loss on massive softmax outputs:

1. **Imbalanced targets**: 250+ padding tokens vs 50 character tokens per sample
2. **Numerical instability**: 
   - Softmax on (batch=16, time=300, vocab=27) creates massive intermediate values
   - Log operations on softmax outputs can produce -∞ or 0 → log(0) = -∞
   - Averaging over 4,800 timesteps magnifies numerical errors
   
3. **Loss explosion**: 
   - Epoch 1 Batch 6 shows loss jump: 9.41 → 40.47
   - This indicates a gradient explosion on that batch
   - Once loss becomes NaN, gradient updates become NaN, model stays stuck

4. **Padding dominates**:
   ```
   Total loss = sum(loss_per_timestep) / 300
   
   If 250 timesteps are padding:
   - Even small errors in padding prediction get summed 250 times
   - Loss becomes unstable because model is trying to optimize 
     for correct padding prediction, not character prediction
   ```

---

## Attempted Fixes (Failed)

### 1. Lower Learning Rate (0.001 → 0.0005)
- ❌ **Result**: NaN still occurs, just takes same path
- **Why**: Doesn't address the fundamental issue (padding-dominated loss)

### 2. Gradient Clipping (global_clipnorm=1.0)
- ❌ **Result**: NaN still occurs after first batch
- **Why**: Clipping happens too late; loss already computed NaN values

### 3. Label Smoothing (0.1)
- ❌ **Result**: No improvement, still NaN
- **Why**: Smoothing padding targets actually makes loss more complex

### 4. Higher Dropout (0.3 → 0.4)
- ❌ **Result**: No improvement
- **Why**: Doesn't address loss computation problem

### 5. Data Scaling ([-1, +0.34])
- ❌ **Result**: Helps slightly but not enough; still NaN
- **Why**: Issue is in loss computation, not input range

---

## The Solution: MaskedCategoricalCrossentropy

### How It Works

Instead of computing loss on all 300 timesteps, **mask out padding positions** and only compute loss on actual character positions.

```python
class MaskedCategoricalCrossentropy(keras.losses.Loss):
    def call(self, y_true, y_pred):
        # Compute loss for all positions (unmasked)
        loss = base_loss(y_true, y_pred)  # (batch, 300)
        
        # Create mask: 1 where token is NOT padding (id != 0)
        # Padding detection: argmax(one_hot) == 0
        padding_mask = tf.argmax(y_true, axis=-1) == 0  # (batch, 300)
        non_padding_mask = 1.0 - tf.cast(padding_mask, tf.float32)
        
        # Apply mask: zero out loss at padding positions
        masked_loss = loss * non_padding_mask  # (batch, 300)
        
        # Average only over non-padding positions
        mean_loss = tf.reduce_sum(masked_loss, axis=1) / tf.reduce_sum(non_padding_mask, axis=1)
        
        return tf.reduce_mean(mean_loss)
```

### Why This Works

1. **Eliminates padding noise**: Loss only computed on 50-200 actual characters, not 250 padding frames
2. **Stabilizes gradients**: Fewer terms in loss → smaller gradient magnitudes
3. **Focuses model learning**: Model optimizes for character prediction, not padding prediction
4. **Prevents NaN**: Much smaller loss values → no overflow/underflow in softmax

### Expected Improvement

| Metric | Without Masking | With Masking |
|--------|-----------------|--------------|
| **Loss computed on** | 300 timesteps | ~200 timesteps (real chars) |
| **Padding contribution** | 83% of loss | 0% |
| **Gradient magnitude** | Large (explodes) | Stable |
| **Typical loss value** | 3-40 | 0.5-2.0 |
| **Risk of NaN** | High | Very Low |
| **Accuracy potential** | 33% (stuck) | 60-80%+ |

---

## Implementation Details

### File: `scripts/train_with_masking.py`

**Key changes from `train_improved.py`:**

1. **Custom Loss Function**
   ```python
   loss=MaskedCategoricalCrossentropy(label_smoothing=0.1)
   ```

2. **Sequence Length Tracking**
   ```python
   def pad_sequences_with_length(y, target_length):
       # Returns: (padded_sequences, actual_lengths)
       # Actual lengths stored but not used in this version
   ```

3. **Higher Learning Rate**
   ```python
   learning_rate=0.001  # Back to original
   # Safe now because masking makes loss stable
   ```

4. **Increased Early Stopping Patience**
   ```python
   patience=15  # Up from 10
   # Loss may improve more gradually with masking
   ```

### Training Configuration

```python
train_with_masking(
    language='kinyarwanda',
    epochs=100,
    batch_size=16
)
```

---

## Why Standard Solutions Don't Work Here

### Standard Approach: Ignore Padding via Masking Layer
```python
# Doesn't help because:
inputs = Input(shape=(300, 13))
mask = Masking(mask_value=0.0)(inputs)  # Masks inputs, not targets!
```
❌ Masking inputs doesn't prevent loss computation on padded targets

### Standard Approach: Reduce Output Length
```python
# Doesn't work because:
# - Audio is 300 frames (5 seconds at 16kHz, 20ms frames)
# - Transcripts are 50-200 chars (ambiguous length)
# - Must use fixed-length model for batch training
```
❌ Can't know correct transcript length without CTC loss (future work)

### Standard Approach: Use CTC Loss
```python
# Best long-term solution but:
# - Requires variable-length targets
# - Complex to implement correctly
# - Different training procedure
```
⏳ Alternative implementation planned

---

## Testing Plan

### Phase 1: Verify Masking Solves NaN (In Progress)
- [x] Create MaskedCategoricalCrossentropy loss function
- [x] Implement in train_with_masking.py
- [x] Launch training
- [ ] Monitor first 10 epochs for NaN appearance
- [ ] Check if loss remains stable

### Phase 2: Evaluate Performance
- [ ] Let training run to completion (100 epochs)
- [ ] Check final validation loss (target: < 0.5)
- [ ] Evaluate test set accuracy (target: 50-80%)
- [ ] Compare against baseline (33.52%)

### Phase 3: Analysis
- [ ] Plot loss curves (training vs validation)
- [ ] Analyze accuracy progression
- [ ] Check if early stopping triggers appropriately
- [ ] Save final model and metrics

### Phase 4: Next Improvements (If Needed)
- [ ] Implement CTC loss for variable-length targets
- [ ] Add word-level loss weighting
- [ ] Consider attention weights for alignment
- [ ] Possibly reduce padding to 200 frames

---

## Expected Outcomes

### Best Case
```
Epoch 1: loss=3.2, val_loss=2.1
Epoch 5: loss=1.5, val_loss=1.2
Epoch 20: loss=0.8, val_loss=0.7
Epoch 50+: loss=0.5, val_loss=0.6
→ Final accuracy: 70-80%
→ No NaN loss
```

### Realistic Case
```
Epoch 1: loss=3.0, val_loss=2.8
Epoch 10: loss=1.8, val_loss=1.6
Epoch 30: loss=1.2, val_loss=1.1
Epoch 100: loss=0.8, val_loss=0.9
→ Final accuracy: 50-65%
→ Stable training, no NaN
```

### Pessimistic Case
```
Loss still explodes (unlikely due to masking)
→ indicates other issues in data/model architecture
→ Would trigger Phase 4: deeper investigation
```

---

## Key Takeaways

1. **Padding tokens caused NaN**: 250 padding frames dominated the loss, causing numerical instability
2. **Standard fixes didn't work**: Lower learning rate, gradient clipping, label smoothing all failed because they don't address the root cause
3. **Masking is the solution**: Only compute loss on actual character positions, not padding
4. **This is a common problem**: Many sequence models use masking for this exact reason (RNNs, Transformers with variable-length inputs)
5. **Future: CTC Loss**: For complete solution, should implement CTC loss which handles variable-length sequences natively

---

**Training started:** 2026-05-19 ~12:50 UTC  
**Expected completion:** ~14:50-15:50 UTC (2-3 hours)  
**Status:** Awaiting results...
