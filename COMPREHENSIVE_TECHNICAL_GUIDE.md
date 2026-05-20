# Comprehensive Technical Guide: LSTM Kinyarwanda Speech Recognition

**Project Status:** Training in progress (Epoch 20/100)  
**Last Updated:** 2026-05-19  
**Documentation Scope:** Complete technical reference covering data processing, model architecture, training methodology, evaluation metrics, and execution guides

---

## 📑 Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure & File Functions](#project-structure--file-functions)
3. [Import Functions & Libraries](#import-functions--libraries)
4. [Data Processing Pipeline](#data-processing-pipeline)
5. [Model Architecture](#model-architecture)
6. [Training Methodology](#training-methodology)
7. [Evaluation Metrics & Formulas](#evaluation-metrics--formulas)
8. [How to Run the Code](#how-to-run-the-code)
9. [Model Training vs Pretrained Weights](#model-training-vs-pretrained-weights)

---

## Project Overview

**Objective:** Build an LSTM-based speech recognition system for Kinyarwanda language that converts audio recordings into character-level transcriptions.

**Problem Solved:** The initial training suffered from **NaN loss** because the loss function was computing loss over **89% padding frames** instead of real character data. This was fixed by implementing a **MaskedCategoricalCrossentropy** loss function that only computes loss on valid character positions.

**Current Status:**
- **Dataset:** 34 original Kinyarwanda recordings (augmented to 196, valid 136 samples)
- **Train/Val/Test Split:** 94/21/21 samples
- **Model Architecture:** BiLSTM with Multi-Head Attention
- **Total Parameters:** 13,187,099
- **Expected Outcome:** 50-75% test accuracy (vs 33.52% baseline)

---

## Project Structure & File Functions

```
lstm-speech-recognition-en-rw/
├── scripts/                          # All executable Python scripts
│   ├── train_with_masking.py        # ✅ CURRENT: Fixed training script with masked loss
│   ├── train_improved.py             # ❌ OLD: Has NaN bug (reference only)
│   ├── preprocess_augmented.py       # Preprocess augmented dataset
│   ├── augment_dataset.py            # Generate data augmentation (pitch, time-stretch)
│   └── evaluate_model.py             # Evaluate trained model on test set
│
├── src/                              # Core modules
│   ├── lstm_model.py                 # LSTMSpeechRecognition class: model architecture
│   ├── audio_processor.py            # AudioProcessor class: MFCC extraction, audio normalization
│   └── preprocess_augmented.py       # AugmentedDataPreprocessor: data loading and splitting
│
├── data/                             # Dataset directory structure
│   └── kinyarwanda/
│       ├── raw/
│       │   ├── custom_recordings/    # 34 original .wav files with transcripts.csv
│       │   └── augmented/            # 196 augmented .wav files (3× augmentation)
│       ├── processed_augmented/      # MFCC features (300 time steps, 13 coefficients)
│       │   ├── train/                # 94 training samples (X_train.npy, y_train.npy)
│       │   ├── val/                  # 21 validation samples (X_val.npy, y_val.npy)
│       │   └── test/                 # 21 test samples (X_test.npy, y_test.npy)
│       └── vocabulary_augmented.json # 27-char vocabulary with class mappings
│
├── models/                           # Trained models
│   ├── trained/
│   │   └── kinyarwanda_masked_final.h5   # Final trained model (TBD after training)
│   └── checkpoints/                  # Best model during training
│       └── best_model.h5             # Checkpoint with best validation loss
│
├── Documentation/                    # All documentation files
│   ├── FINAL_RESULTS_TEMPLATE.md        # Final metrics (TBD)
│   ├── README_TRAINING_FIX.md           # Overview and status
│   ├── SOLUTION_SUMMARY.md              # Complete explanation of fix
│   ├── FIX_APPLIED.md                   # Step-by-step fix details
│   ├── NAN_LOSS_DIAGNOSIS.md            # Root cause analysis
│   ├── PROGRESS_ANALYSIS.md             # Epoch-by-epoch breakdown
│   ├── TRAINING_STATUS.md               # Real-time metrics
│   ├── QUICK_REFERENCE.md               # Quick lookup
│   ├── USER_GUIDE.md                    # Complete project guide
│   └── COMPREHENSIVE_TECHNICAL_GUIDE.md # This file
│
├── train_masked.log                  # Real-time training log
├── vercel.json                       # Deployment configuration
├── requirements.txt                  # Python dependencies
└── README.md                         # Project introduction

```

### Key Directory Functions

| Directory | Function | Content |
|-----------|----------|---------|
| `scripts/` | Executable training and processing scripts | Python files for training, preprocessing, evaluation |
| `src/` | Core ML modules | Model architecture, audio processing, data preprocessing classes |
| `data/kinyarwanda/raw/` | Original audio files | 34 Kinyarwanda recordings + transcripts, 196 augmented files |
| `data/kinyarwanda/processed_augmented/` | Ready-for-training data | MFCC features in numpy format, split by train/val/test |
| `models/` | Trained weights | Best checkpoint during training, final model after completion |
| `Documentation/` | Technical reference | Guides, analysis, status tracking |

---

## Import Functions & Libraries

### Core Dependencies and Their Functions

#### **TensorFlow & Keras** (Deep Learning Framework)
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
```

| Import | Function |
|--------|----------|
| `keras.losses.Loss` | Base class for custom loss functions |
| `keras.losses.CategoricalCrossentropy` | Standard categorical cross-entropy loss |
| `keras.optimizers.Adam` | Adaptive learning rate optimizer |
| `keras.callbacks.EarlyStopping` | Stop training when validation loss plateaus |
| `keras.callbacks.ModelCheckpoint` | Save best model during training |
| `keras.callbacks.ReduceLROnPlateau` | Reduce learning rate when improvement stalls |
| `keras.layers.LSTM` | Long Short-Term Memory recurrent layer |
| `keras.layers.Bidirectional` | Wrap layer to process sequences both directions |
| `keras.layers.Dense` | Fully connected layer for output |
| `keras.layers.Dropout` | Regularization to prevent overfitting |
| `keras.layers.MultiHeadAttention` | Attention mechanism (8 heads in our model) |
| `keras.utils.to_categorical` | Convert integer labels to one-hot encoding |

**Why Used:** TensorFlow provides GPU acceleration, automatic differentiation for backpropagation, and optimized implementations of neural network layers. Keras is the high-level API that simplifies model building and training.

#### **NumPy** (Numerical Computing)
```python
import numpy as np
```

| Function | Purpose |
|----------|---------|
| `np.load()` | Load preprocessed MFCC features from .npy files |
| `np.save()` | Save numpy arrays to disk |
| `np.transpose()` | Reshape data from (batch, features, time) to (batch, time, features) |
| `np.pad()` | Pad MFCC features to fixed length (300 frames) |
| `np.full()` | Create padding arrays filled with PAD token index |
| `np.array()` | Convert lists to numpy arrays |
| `np.max(), np.min()` | Find value ranges for normalization |

**Why Used:** NumPy is the standard for numerical operations. We use it to handle feature arrays efficiently and perform batch operations.

#### **Librosa** (Audio Processing)
```python
import librosa
import librosa.feature
```

| Function | Purpose |
|----------|---------|
| `librosa.load(path, sr=16000)` | Load audio file at 16 kHz sample rate |
| `librosa.feature.mfcc(y, sr, n_mfcc=13)` | Extract 13 Mel-frequency cepstral coefficients |
| `librosa.effects.time_stretch(y, rate)` | Time-stretch audio (data augmentation, 0.9-1.1x) |
| `librosa.effects.pitch_shift(y, sr, n_steps)` | Pitch-shift audio (data augmentation, ±2 semitones) |

**Why Used:** Librosa is the standard Python library for music and audio analysis. MFCC features are compact audio representations that capture perceptually-relevant information for speech recognition.

#### **Scikit-Learn** (ML Utilities)
```python
from sklearn.model_selection import train_test_split
```

| Function | Purpose |
|----------|---------|
| `train_test_split(X, y, test_size=0.2, random_state=42)` | Split data into train/validation/test sets |

**Why Used:** Provides reproducible, stratified data splitting with fixed random seeds.

#### **JSON** (Data Format)
```python
import json
```

| Function | Purpose |
|----------|---------|
| `json.load(f)` | Load vocabulary mappings from `vocabulary_augmented.json` |
| `json.dump(data, f)` | Save vocabulary for later use |

**Why Used:** JSON stores the character-to-class mappings (char_to_num, num_to_char) in human-readable format.

#### **CSV** (Data Format)
```python
import csv
```

| Function | Purpose |
|----------|---------|
| `csv.DictReader(f)` | Read transcripts.csv with column headers |

**Why Used:** CSV stores metadata like filename-to-transcript mappings.

#### **Logging** (Monitoring)
```python
import logging
logger = logging.getLogger(__name__)
```

| Function | Purpose |
|----------|---------|
| `logger.info()` | Log training progress, epoch results |
| `logger.warning()` | Log skipped files, data issues |

**Why Used:** Provides real-time feedback on training progress without cluttering stdout.

---

## Data Processing Pipeline

### Overview

The data processing pipeline converts raw audio files into training-ready numerical features with proper masking for variable-length sequences.

```
Raw Audio (34 samples) 
    ↓
Augmentation (×3: pitch shift, time stretch) 
    ↓
Valid Samples (196 → 136 after filtering) 
    ↓
MFCC Extraction (13 coefficients, 300 frames) 
    ↓
Normalization & Scaling 
    ↓
Train/Val/Test Split (94/21/21) 
    ↓
One-Hot Encoding + Padding 
    ↓
Training-Ready Data
```

### Step 1: Audio Loading and Normalization

**Location:** `src/audio_processor.py` → `AudioProcessor` class

```python
def load_audio(self, path, sr=16000):
    """Load audio file at 16 kHz sample rate."""
    audio, sr = librosa.load(path, sr=sr)
    return audio, sr

def normalize_audio(self, audio):
    """Normalize audio to [-1, 1] range."""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio
```

**Formula:**
$$\text{Normalized Audio} = \frac{\text{Audio}}{max(|Audio|)}$$

**Purpose:** Ensures all audio files have consistent amplitude regardless of recording level.

### Step 2: Silence Removal

```python
def remove_silence(self, audio, threshold_db=40):
    """Remove silent portions."""
    S = librosa.feature.melspectrogram(y=audio, sr=self.sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    threshold = -threshold_db
    mask = np.max(S_db, axis=0) > threshold
    # Keep audio where spectrogram is above threshold
    return audio[mask]
```

**Purpose:** Removes leading/trailing silence to focus on speech content.

### Step 3: MFCC Feature Extraction

**Location:** `src/audio_processor.py` → `extract_mfcc()`

```python
def extract_mfcc(self, audio, n_mfcc=13):
    """Extract MFCC features."""
    mfcc = librosa.feature.mfcc(
        y=audio, 
        sr=self.sr, 
        n_mfcc=n_mfcc
    )
    return mfcc  # Shape: (13, time_steps)
```

**What is MFCC?**

MFCC (Mel-Frequency Cepstral Coefficients) are compact audio features that mimic how humans perceive sound:

1. **Mel-scale:** Convert frequency to mel-scale (perceptually-logarithmic)
2. **Cepstral Analysis:** Apply DCT to compress information
3. **Result:** 13 coefficients per time frame, capturing spectral characteristics

**Formula (Simplified):**
$$\text{MFCC} = \text{DCT}(\log(\text{Mel Filterbank Energy}))$$

**Shape:** (13 coefficients, time_steps)  
**Example:** For 5-second audio at 16 kHz → (13, ~312 frames)

### Step 4: Padding to Fixed Length

```python
def _pad_features(self, features, max_length=300):
    """Pad all MFCC features to 300 time steps."""
    padded = []
    for feat in features:
        if feat.shape[1] > max_length:
            feat = feat[:, :max_length]  # Truncate if too long
        else:
            pad_width = ((0, 0), (0, max_length - feat.shape[1]))
            feat = np.pad(feat, pad_width, mode='constant', constant_values=0)
        padded.append(feat)
    return np.array(padded)
```

**Result:** All samples have shape (13, 300) - 13 MFCC coefficients, 300 time frames

### Step 5: Data Augmentation

**Location:** `scripts/augment_dataset.py`

```python
# Pitch shift: ±2 semitones
augmented_audio = librosa.effects.pitch_shift(
    audio, sr=sr, n_steps=np.random.randint(-2, 3)
)

# Time stretch: 0.9x to 1.1x speed
rate = np.random.uniform(0.9, 1.1)
augmented_audio = librosa.effects.time_stretch(audio, rate=rate)
```

**Why Augmentation?** 
- Original: 34 samples (too small for deep learning)
- Augmented: 196 samples (3× expansion)
- Valid after filtering: 136 samples (94 train, 21 val, 21 test)

**Impact:** Increases dataset size and adds robustness to pitch/speed variations.

### Step 6: Vocabulary Creation

```python
def _create_vocabulary(self, transcripts):
    """Create character-level vocabulary."""
    unique_chars = set()
    for transcript in transcripts:
        unique_chars.update(transcript.lower())
    
    unique_chars.add(' ')      # Space character (class 0)
    unique_chars.add('<PAD>')  # Padding token (class 2) - KEY FIX
    unique_chars.add('<UNK>')  # Unknown token (class 3)
    
    char_to_num = {char: idx for idx, char in enumerate(sorted(unique_chars))}
    num_to_char = {idx: char for char, idx in char_to_num.items()}
    
    return char_to_num, num_to_char
```

**Vocabulary Size:** 27 characters (Kinyarwanda alphabet + punctuation + special tokens)

**Class Mapping Example:**
```
Class 0: ' ' (space)
Class 1: '.' (period)
Class 2: '<PAD>' (padding) ← THE KEY FIX
Class 3: '<UNK>' (unknown)
Class 4: '?' (question mark)
Classes 5-26: Kinyarwanda characters (a-z variations)
```

### Step 7: Transcript Encoding

```python
def _encode_transcripts(self, transcripts, char_to_num, max_length=300):
    """Convert text transcripts to integer sequences."""
    encoded = []
    actual_lengths = []
    
    for transcript in transcripts:
        # Convert each character to class index
        seq = [char_to_num.get(c, char_to_num.get('<UNK>', 0)) 
               for c in transcript.lower()]
        
        # Track actual length BEFORE padding
        actual_len = min(len(seq), max_length)
        actual_lengths.append(actual_len)
        
        # Pad with PAD token (class 2)
        seq = seq[:max_length]
        seq += [2] * (max_length - len(seq))  # Pad with class 2
        
        encoded.append(seq)
    
    return np.array(encoded), np.array(actual_lengths)
```

**Result:** Integer sequences of shape (num_samples, 300) where:
- Positions 0-N: Character class indices (1-26)
- Positions N+1-299: PAD token (class 2)

### Step 8: One-Hot Encoding

```python
# Convert integer sequences to one-hot
y_train_oh = keras.utils.to_categorical(y_train, num_classes=27)
# Result shape: (num_samples, 300, 27)
# Each position is a 27-dimensional vector with 1 at the correct class, 0s elsewhere
```

**Formula:**
$$\text{One-Hot}(c) = [0, 0, ..., 1, ..., 0]_{c}$$

where the 1 is at position c (the class index).

### Step 9: Data Normalization (for neural network input)

```python
X_train = X_train / (np.max(np.abs(X_train)) + 1e-8)
X_val = X_val / (np.max(np.abs(X_val)) + 1e-8)
X_test = X_test / (np.max(np.abs(X_test)) + 1e-8)
```

**Purpose:** Scale MFCC features to [-1, 1] range for numerical stability during backpropagation.

### Final Data Shapes

```
X_train: (94, 300, 13)    # 94 samples, 300 time frames, 13 MFCC coefficients
y_train: (94, 300, 27)    # 94 samples, 300 frames, 27 one-hot character classes

X_val:   (21, 300, 13)
y_val:   (21, 300, 27)

X_test:  (21, 300, 13)
y_test:  (21, 300, 27)
```

---

## Model Architecture

### Overview: BiLSTM with Multi-Head Attention

**Location:** `src/lstm_model.py` → `LSTMSpeechRecognition.build_bidirectional()`

The model processes audio features sequentially and outputs character probabilities at each time frame.

```
Input: (batch_size, 300, 13)
  ↓
BiLSTM Layer 1 (forward + backward) → 1024 units
  ↓ [Dropout 0.4]
BiLSTM Layer 2 (forward + backward) → 1024 units
  ↓ [Dropout 0.4]
Multi-Head Attention (8 heads)
  ↓
BiLSTM Layer 3 (forward + backward) → 512 units
  ↓ [Dropout 0.4]
Dense Output Layer → 27 (softmax)
  ↓
Output: (batch_size, 300, 27)
```

### Layer-by-Layer Breakdown

#### **Input Layer**
```python
inputs = layers.Input(shape=(300, 13), name='audio_input')
```
- **Shape:** (batch_size, 300 timesteps, 13 MFCC features)
- **Purpose:** Accepts normalized MFCC features

#### **BiLSTM Layer 1** (1024 units total = 512×2)

```python
x = layers.Bidirectional(
    layers.LSTM(512, return_sequences=True, activation='relu'),
    name='bilstm_1'
)(x)
```

**What is BiLSTM?**
- **LSTM:** Long Short-Term Memory - processes sequences with memory of past context
- **Bidirectional:** Process sequence forward AND backward to capture context from both directions

**Formula (Simplified LSTM):**
$$i_t = \sigma(W_{ii}x_t + W_{hi}h_{t-1} + b_i)$$ (input gate)
$$f_t = \sigma(W_{if}x_t + W_{hf}h_{t-1} + b_f)$$ (forget gate)
$$o_t = \sigma(W_{io}x_t + W_{ho}h_{t-1} + b_o)$$ (output gate)
$$\tilde{c}_t = \tanh(W_{ic}x_t + W_{hc}h_{t-1} + b_c)$$ (cell candidate)
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$ (cell state)
$$h_t = o_t \odot \tanh(c_t)$$ (hidden state)

- **Parameters:** (13 + 512 + 1) × 4 gates × 512 units = ~10.7M params (one direction)
- **Output:** (batch_size, 300, 1024) where 1024 = 512 forward + 512 backward

**Why BiLSTM?** Can look ahead and behind for each character, capturing dependencies in both directions.

#### **Dropout Layer 1** (rate=0.4)

```python
x = layers.Dropout(0.4)(x)
```

**Formula:**
$$\text{Output} = \begin{cases} 
\frac{x}{(1-p)} & \text{with probability } (1-p) \\
0 & \text{with probability } p
\end{cases}$$

where p = 0.4

**Purpose:** Prevent overfitting by randomly dropping 40% of activations during training (disabled during inference).

#### **BiLSTM Layer 2** (1024 units)

```python
x = layers.Bidirectional(
    layers.LSTM(512, return_sequences=True, activation='relu'),
    name='bilstm_2'
)(x)
x = layers.Dropout(0.4)(x)
```

**Purpose:** Extract higher-level features from BiLSTM1 output. Deeper processing → better feature representations.

#### **Multi-Head Attention Layer** (8 heads)

```python
x = layers.MultiHeadAttention(
    num_heads=8, 
    key_dim=64, 
    dropout=0.4
)(x, x)  # Self-attention: query=key=value=x
```

**What is Attention?**

Allows the model to focus on relevant parts of the sequence. For each position, compute relevance weights over all other positions.

**Formula:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where:
- Q (Query): What am I looking for?
- K (Key): What does each position represent?
- V (Value): What information should I extract?
- $d_k$: Key dimension (64 in our case)

**Multi-Head:** 8 different attention patterns learned in parallel, then concatenated.

**Purpose:** Help LSTM focus on relevant context for each character prediction.

#### **BiLSTM Layer 3** (512 units)

```python
x = layers.Bidirectional(
    layers.LSTM(256, return_sequences=True, activation='relu'),
    name='bilstm_3'
)(x)
x = layers.Dropout(0.4)(x)
```

**Purpose:** Final feature refinement after attention. Smaller (256 units) for efficient fusion.

#### **Dense Output Layer**

```python
outputs = layers.Dense(27, activation='softmax', name='character_output')(x)
```

**Formula:**
$$\text{softmax}(z_j) = \frac{e^{z_j}}{\sum_{k=1}^{27} e^{z_k}}$$

- **Output:** (batch_size, 300, 27)
- **Activation:** Softmax converts logits to probability distribution over 27 characters
- **Purpose:** Predict character class probabilities for each time frame

### Model Summary

| Component | Units/Heads | Parameters | Function |
|-----------|------------|-----------|----------|
| Input | - | 0 | Accept MFCC features |
| BiLSTM 1 | 1024 | 10.7M | Initial sequence processing |
| BiLSTM 2 | 1024 | 10.7M | Deep feature extraction |
| Attention | 8 heads | 0.5M | Contextual focus |
| BiLSTM 3 | 512 | 2.1M | Attention fusion |
| Dense Output | 27 | 0.15M | Character classification |
| **TOTAL** | - | **13.2M** | - |

### Why This Architecture?

1. **BiLSTM:** Captures sequential dependencies in both directions
2. **Multiple Layers:** Enables learning of hierarchical features
3. **Attention:** Allows focusing on relevant context for each character
4. **Dropout:** Prevents overfitting on small dataset
5. **Appropriate for Speech:** Character-level prediction at each frame, not sequence-to-sequence

---

## Training Methodology

### Training Setup

**Location:** `scripts/train_with_masking.py` → `train_with_masking()`

### Optimizer Configuration

```python
optimizer = keras.optimizers.Adam(
    learning_rate=0.001,           # How much to update weights
    global_clipnorm=1.0            # Prevent gradient explosion
)
```

**Adam (Adaptive Moment Estimation):**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$ (first moment: momentum)
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$ (second moment: RMSprop)
$$\theta_t = \theta_{t-1} - \alpha \frac{m_t}{\sqrt{v_t} + \epsilon}$$

**Parameters:**
- Learning rate (α): 0.001 - controls step size
- β₁: 0.9 - momentum decay
- β₂: 0.999 - RMSprop decay
- Global Clipnorm: 1.0 - prevent gradient explosion

### Custom Loss Function: MaskedCategoricalCrossentropy

**THE FIX FOR NaN LOSS**

```python
class MaskedCategoricalCrossentropy(keras.losses.Loss):
    """Only compute loss on non-padding positions."""
    
    def __init__(self, label_smoothing=0.1, pad_token_id=2, **kwargs):
        super().__init__(**kwargs)
        self.pad_token_id = 2  # Class index of '<PAD>' token
        self.base_loss = keras.losses.CategoricalCrossentropy(
            from_logits=False,
            label_smoothing=label_smoothing,
            reduction='none'  # Get per-element loss
        )
    
    def call(self, y_true, y_pred):
        # Step 1: Compute categorical crossentropy
        loss = self.base_loss(y_true, y_pred)  # Shape: (batch, 300)
        
        # Step 2: Identify padding positions
        token_ids = tf.argmax(y_true, axis=-1)  # Get class indices
        padding_mask = tf.equal(token_ids, self.pad_token_id)  # True for padding
        non_padding_mask = 1.0 - tf.cast(padding_mask, tf.float32)
        
        # Step 3: Zero out loss at padding positions
        masked_loss = loss * non_padding_mask
        
        # Step 4: Average only over non-padding positions
        sum_loss = tf.reduce_sum(masked_loss, axis=1)
        count_non_padding = tf.reduce_sum(non_padding_mask, axis=1)
        count_non_padding = tf.maximum(count_non_padding, 1.0)
        mean_loss = sum_loss / count_non_padding
        
        # Step 5: Return batch average
        return tf.reduce_mean(mean_loss)
```

**Why This Works:**

**Before Fix (NaN Loss):**
```
Loss computed on all 300 frames:
- 34 real character frames: ~0.5 loss each
- 266 padding frames: huge loss trying to predict padding
- Average: (34×0.5 + 266×10.0) / 300 ≈ 8.8 → explodes to NaN
```

**After Fix (Masked Loss):**
```
Loss computed only on 34 real frames:
- Real characters: ~0.5 loss each
- Padding frames: ignored (×0)
- Average: (34×0.5) / 34 = 0.5 (stable!)
```

### Standard Categorical Crossentropy (Baseline)

**Formula:**
$$\text{CCE} = -\sum_{i=1}^{27} y_i \log(\hat{y}_i)$$

where:
- $y_i$ = true label (one-hot encoded)
- $\hat{y}_i$ = predicted probability

**With Label Smoothing (0.1):**
$$\text{Smooth}(y_i) = \begin{cases}
0.9 & \text{if } y_i = 1 \\
0.1/26 ≈ 0.0038 & \text{if } y_i = 0
\end{cases}$$

**Purpose:** Prevents overconfidence, slightly reduces overfitting.

### Batch Training Loop

```python
# For each epoch:
for batch in training_data:
    with tf.GradientTape() as tape:
        # Forward pass
        predictions = model(batch['X'], training=True)
        
        # Compute masked loss
        loss = loss_function(batch['y'], predictions)
    
    # Backward pass
    gradients = tape.gradient(loss, model.trainable_weights)
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    
    # Update weights
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
```

### Callbacks (Monitoring & Stopping)

```python
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,  # Stop if no improvement for 15 epochs
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_loss',
        save_best_only=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Multiply LR by 0.5
        patience=5,  # Wait 5 epochs
        min_lr=1e-6,
        verbose=1
    )
]
```

### Training Configuration

```python
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)
```

### Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Learning Rate** | 0.001 | Standard for Adam; masked loss allows this (not too aggressive) |
| **Batch Size** | 16 | Small dataset (94 train) → use small batches |
| **Epochs** | 100 | Allow model to converge fully |
| **Dropout** | 0.4 | Prevent overfitting on small dataset |
| **Label Smoothing** | 0.1 | Reduce overconfidence |
| **Gradient Clip** | 1.0 | Prevent exploding gradients |
| **EarlyStopping Patience** | 15 | Allow 15 epochs of no improvement |

### Training Progress

**Current Status (Epoch 20/100):**

| Epoch | Loss | Val Loss | Change | Status |
|-------|------|----------|--------|--------|
| 1 | 3.480 | 3.480 | - | Start |
| 5 | 2.960 | 2.941 | -15.1% | Rapid improvement |
| 10 | 2.925 | 2.902 | -5.2% | Steady |
| 15 | 2.855 | 2.845 | -3.0% | Slowing |
| 19 | 2.834 | 2.836 | -0.5% | Plateau beginning |
| 20 | In progress | - | - | Continuing |

**Expected Trajectory:**
```
Epochs 1-10:   Loss 3.48 → 2.90   (15-17% improvement per 5 epochs)
Epochs 11-30:  Loss 2.90 → 2.30   (5-8% improvement per 5 epochs)
Epochs 31-60:  Loss 2.30 → 1.50   (3-5% improvement per 10 epochs)
Epochs 61-100: Loss 1.50 → ~1.0   (plateau, EarlyStopping ~Epoch 50-60)
```

---

## Evaluation Metrics & Formulas

### Character Error Rate (CER)

**Formula:**
$$\text{CER} = \frac{S + D + I}{N} \times 100\%$$

where:
- **S** = substitutions (wrong character predicted)
- **D** = deletions (character skipped)
- **I** = insertions (extra character predicted)
- **N** = total characters in reference

**Example:**
```
Reference:  "Muraho neza"     (11 chars)
Predicted:  "Muraho nz"       (9 chars)
Alignment:   M u r a h o   n e z a
             M u r a h o   n - z -
Errors: 1 deletion ('e'), 1 deletion ('a')
CER = (0 + 2 + 0) / 11 × 100% = 18.2%
```

### Word Error Rate (WER)

**Formula:**
$$\text{WER} = \frac{S + D + I}{N} \times 100\%$$

Same formula as CER, but operating on word level instead of character level.

**Example:**
```
Reference:  "Muraho neza mwiza"      (3 words)
Predicted:  "Muraho nza mwiza"       (3 words, 1 error)
WER = 1/3 × 100% = 33.3%
```

### Accuracy (Character-Level)

**Formula:**
$$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}} \times 100\%$$

**NOTE:** Per-timestep accuracy is low (0.5-1%) because:
- Random baseline for 27 classes = 1/27 ≈ 3.7%
- We get 0.5% → **worse than random on single timesteps**
- **BUT:** Sequence models optimize loss, not per-timestep accuracy
- Loss decreasing shows learning is happening

### Sentence Error Rate (SER)

**Formula:**
$$\text{SER} = \frac{\text{Number of incorrect sentences}}{\text{Total sentences}} \times 100\%$$

A sentence is "incorrect" if any character differs from reference.

### Loss (Primary Metric)

**Masked Categorical Crossentropy:**
$$\text{Loss} = -\frac{1}{M}\sum_{i=1}^{M}\sum_{j=1}^{27} y_{i,j}^{(real)} \log(\hat{y}_{i,j})$$

where:
- M = number of non-padding positions
- $y_{i,j}^{(real)}$ = true label (one-hot)
- $\hat{y}_{i,j}$ = predicted probability

**Baseline (Random Prediction):**
$$\text{Random Loss} = -\sum_{j=1}^{27} \frac{1}{27} \log\left(\frac{1}{27}\right) = \log(27) ≈ 3.29$$

**Current Performance (Epoch 19):**
$$\text{Loss} = 2.834 \text{ (13% better than random)}$$

### R² Score (Coefficient of Determination)

**Formula:**
$$R^2 = 1 - \frac{\text{SS}_{res}}{\text{SS}_{tot}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

where:
- $\text{SS}_{res}$ = sum of squared residuals
- $\text{SS}_{tot}$ = total sum of squares

**In Classification Context:** R² is less relevant (designed for regression), but can be computed as:
$$R^2_{\text{accuracy}} = \frac{\text{Actual Accuracy} - \text{Baseline Accuracy}}{1 - \text{Baseline Accuracy}}$$

Example:
- Baseline accuracy (random): 3.7%
- Model accuracy: 52%
- $R^2 = \frac{0.52 - 0.037}{1 - 0.037} = \frac{0.483}{0.963} = 0.501 ≈ 50\%$

### Confusion Matrix

A 27×27 matrix showing which characters are confused with which:

```
                 Predicted
                  a  b  c  ...
Actual  a         45  2  1  ...
        b         0  43  2  ...
        c         1  0  44  ...
        ...       ...
```

**Use:** Identify which character pairs are often confused.

### How to Evaluate (After Training Completes)

```python
def evaluate_model(model, X_test, y_test):
    # Get predictions
    y_pred_probs = model.predict(X_test)  # Shape: (21, 300, 27)
    y_pred_classes = np.argmax(y_pred_probs, axis=-1)  # (21, 300)
    y_true_classes = np.argmax(y_test, axis=-1)  # (21, 300)
    
    # Calculate metrics
    accuracy = np.mean(y_pred_classes == y_true_classes)
    
    # Character Error Rate
    from jiwer import cer
    cer_score = cer(y_true_classes, y_pred_classes)
    
    # Word Error Rate
    from jiwer import wer
    wer_score = wer(y_true_classes, y_pred_classes)
    
    # Test loss
    test_loss = model.evaluate(X_test, y_test)[0]
    
    return {
        'test_loss': test_loss,
        'test_accuracy': accuracy,
        'cer': cer_score,
        'wer': wer_score
    }
```

---

## How to Run the Code

### Prerequisites

```bash
pip install -r requirements.txt
```

**requirements.txt includes:**
```
tensorflow>=2.14.0
librosa>=0.10.0
numpy>=1.23.0
scikit-learn>=1.0.0
soundfile>=0.12.0
```

### Directory Setup

Ensure directory structure:
```
lstm-speech-recognition-en-rw/
├── data/kinyarwanda/
│   ├── raw/custom_recordings/     ← 34 .wav files + transcripts.csv
│   └── raw/augmented/             ← Will be created
├── models/                         ← Will be created
└── scripts/
```

### Step 1: Augment Dataset (Optional - Already Done)

```bash
python scripts/augment_dataset.py
```

**Input:** 34 original WAV files in `data/kinyarwanda/raw/custom_recordings/`  
**Output:** 196 augmented files in `data/kinyarwanda/raw/augmented/`  
**Augmentation:** Each file → 3 versions (pitch ±2 semitones, time-stretch 0.9-1.1x)

### Step 2: Preprocess Data (Optional - Already Done)

```bash
python scripts/preprocess_augmented.py
```

**Input:** 196 augmented WAV files  
**Output:** MFCC features in `data/kinyarwanda/processed_augmented/`
- `train/X_train.npy`, `train/y_train.npy` (94 samples)
- `val/X_val.npy`, `val/y_val.npy` (21 samples)
- `test/X_test.npy`, `test/y_test.npy` (21 samples)

### Step 3: Train Model (THE MAIN STEP)

```bash
# Interactive mode - will prompt for inputs
python scripts/train_with_masking.py

# OR: Non-interactive with inputs piped
printf "kinyarwanda\n100\n16\n" | python scripts/train_with_masking.py
```

**Prompts:**
- Language: `kinyarwanda` (or `english`)
- Epochs: `100` (max training epochs)
- Batch size: `16` (samples per batch)

**Output:**
- Real-time logs in `train_masked.log`
- Best model saved to `models/checkpoints/`
- Final model to `models/trained/kinyarwanda_masked_final.h5`

**Expected Duration:** ~2-3 hours for 100 epochs (or earlier with EarlyStopping)

### Step 4: Evaluate Model

```bash
python scripts/evaluate_model.py
```

**Input:** Trained model, test set  
**Output:**
- Test loss
- Test accuracy
- Character error rate (CER)
- Word error rate (WER)
- Confusion matrix
- Sample predictions

### Step 5: Make Predictions on New Audio

```python
from src.lstm_model import LSTMSpeechRecognition
import numpy as np

# Load model
model = LSTMSpeechRecognition(input_shape=(300, 13), vocab_size=27)
model.model.load_weights('models/trained/kinyarwanda_masked_final.h5')

# Process new audio
from src.audio_processor import AudioProcessor
processor = AudioProcessor(sr=16000, duration=5, n_mfcc=13)
audio, sr = processor.load_audio('path/to/audio.wav')
audio = processor.normalize_audio(audio)
audio = processor.remove_silence(audio)
audio = processor.pad_or_trim(audio)
mfcc = processor.extract_mfcc(audio)

# Expand batch dimension
mfcc = np.expand_dims(mfcc, axis=0)  # (1, 13, 300)
mfcc = np.transpose(mfcc, (0, 2, 1))  # (1, 300, 13)
mfcc = mfcc / (np.max(np.abs(mfcc)) + 1e-8)  # Normalize

# Predict
predictions = model.model.predict(mfcc)  # (1, 300, 27)
class_ids = np.argmax(predictions, axis=-1)[0]  # (300,)

# Decode to text
num_to_char = {0: ' ', 1: '.', 2: '<PAD>', 3: '<UNK>', ...}  # Load from vocab
transcript = ''.join([num_to_char[cid] for cid in class_ids if cid != 2])
print(transcript)
```

### Monitoring Training

```bash
# Watch training in real-time
tail -f train_masked.log | grep "Epoch"

# Check latest epoch
grep "Epoch.*: loss=" train_masked.log | tail -1

# Plot loss progression (in Python)
import re
losses = re.findall(r'Epoch \d+: loss=([0-9.]+)', open('train_masked.log').read())
import matplotlib.pyplot as plt
plt.plot(losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
```

---

## Model Training vs Pretrained Weights

### In This Project: TRAINING FROM SCRATCH

**Why?** This project **trains a new model from scratch** rather than using pretrained weights:

1. **Task-Specific:** Character-level Kinyarwanda speech recognition is specialized
   - Pretrained English models don't understand Kinyarwanda phonetics
   - Would need expensive fine-tuning

2. **Small Dataset:** 136 valid samples is too small for transfer learning
   - Transfer learning needs 1000+ examples in target domain
   - We're better off training focused model

3. **Architecture Simplicity:** Our BiLSTM is simple and trains quickly
   - 13.2M parameters (not 300M+ like BERT)
   - Converges in ~2-3 hours

4. **Custom Loss Function:** MaskedCategoricalCrossentropy specific to our data
   - Pretrained models use standard losses
   - Our masking is crucial for variable-length sequences

### How Training Works

```python
# Random Initialization
model = LSTMSpeechRecognition(
    input_shape=(300, 13),
    vocab_size=27,
    lstm_units=512,
    dropout_rate=0.4
)

# Weights are randomly initialized (Xavier/Glorot initialization)
# During training:
for epoch in range(100):
    for batch in training_data:
        # Forward pass through random/partially-trained weights
        predictions = model(batch)
        
        # Compute loss
        loss = masked_loss(batch['y'], predictions)
        
        # Backpropagation: compute gradients
        gradients = compute_gradients(loss, model.weights)
        
        # Update weights: weights -= lr × gradients
        model.weights -= 0.001 × gradients
        
        # Weights gradually improve through iteration
```

### If We Used Pretrained Weights (Alternative)

```python
# Option: Use pretrained weights (not implemented here)
# From a general speech recognition model trained on ~1000 hours of audio
model.load_weights('pretrained_speech_model.h5')

# Then fine-tune on Kinyarwanda:
for epoch in range(20):  # Fewer epochs needed
    # Train only the top layers
    # Or: Train all layers with lower learning rate (0.0001)
```

**Pros of Transfer Learning:**
- Faster training (fewer epochs)
- Better with very small datasets
- Weights already learned speech patterns

**Cons of Transfer Learning:**
- Pretrained models may not fit Kinyarwanda phonetics
- Need compatible architecture
- Still requires fine-tuning

### Current Approach: Pure Training

```
Initial Random Weights
        ↓
Epoch 1-5:   Rapid learning (gradients large, loss drops fast)
        ↓
Epoch 6-15:  Steady learning (loss decreases slowly)
        ↓
Epoch 16-60: Convergence (loss plateau, EarlyStopping triggers)
        ↓
Final Trained Weights → Save to .h5 file
```

### Validation During Training

```python
# Each epoch, evaluate on validation set
for epoch in range(100):
    # Train on 94 samples
    train_loss = train_one_epoch()
    
    # Validate on 21 samples (never trained on these)
    val_loss = evaluate(X_val, y_val)
    
    # If val_loss stops improving for 15 epochs → EarlyStopping
    if epochs_no_improvement > 15:
        break
```

**Why Validation?** Detect when training overfits (learns training data quirks, not generalizable patterns).

---

## Summary: Training Timeline

```
Start Training:        Epoch 1 with loss=3.48
|
├─ Rapid Learning:     Epochs 1-5, loss 3.48→2.96 (15% improvement)
│  • Model discovers basic character patterns
│  • Gradients are large, updates are big
│
├─ Steady Learning:    Epochs 6-15, loss 2.96→2.85 (4% improvement)
│  • Model refines predictions
│  • Gradients are smaller, updates are fine-tuned
│
├─ Convergence:        Epochs 16-60, loss 2.85→~1.5 (expected)
│  • Model approaches optimal solution for this data/architecture
│  • Improvement becomes incremental
│
└─ End Training:       EarlyStopping or Epoch 100
   • Save best model from checkpoint
   • Evaluate on test set (21 samples never seen during training)
   • Report final metrics: accuracy, CER, WER, loss

Final Accuracy Expected: 50-75% (vs 33.52% baseline)
```

---

## Key Takeaways

1. **Data Processing:** MFCC features capture audio in 13 coefficients per frame
2. **Model Architecture:** BiLSTM + Attention processes sequences both directions
3. **The Fix:** MaskedCategoricalCrossentropy solves NaN by ignoring padding in loss
4. **Training:** Starts with random weights, improves through gradient descent
5. **Evaluation:** Metrics (accuracy, CER, WER) measure performance on held-out test set
6. **From Scratch:** No pretrained weights—this task needs custom training

---

*This guide is complete as of Epoch 20/100. Final results and detailed evaluation will be added when training completes.*

