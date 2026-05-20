# LSTM Speech Recognition System - Complete Overview

## What You Now Have

A professional, complete LSTM speech-to-text system with three integrated components:

### 1️⃣ Recording System (`record_audio.py`)
**Purpose:** Capture your voice and automatically organize training data

**How it works:**
- Interactive prompts guide you to read sentences
- Records audio at 16kHz (speech-optimized)
- Automatically saves with timestamps
- Creates metadata CSV with transcripts
- Supports both English and Kinyarwanda

**Key Features:**
- ✓ Multiple speaker support
- ✓ Configurable duration & sample count
- ✓ Professional recording workflow
- ✓ Metadata tracking

**Input:** Your voice + phonetically diverse sentences
**Output:** Organized WAV files + transcripts.csv

---

### 2️⃣ Preprocessing Pipeline (`preprocess_data.py`)
**Purpose:** Convert raw audio into machine learning features

**How it works:**
```
Raw Audio Files (.wav)
    ↓
Load & Normalize Levels
    ↓
Remove Silence (audio below -40dB)
    ↓
Extract MFCC Features (13 coefficients)
    ↓
Pad/Truncate to Fixed Length (300 frames)
    ↓
Create Character Vocabulary (128 unique chars)
    ↓
Encode Transcripts (text → numbers)
    ↓
Split Data (70% train, 15% val, 15% test)
    ↓
Save as NumPy Arrays (.npy files)
```

**Key Features:**
- ✓ Multi-source support (Kaggle, Digital Umuganda, Custom)
- ✓ Automatic feature extraction
- ✓ Data augmentation (pitch shift, time stretch)
- ✓ Vocabulary management
- ✓ Train/validation/test split

**Input:** Raw audio files + transcripts
**Output:** 
- `X_train.npy`, `y_train.npy` (training data)
- `X_val.npy`, `y_val.npy` (validation data)
- `X_test.npy`, `y_test.npy` (test data)
- `vocabulary.json` (character mappings)

---

### 3️⃣ LSTM Model Architecture (`lstm_model.py`)
**Purpose:** Neural network for speech-to-text translation

**Three Model Options:**

#### Option A: Encoder-Decoder LSTM
```
Input Audio Features (300, 13)
    ↓
[Encoder] 2 LSTM layers (512 units)
    ↓
Bidirectional LSTM (256 units)
    ↓
[Decoder] 2 LSTM layers (512 units)
    ↓
Dense Output Layer (softmax)
    ↓
Character Predictions
```
- **Best for:** Maximum accuracy (target 85-92%)
- **Training time:** 2-4 hours (100 hours data)
- **Params:** ~3M

#### Option B: Bidirectional LSTM (RECOMMENDED)
```
Input Audio Features (300, 13)
    ↓
BiLSTM Layer 1 (512 units)
    ↓
BiLSTM Layer 2 (512 units)
    ↓
Multi-Head Attention (8 heads)
    ↓
BiLSTM Layer 3 (256 units)
    ↓
Dense Output Layer (softmax)
    ↓
Character Predictions
```
- **Best for:** Balanced accuracy & speed
- **Training time:** 1-2 hours (100 hours data)
- **Params:** ~2.5M

#### Option C: CTC LSTM
```
Input Audio Features (300, 13)
    ↓
Dense Layer (256 units)
    ↓
BiLSTM Layer 1 (512 units)
    ↓
BiLSTM Layer 2 (256 units)
    ↓
Dense Output Layer (softmax)
    ↓
CTC Loss
```
- **Best for:** Real-time streaming, no alignment needed
- **Training time:** 30 min - 1 hour
- **Params:** ~1.5M

---

## Training Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     LSTM Training Pipeline                       │
└─────────────────────────────────────────────────────────────────┘

Step 1: RECORD AUDIO
├─ Run: python scripts/record_audio.py
├─ Action: Speak into microphone (50+ samples)
├─ Output: data/{lang}/raw/custom_recordings/*.wav
└─ Duration: 30 min - 2 hours

Step 2: PREPROCESS DATA
├─ Run: python scripts/preprocess_data.py
├─ Action: Convert audio → features
├─ Process: Normalize → Remove Silence → Extract MFCC → Encode
├─ Output: data/{lang}/processed/{train|val|test}/*.npy
└─ Duration: 10-30 min

Step 3: TRAIN MODEL
├─ Run: python scripts/train_model.py
├─ Action: Feed features to LSTM
├─ Monitor: Loss/accuracy per epoch
├─ Saves: Best checkpoint automatically
├─ Output: models/trained/{lang}_bidirectional_final.h5
└─ Duration: 2-6 hours (depends on data size)

Step 4: EVALUATE MODEL
├─ Run: python src/inference.py
├─ Action: Test on held-out test set
├─ Metrics: Accuracy, confidence scores
└─ Output: Transcript + confidence

Step 5: DEPLOY
├─ Use: Integrate with Digital Library
├─ Method: Flask/FastAPI endpoint
├─ Real-time: WebSocket streaming
└─ Status: Production-ready
```

---

## Data Flow Diagram

```
┌──────────────────────┐
│   Your Voice         │
│   (microphone)       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────┐
│   record_audio.py        │
│   ├─ Record 5-sec clips  │
│   ├─ Normalize levels    │
│   └─ Save + transcript   │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────────────┐
│  data/{lang}/raw/               │
│  └─ custom_recordings/          │
│     ├─ audio_001.wav           │
│     ├─ audio_002.wav           │
│     └─ transcripts.csv         │
└──────────┬──────────────────────┘
           │
           ▼
┌──────────────────────────┐
│  preprocess_data.py      │
│  ├─ Load audio          │
│  ├─ Extract MFCC        │
│  ├─ Pad features        │
│  ├─ Encode text         │
│  └─ Split train/val/test│
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────────────┐
│  data/{lang}/processed/         │
│  ├─ train/                       │
│  │  ├─ X_train.npy (features)  │
│  │  └─ y_train.npy (labels)    │
│  ├─ val/                         │
│  ├─ test/                        │
│  └─ vocabulary.json             │
└──────────┬──────────────────────┘
           │
           ▼
┌──────────────────────────┐
│  train_model.py          │
│  ├─ Load data           │
│  ├─ Build LSTM model    │
│  ├─ Train 50 epochs     │
│  ├─ Monitor loss/acc    │
│  └─ Save best model     │
└──────────┬───────────────┘
           │
           ▼
┌────────────────────────────────────┐
│  models/trained/                  │
│  └─ {lang}_bidirectional_final.h5 │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────┐
│  inference.py            │
│  ├─ Load model          │
│  ├─ Transcribe audio    │
│  ├─ Decode predictions  │
│  └─ Return transcript   │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Output: Text            │
│  "Muraho, wacu mwire"    │
│  (Confidence: 85.2%)     │
└──────────────────────────┘
```

---

## Feature Extraction Process

### MFCC (Mel-Frequency Cepstral Coefficients)

Why MFCC? It mimics how humans hear - sensitive to frequency ranges important for speech, less sensitive to noise.

```
Raw Audio Waveform (16kHz)
    ↓
[FFT] Convert to frequency domain
    ↓
[Mel Scale] Map frequencies to human perception
    ↓
[Log] Compress amplitude
    ↓
[DCT] Extract important patterns
    ↓
13 MFCC Coefficients
```

**What gets extracted:**
- 13 MFCC values per 20ms audio frame
- Represents "shape" of speech spectrum
- 300 frames × 13 coefficients = input shape (300, 13)

**Example:**
```
Frame 1: [0.234, -0.567, 0.123, ..., 0.456]  ← 13 values
Frame 2: [0.245, -0.578, 0.134, ..., 0.467]  ← 13 values
Frame 3: [0.256, -0.589, 0.145, ..., 0.478]  ← 13 values
...
Frame 300: [0.001, 0.002, 0.003, ..., 0.004]  ← 13 values
```

---

## Character Encoding

### How Text Becomes Numbers

The model learns to predict one character at a time:

```
Vocabulary created from transcripts:
{
  '<PAD>': 0,       ← padding token
  '<UNK>': 1,       ← unknown character
  ' ': 2,           ← space
  'a': 3,
  'b': 4,
  ...
  'z': 29,
  'à': 30,
  ...
}

Text: "Muraho"
  ↓
Encoded: [14, 20, 1, 0, 7, 13]
  ↓
Decoded: "Muraho"
```

---

## Vocabulary Management

### Building Vocabulary

```
process_transcript("Muraho") 
  → unique_chars = {'m', 'u', 'r', 'a', 'h', 'o'}

process_transcript("wacu mwire")
  → unique_chars += {'w', 'c', ' ', 'i', 'e'}

All transcripts processed
  → Final vocab = sorted unique characters
  → char_to_num = {char: index for char, index in enumerate(vocab)}
  → num_to_char = {index: char for char, index in char_to_num.items()}
```

**Saved in:** `data/{language}/vocabulary.json`
```json
{
  "char_to_num": {
    " ": 2,
    "'": 3,
    "a": 4,
    ...
  },
  "num_to_char": {
    "2": " ",
    "3": "'",
    "4": "a",
    ...
  }
}
```

---

## Training Process

### Per-Epoch Training

```
Epoch 1:
├─ Shuffle training data
├─ For each batch of 32 samples:
│   ├─ Forward pass through LSTM
│   ├─ Compute cross-entropy loss
│   ├─ Backward pass (backpropagation)
│   └─ Update weights (Adam optimizer)
├─ Evaluate on validation set
├─ Log loss, accuracy
└─ Save checkpoint if best so far

Epoch 2-50: Repeat
```

### Key Metrics

- **Loss:** Cross-entropy (how wrong predictions are)
  - Lower is better
  - Starts ~2.0, should drop to <0.5

- **Accuracy:** % characters predicted correctly
  - Higher is better
  - Target: 80%+ on validation

### Early Stopping

If validation loss doesn't improve for 5 epochs → stop training automatically
(Prevents overfitting)

---

## File Organization Reference

```
lstm-speech-recognition-en-rw/
│
├── data/
│   ├── english/
│   │   ├── raw/
│   │   │   ├── kaggle_speech/       (← Download from Kaggle)
│   │   │   ├── podcasts/            (← Optional)
│   │   │   └── custom_recordings/   (← Your recordings)
│   │   └── processed/               (← Auto-generated)
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   │
│   └── kinyarwanda/                 (Same structure)
│
├── src/
│   ├── lstm_model.py               (LSTM architectures)
│   ├── audio_processor.py          (Feature extraction)
│   ├── inference.py                (Transcription)
│   └── constants.py                (Configuration)
│
├── scripts/
│   ├── record_audio.py             (Interactive recording)
│   ├── preprocess_data.py          (Feature extraction)
│   └── train_model.py              (Model training)
│
├── models/
│   ├── checkpoints/                (Intermediate saves)
│   └── trained/                    (Final models)
│       ├── english_bidirectional_final.h5
│       └── kinyarwanda_bidirectional_final.h5
│
├── requirements.txt                (Python dependencies)
├── README.md                       (Full documentation)
├── QUICK_START.md                  (Step-by-step guide)
└── SYSTEM_OVERVIEW.md              (This file)
```

---

## Integration with Digital Library

### Connection Points

```
┌──────────────────────────────────────┐
│  Digital Library Frontend            │
│  (React, Upload Book)                │
└──────────────┬───────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │  /api/transcribe     │
    │  (POST audio file)   │
    └──────────┬───────────┘
               │
               ▼
    ┌────────────────────────────────────┐
    │  Backend API (Flask/FastAPI)       │
    │  ├─ Load LSTM model                │
    │  ├─ Process audio                  │
    │  └─ Return transcript              │
    └──────────┬───────────────────────┘
               │
               ▼
    ┌────────────────────────────────┐
    │  inference.py                  │
    │  ├─ STTInference class         │
    │  ├─ transcribe_audio_file()    │
    │  └─ Return text                │
    └──────────┬────────────────────┘
               │
               ▼
    ┌────────────────────────────────┐
    │  Response to Frontend          │
    │  {                             │
    │    "transcript": "Muraho...",   │
    │    "confidence": 0.854         │
    │  }                             │
    └────────────────────────────────┘
```

### Python Integration Code

```python
# In your Digital Library backend
from src.inference import STTInference

# Load model once at startup
stt_english = STTInference(
    'models/trained/english_bidirectional_final.h5',
    'english'
)

stt_kinyarwanda = STTInference(
    'models/trained/kinyarwanda_bidirectional_final.h5',
    'kinyarwanda'
)

@app.post("/api/transcribe")
async def transcribe(file: UploadFile, language: str):
    stt = stt_english if language == 'en' else stt_kinyarwanda
    transcript = stt.transcribe_audio_file(file.file)
    
    return {
        "transcript": transcript,
        "language": language
    }

# Real-time WebSocket
@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    
    while True:
        audio_chunk = await websocket.receive_bytes()
        # Process audio chunk
        result = stt_english.transcribe_raw_audio(audio_chunk)
        await websocket.send_json({"partial": result})
```

---

## Performance Optimization

### Faster Inference

```python
# Option 1: Reduce model size
LSTM_UNITS = 256  # was 512
EMBEDDING_DIM = 128  # was 256

# Option 2: Quantize model (TensorFlow Lite)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Option 3: Batch processing
stt.batch_transcribe(['audio1.wav', 'audio2.wav', ...])
```

### Better Accuracy

```python
# 1. More training data
# Record 50+ hours instead of 10

# 2. Longer training
EPOCHS = 100  # was 50

# 3. Better model
model_type = 'encoder_decoder'  # was bidirectional

# 4. Data augmentation
# Automatically done in preprocessing
```

---

## Troubleshooting Decision Tree

```
Problem: Low accuracy
├─ Cause 1: Not enough data
│  └─ Solution: Record more samples (target 50+ hours)
├─ Cause 2: Model too simple
│  └─ Solution: Use encoder_decoder instead of bidirectional
├─ Cause 3: Model underfitted
│  └─ Solution: Increase epochs (100+ instead of 50)
└─ Cause 4: Noisy audio
   └─ Solution: Record in quieter room

Problem: Training crashes (memory)
├─ Cause 1: Batch size too large
│  └─ Solution: Reduce batch_size from 32 to 16
├─ Cause 2: LSTM units too large
│  └─ Solution: Reduce LSTM_UNITS from 512 to 256
└─ Cause 3: Not enough GPU memory
   └─ Solution: Use CPU instead or reduce dataset

Problem: Recording fails
├─ Cause 1: PyAudio not installed
│  └─ Solution: pip install pyaudio
├─ Cause 2: No microphone access
│  └─ Solution: Check device permissions
└─ Cause 3: Wrong device selected
   └─ Solution: List devices and select correct one

Problem: Can't find model
├─ Cause 1: Model not trained yet
│  └─ Solution: Run train_model.py first
├─ Cause 2: Wrong path
│  └─ Solution: Check models/trained/ directory
└─ Cause 3: Model training failed
   └─ Solution: Check logs and error messages
```

---

## Summary

You now have a complete, professional LSTM speech-to-text system with:

✓ **Recording System** - Capture phonetically diverse training data
✓ **Preprocessing Pipeline** - Convert audio to ML-ready features
✓ **Three LSTM Models** - From simple to complex architectures
✓ **Training Script** - End-to-end model training with monitoring
✓ **Inference Engine** - Real-time & batch transcription
✓ **Complete Documentation** - README, QUICK_START, this overview

**Next Step:** Run `QUICK_START.md` - follow the 5 steps to train your first model!

Good luck! 🎙️🚀
