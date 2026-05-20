# LSTM Speech Recognition - Quick Start Guide

## Complete 5-Step Training Pipeline

### Step 1: Setup Environment (5 minutes)

```bash
# Create project directory
mkdir lstm-speech-recognition-en-rw
cd lstm-speech-recognition-en-rw

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Verify installation:**
```bash
python -c "import tensorflow, librosa, pyaudio; print('✓ All packages installed')"
```

---

### Step 2: Record Custom Audio Dataset (30 minutes - 2 hours)

**Start Recording:**
```bash
python scripts/record_audio.py
```

**Interactive Prompts:**
```
Language (english/kinyarwanda): kinyarwanda
Duration per sample (seconds, default 5): 5
Number of samples to record (default 10): 50
Enter speaker ID (or press Enter for 'user'): speaker1
```

**What Happens:**
- You'll see prompts like: `"Muraho, wacu mwire."` (Hello, how are you?)
- Read the sentence clearly
- Audio saved to: `data/kinyarwanda/raw/custom_recordings/`
- Transcripts saved to: `data/kinyarwanda/raw/custom_recordings/transcripts.csv`

**Recording Tips:**
- ✓ Quiet room (minimize background noise)
- ✓ Clear pronunciation (important for training)
- ✓ Normal speaking speed
- ✓ 50+ samples recommended (start with 10-20 for testing)

**Output Structure:**
```
data/kinyarwanda/raw/custom_recordings/
├── kinyarwanda_speaker1_20260519_143022_1.wav
├── kinyarwanda_speaker1_20260519_143522_2.wav
├── ... (50 files total)
└── transcripts.csv
```

---

### Step 3: Preprocess Audio Data (10-30 minutes)

**Convert Audio to Training Features:**
```bash
python scripts/preprocess_data.py
```

**Interactive Prompts:**
```
Language (english/kinyarwanda): kinyarwanda
```

**What Happens:**
1. Scans all audio files in `data/kinyarwanda/raw/`
2. Normalizes audio levels
3. Removes silence automatically
4. Extracts MFCC (audio features)
5. Creates character vocabulary
6. Splits into train/val/test (70/15/15)
7. Saves to `data/kinyarwanda/processed/`

**Processing Pipeline:**
```
Raw Audio (wav files)
    ↓
Load & Normalize
    ↓
Remove Silence
    ↓
Extract MFCC Features
    ↓
Pad to Fixed Length
    ↓
Create Vocabulary
    ↓
Split Data
    ↓
Save (train/val/test folders)
```

**Output Files:**
```
data/kinyarwanda/processed/
├── train/
│   ├── X_train.npy (training features)
│   └── y_train.npy (training labels)
├── val/
│   ├── X_val.npy (validation features)
│   └── y_val.npy (validation labels)
├── test/
│   ├── X_test.npy (test features)
│   └── y_test.npy (test labels)
└── vocabulary.json (character mappings)
```

**Example Output:**
```
Dataset loaded!
Train: X=(35, 300, 13), y=(35, 200)
Val: X=(8, 300, 13), y=(8, 200)
Test: X=(7, 300, 13), y=(7, 200)
```

---

### Step 4: Train LSTM Model (2-6 hours)

**Start Training:**
```bash
python scripts/train_model.py
```

**Interactive Prompts:**
```
Language (english/kinyarwanda/both): kinyarwanda
Model type (encoder_decoder/bidirectional/ctc) [bidirectional]: bidirectional
Epochs [50]: 50
Batch size [32]: 32
```

**What Happens:**
- Model trains on preprocessed data
- Real-time accuracy/loss monitoring
- Saves checkpoints every epoch
- Auto-saves best performing model
- Training logs stored

**Model Architecture (Bidirectional):**
```
Input (300, 13)
    ↓
BiLSTM Layer 1 (512 units) + Dropout(0.3)
    ↓
BiLSTM Layer 2 (512 units) + Dropout(0.3)
    ↓
Multi-Head Attention (8 heads)
    ↓
BiLSTM Layer 3 (256 units) + Dropout(0.3)
    ↓
Dense Output (128 softmax)
    ↓
Character Predictions
```

**Example Training Output:**
```
Epoch 1/50
35/35 [==============================] - 45s - loss: 2.1234 - accuracy: 0.3456 - val_loss: 1.8901 - val_accuracy: 0.4567
Epoch 2/50
35/35 [==============================] - 43s - loss: 1.8765 - accuracy: 0.4678 - val_loss: 1.6234 - val_accuracy: 0.5234
...
Epoch 50/50
35/35 [==============================] - 42s - loss: 0.2134 - accuracy: 0.8923 - val_loss: 0.3456 - val_accuracy: 0.8712
```

**Saved Outputs:**
```
models/
├── checkpoints/kinyarwanda_bidirectional_20260519_143022/
│   ├── best_model.h5 (best checkpoint)
│   └── logs/ (TensorBoard logs)
└── trained/
    └── kinyarwanda_bidirectional_final.h5 (final model)
```

**GPU Acceleration (Optional):**
If you have NVIDIA GPU:
```bash
pip install tensorflow-gpu
# CUDA 11.8+ required
```

---

### Step 5: Test & Use the Model (5-15 minutes)

**Transcribe an Audio File:**
```bash
python src/inference.py
```

**Interactive Prompts:**
```
Language (english/kinyarwanda) [english]: kinyarwanda
Model path [models/trained/english_bidirectional_final.h5]: models/trained/kinyarwanda_bidirectional_final.h5
Mode (file/realtime) [file]: file
Audio file path: /path/to/audio.wav
```

**Output Example:**
```
Transcript: muraho wacu mwire
Avg Confidence: 82.34%
```

**Real-Time Transcription:**
```
Mode (file/realtime) [file]: realtime
Recording duration (seconds) [10]: 10
# Speak into microphone for 10 seconds...
Final Transcript: ndagukunda igitabo
```

---

## Model Architecture Options

### 1. Encoder-Decoder LSTM (Best Accuracy)
- **Recommended for:** Maximum accuracy (target 85%+)
- **Training time:** 4-6 hours (100 hours data)
- **Architecture:** 2 encoder layers + Bidirectional + 2 decoder layers
```bash
Model type: encoder_decoder
```

### 2. Bidirectional LSTM (Balanced)
- **Recommended for:** Good accuracy + reasonable speed
- **Training time:** 2-3 hours (100 hours data)
- **Architecture:** 3 BiLSTM + Multi-head attention
```bash
Model type: bidirectional
```

### 3. CTC LSTM (Real-Time)
- **Recommended for:** Unsegmented data, real-time streaming
- **Training time:** 1-2 hours
- **Architecture:** Bidirectional + CTC loss
```bash
Model type: ctc
```

---

## Common Workflows

### Workflow A: Quick Test (30 min)
1. Record 10 samples (~5 min)
2. Preprocess (~5 min)
3. Train for 5 epochs (~10 min)
4. Test inference (~2 min)

```bash
# Record
python scripts/record_audio.py  # 10 samples, 5 sec each
# Preprocess
python scripts/preprocess_data.py
# Train light
python scripts/train_model.py  # 5 epochs
# Test
python src/inference.py
```

### Workflow B: Production Model (4-6 hours)
1. Record 50-100 samples (~1-2 hours)
2. Preprocess (~15 min)
3. Train 50 epochs (~2-3 hours)
4. Evaluate & iterate

```bash
# Record
python scripts/record_audio.py  # 50+ samples
# Preprocess
python scripts/preprocess_data.py
# Train full
python scripts/train_model.py  # 50 epochs
# Test
python src/inference.py
```

### Workflow C: Bilingual Models (8-12 hours)
1. Record English + Kinyarwanda (2-4 hours)
2. Preprocess both (30 min)
3. Train both models in parallel (4-6 hours)
4. Deploy both

```bash
# Record English
python scripts/record_audio.py  # language: english, 50+ samples
# Record Kinyarwanda
python scripts/record_audio.py  # language: kinyarwanda, 50+ samples
# Preprocess
python scripts/preprocess_data.py  # english
python scripts/preprocess_data.py  # kinyarwanda
# Train both
python scripts/train_model.py  # language: both
```

---

## Troubleshooting

### Issue: "No module named 'pyaudio'"
**Solution:**
```bash
# Windows (might need Visual C++ build tools)
pip install pipwin
pipwin install pyaudio

# Mac
brew install portaudio
pip install pyaudio

# Linux
sudo apt-get install portaudio19-dev
pip install pyaudio
```

### Issue: "No audio devices detected"
**Solution:**
```bash
# Check available devices
python -c "import pyaudio; p = pyaudio.PyAudio(); [print(f'{i}: {p.get_device_info_by_index(i)[\"name\"]}') for i in range(p.get_device_count())]"
```

### Issue: Out of memory during training
**Solution:**
```bash
# In train_model.py, reduce batch size
Batch size: 16  # was 32
# Or reduce LSTM units in constants.py
LSTM_UNITS = 256  # was 512
```

### Issue: Low accuracy on Kinyarwanda
**Reason:** Limited training data
**Solution:**
1. Record more Kinyarwanda audio (target 50+ hours)
2. Use data augmentation
3. Increase epochs (100+ instead of 50)
4. Use encoder_decoder model (higher accuracy)

---

## Performance Expectations

### Training Time per Language
| Dataset Size | Model | GPU | CPU |
|-------------|-------|-----|-----|
| 10 hours | Bidirectional | 30 min | 1.5 hours |
| 50 hours | Bidirectional | 1.5 hours | 8 hours |
| 100 hours | Encoder-Decoder | 2 hours | 10 hours |

### Expected Accuracy
| Language | Dataset | Accuracy |
|----------|---------|----------|
| English | 100 hours | 85-92% |
| English | 10 hours | 70-80% |
| Kinyarwanda | 50 hours | 75-85% |
| Kinyarwanda | 10 hours | 60-70% |

---

## Next Steps

1. **Complete the 5-step pipeline** (Steps 1-5 above)
2. **Evaluate your model** using test_inference.py
3. **Collect feedback** - record more data for domains where accuracy is low
4. **Fine-tune** - train for more epochs or adjust learning rate
5. **Deploy** - integrate with Digital Library application

---

## Integration with Digital Library

Once model is trained:

```python
# In your React backend
from src.inference import STTInference

stt = STTInference('models/trained/kinyarwanda_bidirectional_final.h5', 'kinyarwanda')

@app.post("/api/transcribe")
async def transcribe(audio: UploadFile):
    transcript = stt.transcribe_audio_file(audio.file)
    return {"transcript": transcript}
```

---

## Resources

- **TensorFlow Docs:** https://tensorflow.org/guide
- **Librosa Audio Processing:** https://librosa.org/doc/latest/
- **LSTM Tutorial:** https://colah.github.io/posts/2015-08-Understanding-LSTMs/

---

**Good luck! Your speech recognition model is just 5 steps away! 🎙️**
