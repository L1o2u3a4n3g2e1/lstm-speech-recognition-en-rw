# LSTM Speech Recognition - English & Kinyarwanda

A comprehensive speech-to-text system using LSTM neural networks supporting English and Kinyarwanda languages.

## Project Structure

```
lstm-speech-recognition-en-rw/
├── data/                           # Datasets directory
│   ├── english/                    # English language data
│   │   ├── raw/                    # Raw audio files
│   │   │   ├── kaggle_speech/      # Kaggle datasets
│   │   │   ├── podcasts/           # Podcast audio
│   │   │   └── custom_recordings/  # User recordings
│   │   ├── processed/              # Preprocessed features
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── metadata.json
│   └── kinyarwanda/                # Kinyarwanda language data
│       ├── raw/
│       │   ├── digital_umuganda/
│       │   ├── huggingface/
│       │   └── custom_recordings/
│       └── processed/
├── src/                            # Source code
│   ├── lstm_model.py              # LSTM model architectures
│   ├── audio_processor.py         # Audio processing utilities
│   ├── data_loader.py             # Data loading utilities
│   ├── trainer.py                 # Training utilities
│   └── constants.py               # Configuration constants
├── scripts/                        # Standalone scripts
│   ├── record_audio.py            # Interactive recording script
│   ├── preprocess_data.py         # Data preprocessing pipeline
│   ├── train_model.py             # Model training script
│   ├── evaluate_model.py          # Model evaluation
│   └── test_inference.py          # Testing inference
├── models/                         # Trained models
│   ├── checkpoints/               # Training checkpoints
│   └── trained/                   # Final trained models
├── notebooks/                      # Jupyter notebooks
├── requirements.txt                # Python dependencies
└── README.md
```

## Installation

1. **Clone/Create project directory:**
```bash
mkdir lstm-speech-recognition-en-rw
cd lstm-speech-recognition-en-rw
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Quick Start

### Step 1: Record Custom Audio Dataset

Start by recording your own speech data in English and Kinyarwanda:

```bash
python scripts/record_audio.py
```

**What to expect:**
- Choose language: `english` or `kinyarwanda`
- Specify duration per sample: `5` seconds (recommended)
- Specify number of samples: `10` (start small, expand later)
- Read the prompts and speak clearly
- Audio saved to `data/{language}/raw/custom_recordings/`
- Transcript metadata saved to `transcripts.csv`

**Recording tips:**
- Use a quiet room
- Speak at normal speed
- Clear pronunciation helps model accuracy
- Record 50+ hours for best results (start with 10 hours)

### Step 2: Preprocess Data

Convert raw audio into MFCC features for training:

```bash
python scripts/preprocess_data.py
```

**Process:**
1. Loads audio from `data/{language}/raw/`
2. Normalizes audio levels
3. Removes silence
4. Extracts MFCC (Mel-Frequency Cepstral Coefficients) features
5. Creates character-level vocabulary
6. Splits into train/val/test (70/15/15)
7. Saves preprocessed data to `data/{language}/processed/`

### Step 3: Train Model

Train LSTM model on preprocessed data:

```bash
python scripts/train_model.py
```

**Configuration options:**
- **Language:** `english`, `kinyarwanda`, or `both`
- **Model type:**
  - `encoder_decoder` - Best for accuracy (recommended)
  - `bidirectional` - Good balance of speed/accuracy
  - `ctc` - Connectionist Temporal Classification
- **Epochs:** How many training iterations (50+ recommended)
- **Batch size:** Training batch size (32 default)

**Example:**
```
Language: english
Model type: bidirectional
Epochs: 50
Batch size: 32
```

**Training output:**
- Model checkpoints saved every epoch
- Validation metrics logged
- Best model automatically saved
- Training logs in `models/checkpoints/`
- Final model saved to `models/trained/`

### Step 4: Evaluate Model

Test model performance on held-out test set:

```bash
python scripts/test_inference.py
```

## Model Architectures

### 1. Encoder-Decoder LSTM
- 2 stacked LSTM layers (512 units each)
- Bidirectional LSTM layer (256 units)
- 2 decoder LSTM layers
- Dropout layers for regularization
- **Best for:** Maximum accuracy, longer training time

### 2. Bidirectional LSTM
- 3 Bidirectional LSTM layers
- Multi-head attention mechanism
- Lower computational cost
- **Best for:** Balanced performance, faster training

### 3. CTC LSTM
- Bidirectional LSTM with CTC loss
- No need for alignment annotations
- **Best for:** Unsegmented data, real-time processing

## Data Source Integration

### Available Datasets

#### English
1. **Librispeech** (Kaggle)
   - 1000 hours of clean audiobook data
   - High quality, well-transcribed
   - Download: https://www.kaggle.com/datasets/bhavanshiv/librispeech

2. **Common Voice** (Mozilla)
   - Multilingual, crowdsourced
   - Free and open-source
   - Download: https://commonvoice.mozilla.org/

3. **Custom Recordings**
   - Your own voice data
   - Best for domain adaptation

#### Kinyarwanda
1. **Digital Umuganda**
   - Government of Rwanda platform
   - Kinyarwanda language preservation
   - Status: Check availability

2. **Common Voice - Kinyarwanda**
   - Limited but growing dataset
   - Open-source

3. **Custom Recordings** (Recommended)
   - Record native speakers
   - Ensure phonetic diversity

### Adding Datasets

1. **Kaggle Dataset:**
   ```bash
   # Download from Kaggle, extract to:
   # data/english/raw/kaggle_speech/
   ```

2. **Custom Recordings:**
   ```bash
   python scripts/record_audio.py
   # Audio saved to: data/{language}/raw/custom_recordings/
   ```

3. **Digital Umuganda:**
   ```bash
   # Download Kinyarwanda audio
   # Extract to: data/kinyarwanda/raw/digital_umuganda/
   ```

## Workflow: From Recording to Deployment

```
┌─────────────────────────────────┐
│ 1. Record Audio                 │
│    (record_audio.py)            │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ 2. Preprocess Data              │
│    (preprocess_data.py)         │
│    - Normalize                  │
│    - Extract Features (MFCC)    │
│    - Create Vocabulary          │
│    - Split Train/Val/Test       │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ 3. Train Model                  │
│    (train_model.py)             │
│    - Load preprocessed data     │
│    - Build LSTM                 │
│    - Train on GPU/CPU           │
│    - Save checkpoints           │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ 4. Evaluate Model               │
│    (test_inference.py)          │
│    - Test accuracy              │
│    - Generate predictions       │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ 5. Deploy to Digital Library    │
│    - WebSocket integration      │
│    - Real-time transcription    │
│    - Integration with UI        │
└─────────────────────────────────┘
```

## Configuration Files

### model_config.json
```json
{
  "input_shape": [300, 13],
  "vocab_size": 128,
  "lstm_units": 512,
  "embedding_dim": 256,
  "dropout_rate": 0.3,
  "learning_rate": 0.001
}
```

### training_config.json
```json
{
  "epochs": 50,
  "batch_size": 32,
  "validation_split": 0.15,
  "early_stopping_patience": 5
}
```

## Usage Examples

### Record 20 Kinyarwanda Samples
```bash
python scripts/record_audio.py
# Language: kinyarwanda
# Duration: 5
# Samples: 20
```

### Preprocess English Data
```bash
python scripts/preprocess_data.py
# Language: english
```

### Train Bilingual Models
```bash
python scripts/train_model.py
# Language: both
# Model type: bidirectional
# Epochs: 50
```

## Common Issues & Solutions

### Issue: No audio devices detected
**Solution:**
```bash
pip install --upgrade pyaudio
# On Mac: brew install portaudio && pip install pyaudio
# On Linux: sudo apt-get install portaudio19-dev && pip install pyaudio
```

### Issue: Out of memory during training
**Solution:**
- Reduce `batch_size` (from 32 to 16)
- Reduce `lstm_units` (from 512 to 256)
- Reduce dataset size
- Use GPU: Set `CUDA_VISIBLE_DEVICES=0`

### Issue: Low accuracy on Kinyarwanda
**Reason:** Limited training data
**Solution:**
- Record more Kinyarwanda samples (target: 50+ hours)
- Use data augmentation
- Fine-tune on English model

## Integration with Digital Library

### WebSocket Connection for Real-time STT

```python
from inference import STTInference

stt = STTInference(model_path='models/trained/english_bidirectional_final.h5')

# Real-time transcription
async def transcribe_audio_stream(websocket, audio_data):
    transcript = await stt.transcribe(audio_data)
    await websocket.send(transcript)
```

### API Endpoint

```python
@app.post("/api/transcribe")
async def transcribe(audio: UploadFile):
    stt = STTInference()
    transcript = stt.transcribe(audio.file)
    return {"transcript": transcript}
```

## Performance Metrics

### Expected Accuracy
- **English:** 85-92% (with 100+ hours)
- **Kinyarwanda:** 75-85% (with 50+ hours custom data)

### Training Time
- **Single GPU:** 2-4 hours for 50 epochs (100 hours data)
- **CPU:** 12-24 hours for 50 epochs

## License

MIT License

## Support

For questions or issues:
1. Check existing GitHub issues
2. Create detailed bug report with:
   - Language used
   - Model architecture
   - Dataset size
   - Error message
3. Include logs from `models/checkpoints/logs/`

---

**Next Steps:**
1. Run `record_audio.py` to create dataset
2. Run `preprocess_data.py` to prepare data
3. Run `train_model.py` to train model
4. Check `models/trained/` for your trained model
