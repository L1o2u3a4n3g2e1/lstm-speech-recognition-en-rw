# Using Kaggle Speech Datasets

This guide shows how to download and use English speech datasets from Kaggle for training your LSTM model.

---

## 🚀 Quick Start: Download Kaggle Dataset

### Step 1: Get Kaggle API Credentials

1. **Visit Kaggle:** https://www.kaggle.com/settings/account
2. **Click "Create New API Token"** button
3. **File downloads:** `kaggle.json`
4. **Move file to:** `C:\Users\Anne Louange\.kaggle\kaggle.json`
   - Create `.kaggle` folder if it doesn't exist
   - On Windows, use: `mkdir %USERPROFILE%\.kaggle`

### Step 2: Install Kaggle CLI

```powershell
cd "c:\Users\Anne Louange\Desktop\lstm-speech-recognition-en-rw"
.\venv\Scripts\pip install kaggle
```

### Step 3: Download a Dataset

```powershell
# Set API credentials
$env:KAGGLE_CONFIG_DIR = "$env:USERPROFILE\.kaggle"

# Download Librispeech (High quality, ~100GB)
kaggle datasets download -d bhavanshiv/librispeech -p data/english/raw/

# Or: Common Voice English (Smaller, ~20GB)
kaggle datasets download -d bhavanshiv/commonvoice -p data/english/raw/

# Or: Speech Emotion Recognition
kaggle datasets download -d ejlok1/cremad -p data/english/raw/
```

---

## 📊 Recommended Kaggle Datasets

### 1. **Librispeech** (BEST for Speech Recognition)
- **Download:** https://www.kaggle.com/datasets/bhavanshiv/librispeech
- **Size:** ~100 GB (full) or 6 GB (test-clean subset)
- **Quality:** Excellent (audiobook recordings)
- **Samples:** 1000+ hours
- **Format:** WAV files + transcriptions
- **Use for:** Production-quality model

**Download:**
```powershell
kaggle datasets download -d bhavanshiv/librispeech -p data/english/raw/
```

---

### 2. **Common Voice - English**
- **Download:** https://www.kaggle.com/datasets/mozilla/common-voice
- **Size:** ~20-50 GB (depending on language)
- **Quality:** Good (crowdsourced)
- **Samples:** 500+ hours
- **Format:** MP3/WAV + transcriptions
- **Use for:** Diverse accent training

**Download:**
```powershell
kaggle datasets download -d mozilla/common-voice -p data/english/raw/
```

---

### 3. **CREMAD - Speech Emotion Recognition**
- **Download:** https://www.kaggle.com/datasets/ejlok1/cremad
- **Size:** ~2 GB
- **Quality:** Excellent (studio quality)
- **Samples:** 7,000+ audio clips
- **Format:** WAV files + emotions
- **Use for:** Smaller subset for quick testing

**Download:**
```powershell
kaggle datasets download -d ejlok1/cremad -p data/english/raw/
```

---

### 4. **Free Spoken Digit Dataset**
- **Download:** https://www.kaggle.com/datasets/joshuaswords/free-spoken-digit-dataset
- **Size:** 80 MB
- **Quality:** Good
- **Samples:** 3,000 (digits 0-9)
- **Format:** WAV files
- **Use for:** Quick testing & validation

**Download:**
```powershell
kaggle datasets download -d joshuaswords/free-spoken-digit-dataset -p data/english/raw/
```

---

## 🎯 Strategy: Combine Kaggle + Your Custom Recordings

### For Best Results:
1. **Download Librispeech** (English baseline)
2. **Record custom Kinyarwanda** (your voices)
3. **Train separate models:**
   - English: Librispeech
   - Kinyarwanda: Custom recordings
4. **Fine-tune:** Mix datasets if needed

---

## 📥 Step-by-Step Example: Download Librispeech

### Step 1: Check Kaggle Credentials
```powershell
ls $env:USERPROFILE\.kaggle\
# Should show: kaggle.json
```

### Step 2: Create Download Directory
```powershell
mkdir "c:\Users\Anne Louange\Desktop\lstm-speech-recognition-en-rw\data\english\raw\librispeech"
```

### Step 3: Download Dataset
```powershell
cd "c:\Users\Anne Louange\Desktop\lstm-speech-recognition-en-rw"

# Download (this may take 10-30 minutes depending on dataset size)
kaggle datasets download -d bhavanshiv/librispeech -p data/english/raw/

# The download will create a ZIP file
# It may auto-extract, or you need to unzip
```

### Step 4: Extract Files (if needed)
```powershell
# On Windows, use PowerShell to extract
Expand-Archive -Path "data\english\raw\librispeech.zip" -DestinationPath "data\english\raw\librispeech"

# Or use Windows Explorer: Right-click ZIP → Extract All
```

### Step 5: Verify Downloaded Files
```powershell
ls data/english/raw/librispeech -Recurse | Where-Object {$_.Extension -eq ".wav"} | Measure-Object
# Should show: Count = number of WAV files
```

---

## 🔧 Preprocessing Kaggle Datasets

Once downloaded, preprocess them:

```powershell
# Update preprocess_data.py to use Kaggle data
python scripts/preprocess_data.py

# When prompted for language: english
# This will:
# 1. Scan librispeech directory
# 2. Extract MFCC features from all files
# 3. Create training/validation/test splits
# 4. Save to data/english/processed/
```

---

## 📋 Organizing Downloaded Data

**After downloading from Kaggle:**

```
data/
├── english/
│   ├── raw/
│   │   ├── librispeech/          ← Downloaded from Kaggle
│   │   │   ├── train-clean-100/
│   │   │   ├── test-clean/
│   │   │   └── ...
│   │   ├── common_voice/         ← Alternative: Common Voice
│   │   │   ├── cv-corpus-12.0-2022-12-07/
│   │   │   └── ...
│   │   └── custom_recordings/    ← Your voice recordings
│   │       ├── english_*.wav
│   │       └── transcripts.csv
│   └── processed/                ← Auto-generated after preprocessing
│       ├── train/
│       ├── val/
│       └── test/
│
└── kinyarwanda/
    ├── raw/
    │   └── custom_recordings/    ← Your Kinyarwanda recordings
    │       ├── kinyarwanda_*.wav
    │       └── transcripts.csv
    └── processed/
```

---

## ⚙️ Kaggle API Troubleshooting

### Problem: "403 - Forbidden Error"
**Solution:** Check API credentials
```powershell
# Verify kaggle.json exists
ls $env:USERPROFILE\.kaggle\kaggle.json

# Re-download from: https://www.kaggle.com/settings/account
# Click "Create New API Token"
```

### Problem: "SSL Certificate Error"
**Solution:** Update certificates
```powershell
.\venv\Scripts\pip install --upgrade certifi
```

### Problem: "Dataset not found"
**Solution:** Check dataset name
```powershell
# List your datasets
kaggle datasets list

# Search for specific dataset
kaggle datasets list -s "librispeech"
```

### Problem: Download is very slow
**Solution:** Consider downloading manually from Kaggle website
1. Visit dataset page
2. Click "Download" button
3. Save to `data/english/raw/`

---

## 🎯 Complete Workflow: Kaggle + Custom + Training

```
┌─────────────────────────────┐
│ Download from Kaggle        │
│ (librispeech or CommonVoice)│
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ Record Custom Kinyarwanda   │
│ (Using EN_RW_DICTIONARY)    │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ Preprocess Both Datasets    │
│ (preprocess_data.py)        │
│ - English from Kaggle       │
│ - Kinyarwanda from custom   │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ Train LSTM Models           │
│ (train_model.py)            │
│ - English model             │
│ - Kinyarwanda model         │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ Test & Deploy               │
│ (inference.py)              │
│ - Real-time transcription   │
│ - Digital Library integration│
└─────────────────────────────┘
```

---

## 📊 Dataset Comparison

| Dataset | Size | Hours | Quality | Best For |
|---------|------|-------|---------|----------|
| **Librispeech** | 100GB | 1000+ | Excellent | Production |
| **Common Voice** | 20-50GB | 500+ | Good | Diverse accents |
| **CREMAD** | 2GB | 7000 clips | Excellent | Quick testing |
| **Free Digit** | 80MB | 3000 | Good | Validation |
| **Your Custom** | 1-10GB | 10-100 | Variable | Domain-specific |

---

## 💡 Recommended Path

1. **Week 1:**
   - Record 10-20 Kinyarwanda samples (EN_RW_DICTIONARY)
   - Download CREMAD dataset (small, fast)
   - Train quick model for testing

2. **Week 2:**
   - Record 50+ Kinyarwanda samples
   - Download Librispeech test-clean (~6GB)
   - Train English + Kinyarwanda models

3. **Week 3+:**
   - Download full Librispeech or Common Voice
   - Record 100+ Kinyarwanda samples
   - Train production-quality bilingual models

---

## 🔗 Resources

- **Kaggle Datasets:** https://www.kaggle.com/datasets
- **Kaggle API Docs:** https://github.com/Kaggle/kaggle-api
- **Librispeech Paper:** http://www.openslr.org/12/
- **Common Voice:** https://commonvoice.mozilla.org/

---

## ✅ Checklist

- [ ] Created `~/.kaggle/` directory
- [ ] Downloaded `kaggle.json` from Kaggle
- [ ] Installed Kaggle CLI: `pip install kaggle`
- [ ] Verified API credentials work
- [ ] Downloaded dataset to `data/english/raw/`
- [ ] Extracted files if needed
- [ ] Verified WAV files exist
- [ ] Ready to preprocess with `preprocess_data.py`

Good luck! 🚀
