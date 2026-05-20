# Web App Guide: LSTM Kinyarwanda Speech Recognition

**Complete guide for running the web application on desktop and mobile devices.**

---

## 🚀 Quick Start

### Windows Users
```bash
# 1. Double-click START_WEB_APP.bat
# or from terminal:
START_WEB_APP.bat

# 2. Open browser to http://localhost:5000
```

### Mac/Linux Users
```bash
# 1. Make script executable
chmod +x start_web_app.sh

# 2. Run the script
./start_web_app.sh

# 3. Open browser to http://localhost:5000
```

### Manual Setup
```bash
# Navigate to web_app directory
cd web_app

# Create virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# Create virtual environment (Mac/Linux)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Flask server
python app.py
```

---

## 📱 Features

### ✅ Desktop Features
- **Upload Audio Files**: Drag-and-drop or click to browse
- **Real-Time Recording**: Record directly from microphone
- **Instant Transcription**: Get results in seconds
- **Confidence Scores**: See accuracy of each transcription
- **Copy/Download**: Save transcriptions as text
- **History**: Track all transcriptions
- **Statistics**: View usage stats and model info
- **Responsive Layout**: Works on all screen sizes

### ✅ Mobile Features (Phone/Tablet)
- **Full Mobile UI**: Optimized touch interface
- **Microphone Recording**: Record directly from phone
- **File Upload**: Select audio from phone storage
- **Portrait Mode**: Vertical layout for phones
- **Fast Loading**: Optimized for mobile networks
- **Offline Support**: Works without internet after initial load
- **Battery Efficient**: Minimal resource usage

---

## 🎯 How to Use the Web App

### 1. **Start the Application**

**Windows:**
```
Double-click: START_WEB_APP.bat
```

**Mac/Linux:**
```bash
./start_web_app.sh
```

You should see:
```
 * Running on http://0.0.0.0:5000
 * Press CTRL+C to quit
```

### 2. **Access the App**

**On Same Computer:**
- Desktop: http://localhost:5000
- Phone on same WiFi: http://<YOUR_IP>:5000

**Get Your IP Address:**
```bash
# Windows
ipconfig

# Mac/Linux
ifconfig
```

Look for `IPv4 Address` (e.g., 192.168.1.100)

### 3. **Record Audio**

**Method 1: Microphone Recording**
1. Click "Start Recording" button
2. Speak clearly in Kinyarwanda
3. Click "Stop Recording" to submit
4. Wait for transcription

**Method 2: Upload File**
1. Drag audio file into upload area OR click to browse
2. Supports: WAV, MP3, OGG, M4A
3. Max file size: 50MB
4. Wait for transcription

### 4. **View Results**
- See transcribed text
- Confidence score (0-100%)
- Copy text with one click
- Download as TXT file
- Automatically added to history

### 5. **History & Statistics**
- Click "History" tab to see past transcriptions
- Click "Stats" tab for usage statistics
- Export history as JSON
- Clear history anytime

---

## 🌐 API Endpoints

If you want to use the app programmatically:

### Upload & Transcribe
```bash
curl -X POST -F "audio=@recording.wav" http://localhost:5000/api/transcribe
```

Response:
```json
{
  "success": true,
  "transcript": "Muraho neza.",
  "confidence": 92.3,
  "filename": "recording.wav",
  "timestamp": "2026-05-19T15:30:00"
}
```

### Get History
```bash
curl http://localhost:5000/api/history?limit=10
```

### Get Status
```bash
curl http://localhost:5000/api/status
```

### Get Statistics
```bash
curl http://localhost:5000/api/stats
```

---

## 🔧 Troubleshooting

### "Port 5000 already in use"
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Mac/Linux
lsof -i :5000
kill <PID>
```

Or use different port:
```bash
# Modify app.py last line:
# app.run(host='0.0.0.0', port=5001)
```

### "Model not found"
Make sure training is complete:
```bash
# Check if model file exists:
models/trained/kinyarwanda_masked_final.h5
# or checkpoints:
models/checkpoints/*/best_model.h5
```

If not, run training:
```bash
python scripts/train_with_masking.py
```

### "Audio file not recognized"
- Check file format (WAV, MP3, OGG, M4A)
- Check file size (< 50MB)
- Ensure audio quality (16-bit, 16kHz preferred)
- Try converting with FFmpeg:
  ```bash
  ffmpeg -i input.mp3 -acodec pcm_s16le -ar 16000 output.wav
  ```

### "Poor transcription accuracy"
- Ensure clear Kinyarwanda pronunciation
- Reduce background noise
- Avoid overlapping voices
- Keep recording 3-5 seconds
- Check confidence score (lower = less certain)

---

## 📊 File Structure

```
web_app/
├── app.py                      # Flask server
├── requirements.txt            # Python dependencies
├── templates/
│   └── index.html             # Web interface
├── static/
│   ├── css/                   # Styles (in HTML)
│   └── js/                    # Scripts (in HTML)
├── uploads/                   # Temporary uploaded files
└── results/
    └── history.json           # Transcription history
```

---

## 🛠️ Configuration

### Change Port Number
Edit the last line of `app.py`:
```python
if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=8000)  # Change 5000 to 8000
```

### Enable/Disable Debug Mode
```python
# Production (debug=False):
app.run(debug=False, host='0.0.0.0', port=5000)

# Development (debug=True):
app.run(debug=True, host='0.0.0.0', port=5000)
```

### Change Upload Limit
In `app.py`, line ~20:
```python
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
```

### Use Different Model
In `app.py`, modify `load_model_and_vocab()`:
```python
# Change to English instead of Kinyarwanda:
vocab_path = f'../data/english/vocabulary.json'
model_path = f'../models/trained/english_masked_final.h5'
```

---

## 🚀 Production Deployment

### Using Gunicorn (Linux/Mac)
```bash
pip install gunicorn
cd web_app
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Heroku
```bash
# Install Heroku CLI
# Create Procfile:
echo "web: gunicorn app:app" > Procfile

# Deploy:
heroku login
heroku create
git push heroku main
```

### Using Docker
Create `Dockerfile`:
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t lstm-speech-recognition .
docker run -p 5000:5000 lstm-speech-recognition
```

### Using Nginx Reverse Proxy
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 📱 Mobile Access

### Access from Phone on Same WiFi
1. Get your computer's IP: 
   - Windows: `ipconfig` → IPv4 Address
   - Mac/Linux: `ifconfig` → inet address
   
2. On phone, open browser:
   - `http://<YOUR_IP>:5000`
   - Example: `http://192.168.1.100:5000`

3. Bookmark for quick access

### Mobile Browser Support
✅ **Chrome** - Full support  
✅ **Firefox** - Full support  
✅ **Safari** (iPhone/iPad) - Full support  
✅ **Edge** - Full support  

---

## 🔒 Security Considerations

### For Public Deployment:
1. **Enable HTTPS** (use Let's Encrypt)
2. **Add Authentication** (user login)
3. **Rate Limiting** (prevent abuse)
4. **File Validation** (verify audio files)
5. **CORS Protection** (restrict domains)

Example with Flask-Limiter:
```python
from flask_limiter import Limiter

limiter = Limiter(app)

@app.route('/api/transcribe', methods=['POST'])
@limiter.limit("10 per minute")
def transcribe():
    # ...
```

---

## 📊 Monitoring

### View Real-Time Logs
```bash
# While app is running:
# Check terminal output for errors and requests
# Each request logs to console
```

### Monitor System Resources
```bash
# Windows (Task Manager)
# Press: Ctrl + Shift + Esc

# Mac (Activity Monitor)
# Cmd + Space → Activity Monitor

# Linux (Terminal)
top  # or htop
```

### Check Disk Usage
```bash
# See history file size:
ls -lh web_app/results/history.json

# Clear cache if needed:
# Delete old files in web_app/uploads/
```

---

## 🎓 Advanced Usage

### Batch Processing
Create `batch_transcribe.py`:
```python
import os
import requests

folder = 'audio_files/'
results = []

for file in os.listdir(folder):
    with open(os.path.join(folder, file), 'rb') as f:
        response = requests.post(
            'http://localhost:5000/api/transcribe',
            files={'audio': f}
        )
        if response.json()['success']:
            results.append({
                'file': file,
                'transcript': response.json()['transcript']
            })

import json
with open('batch_results.json', 'w') as f:
    json.dump(results, f)
```

### Integrate with Other Apps
```python
import requests

def transcribe_audio(audio_file_path):
    with open(audio_file_path, 'rb') as f:
        response = requests.post(
            'http://localhost:5000/api/transcribe',
            files={'audio': f}
        )
    return response.json()

# Usage:
result = transcribe_audio('my_recording.wav')
print(result['transcript'])
print(f"Confidence: {result['confidence']}%")
```

---

## 📝 Logs

### Where are logs stored?
- **Console Output**: Displayed in terminal while running
- **History**: `web_app/results/history.json`
- **Uploads**: `web_app/uploads/` (temporary files)

### View History JSON
```bash
# Pretty print history:
python -m json.tool web_app/results/history.json | head -50
```

---

## 💡 Tips & Best Practices

1. **Recording Tips:**
   - Speak clearly and naturally
   - Minimize background noise
   - Keep 3-5 seconds of audio
   - Use quality microphone for better results

2. **File Upload Tips:**
   - Use WAV format for best quality
   - 16 kHz sample rate recommended
   - Keep file under 50MB
   - One speaker at a time

3. **Performance Tips:**
   - First transcription takes longer (model loading)
   - Subsequent requests are faster
   - Use Firefox/Chrome for best compatibility
   - Close other browser tabs to free memory

4. **Troubleshooting Tips:**
   - Restart app if having issues
   - Clear browser cache (Ctrl+Shift+Del)
   - Try different browser
   - Check internet connection stability

---

## 🆘 Support & Help

### Check System Requirements
```bash
python --version  # Should be 3.8+
pip --version     # Should be 20+
```

### Install Dependencies Manually
```bash
pip install Flask==2.3.0
pip install tensorflow==2.14.0
pip install librosa==0.10.0
# ... (see requirements.txt)
```

### Reinstall Everything
```bash
# Backup history
cp web_app/results/history.json web_app/results/history.backup.json

# Remove virtual environment
rm -rf web_app/venv  # or use Windows equivalent

# Restart app (will recreate venv)
./start_web_app.sh
```

---

## 📚 Related Documentation

- **COMPREHENSIVE_TECHNICAL_GUIDE.md** - Technical details
- **EVALUATION_AND_RESULTS.md** - Results interpretation
- **README_LSTM_SPEECH_RECOGNITION.md** - Project overview
- **FIX_APPLIED.md** - NaN loss fix details

---

**Happy Transcribing! 🎤**

