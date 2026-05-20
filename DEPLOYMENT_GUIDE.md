# Deploy LSTM Kinyarwanda Speech Recognition to Railway

## What's Ready
✅ `requirements.txt` - All Python dependencies
✅ `Dockerfile` - Container configuration
✅ `Procfile` - Process configuration
✅ `.gitignore` - Git ignore rules

## Deployment Steps

### 1. Create a GitHub Repository
```bash
cd lstm-speech-recognition-en-rw
git init
git add .
git commit -m "Initial commit: LSTM speech recognition with Flask"
git remote add origin https://github.com/YOUR_USERNAME/lstm-speech-recognition-en-rw.git
git branch -M main
git push -u origin main
```

### 2. Deploy to Railway
1. Go to: https://railway.app
2. Sign up with GitHub (recommended)
3. Click "New Project" → "Deploy from GitHub"
4. Connect your repository
5. Select the repo: `lstm-speech-recognition-en-rw`
6. Click "Deploy"
7. Railway will build and deploy automatically (takes 2-5 minutes)

### 3. Get Your Live API URL
Once deployed, Railway gives you a URL like:
```
https://lstm-speech-recognition-xxx.railway.app
```

Your API endpoints are now live:
- `POST https://lstm-speech-recognition-xxx.railway.app/api/transcribe` - Transcribe audio
- `GET https://lstm-speech-recognition-xxx.railway.app/admin/login` - Admin panel
- `GET https://lstm-speech-recognition-xxx.railway.app/` - Web interface

### 4. Set Environment Variables (Optional)
In Railway dashboard:
- Go to your project settings
- Add variables:
  - `SECRET_KEY`: Random string for Flask sessions
  - `FLASK_ENV`: `production`

### 5. Test the Live App
```bash
curl https://lstm-speech-recognition-xxx.railway.app/api/status
# Should return: {"status": "Ready for transcription", ...}
```

## Next: Integrate into Digital Library

Once LSTM is live, update your Digital Library to call the LSTM API:

### In your Digital Library frontend:
```javascript
// Send audio to LSTM for transcription
const transcribeAudio = async (audioFile) => {
  const formData = new FormData();
  formData.append('audio', audioFile);
  
  const response = await fetch(
    'https://lstm-speech-recognition-xxx.railway.app/api/transcribe',
    {
      method: 'POST',
      body: formData
    }
  );
  
  const result = await response.json();
  return result.transcript; // Get the transcribed text
};
```

## Troubleshooting

**App won't deploy?**
- Check `Procfile` syntax
- Verify `requirements.txt` has all dependencies
- Check Railway logs for errors

**Model file too large?**
- Railway supports up to 100MB per file
- Your model is 152MB - might need to compress or use Railway's file storage

**Need to update code?**
- Just push to GitHub
- Railway automatically redeploys on every push

