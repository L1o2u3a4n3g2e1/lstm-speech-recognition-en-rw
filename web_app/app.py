"""
Flask Web Application for LSTM Kinyarwanda Speech Recognition
Supports audio upload and real-time transcription
Mobile and desktop responsive interface
"""

import os
import json
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
import logging
import tensorflow as tf
from pathlib import Path
import librosa
import threading

# Add parent directory to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.lstm_model import LSTMSpeechRecognition
from src.audio_processor import AudioProcessor
from scripts.train_from_web import ContinuousTrainer
from admin_auth import admin_manager, login_required, check_admin_password

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'ogg', 'm4a', 'webm'}
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'lstm-speech-recognition-secret-2026')

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('results', exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model and processor
model = None
processor = None
char_to_num = None
num_to_char = None
vocab_size = 27
trainer = None  # Continuous trainer for fine-tuning

class TranscriptionHistory:
    """Store transcription history"""
    def __init__(self, history_file='results/history.json'):
        self.history_file = history_file
        self.history = self.load_history()

    def load_history(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return []

    def save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def add(self, audio_file, transcript, confidence, language='kinyarwanda'):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'filename': audio_file,
            'transcript': transcript,
            'confidence': confidence,
            'language': language
        }
        self.history.append(entry)
        self.save_history()
        return entry

    def get_recent(self, limit=10):
        return self.history[-limit:][::-1]  # Reverse to show newest first

    def clear(self):
        self.history = []
        self.save_history()

history = TranscriptionHistory()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def convert_webm_to_wav(webm_path):
    """Convert WebM audio file to WAV format for librosa compatibility"""
    try:
        import soundfile as sf

        # Try to load with librosa (handles some WebM files)
        try:
            audio, sr = librosa.load(webm_path, sr=16000, mono=True)
            # If successful, save as WAV
            wav_path = webm_path.replace('.webm', '.wav')
            sf.write(wav_path, audio, 16000)
            os.remove(webm_path)
            return wav_path
        except Exception as librosa_err:
            logger.warning(f"Librosa couldn't load WebM directly: {librosa_err}")

            # Try ffmpeg as fallback
            try:
                import subprocess
                wav_path = webm_path.replace('.webm', '.wav')
                cmd = ['ffmpeg', '-i', webm_path, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', wav_path]
                result = subprocess.run(cmd, capture_output=True, timeout=30)
                if result.returncode == 0:
                    os.remove(webm_path)
                    return wav_path
            except Exception as ffmpeg_err:
                logger.warning(f"ffmpeg not available: {ffmpeg_err}")

            raise Exception(f"Could not convert WebM to WAV. Install ffmpeg or update audio libraries.")
    except Exception as e:
        logger.error(f"Error converting WebM: {e}")
        return None

def load_model_and_vocab(language='kinyarwanda'):
    """Load trained model and vocabulary"""
    global model, processor, char_to_num, num_to_char, vocab_size, trainer

    try:
        # Load vocabulary
        vocab_path = f'../data/{language}/vocabulary_augmented.json'
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            char_to_num = vocab_data['char_to_num']
            num_to_char = vocab_data['num_to_char']
            vocab_size = len(char_to_num)

        # Build and load model
        model = LSTMSpeechRecognition(
            input_shape=(300, 13),
            vocab_size=vocab_size,
            embedding_dim=256,
            lstm_units=512,
            dropout_rate=0.4,
            learning_rate=0.001
        )
        model.build_bidirectional()

        # Load weights
        model_path = f'../models/trained/{language}_masked_final.h5'
        checkpoint_path = f'../models/checkpoints/{language}_masked_*/best_model.h5'

        # Try to load final model first, then checkpoint
        if os.path.exists(model_path):
            model.model.load_weights(model_path)
            logger.info(f"Loaded final model: {model_path}")
        else:
            # Find latest checkpoint
            checkpoint_dir = f'../models/checkpoints/'
            if os.path.exists(checkpoint_dir):
                latest_dir = max(Path(checkpoint_dir).glob('**/'), key=os.path.getmtime)
                checkpoint_file = latest_dir / 'best_model.h5'
                if checkpoint_file.exists():
                    model.model.load_weights(str(checkpoint_file))
                    logger.info(f"Loaded checkpoint: {checkpoint_file}")
                else:
                    logger.warning("No trained model found. Using untrained model.")
            else:
                logger.warning("No model checkpoint directory found.")

        # Initialize audio processor
        processor = AudioProcessor(sr=16000, duration=5, n_mfcc=13)

        # Initialize continuous trainer for fine-tuning
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, 'models', 'trained', f'{language}_masked_final.h5')
            vocab_path = os.path.join(base_dir, 'data', language, 'vocabulary_augmented.json')
            trainer = ContinuousTrainer(model_path, vocab_path)
            logger.info("Continuous trainer initialized")
        except Exception as e:
            logger.warning(f"Could not initialize trainer: {e}")

        logger.info(f"Model loaded successfully. Vocab size: {vocab_size}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def process_audio_file(audio_path):
    """Process audio file and return transcription"""
    try:
        # Load and process audio
        audio, sr = processor.load_audio(audio_path)
        if audio is None:
            return None, 0.0, "Failed to load audio file"

        # Normalize and preprocess
        audio = processor.normalize_audio(audio)
        audio = processor.remove_silence(audio)
        audio = processor.pad_or_trim(audio)

        # Extract MFCC
        mfcc = processor.extract_mfcc(audio)
        mfcc = np.log(np.abs(mfcc) + 1e-9)

        # Pad to fixed length
        if mfcc.shape[1] < 300:
            pad_width = ((0, 0), (0, 300 - mfcc.shape[1]))
            mfcc = np.pad(mfcc, pad_width, mode='constant')
        else:
            mfcc = mfcc[:, :300]

        # Prepare for model
        mfcc = np.transpose(mfcc, (1, 0))  # (300, 13)
        mfcc = mfcc / (np.max(np.abs(mfcc)) + 1e-8)
        mfcc = np.expand_dims(mfcc, axis=0)  # (1, 300, 13)

        # Get predictions
        predictions = model.model.predict(mfcc, verbose=0)
        predicted_classes = np.argmax(predictions[0], axis=-1)
        confidences = np.max(predictions[0], axis=-1)

        # Decode to text
        transcript = ''
        for class_id in predicted_classes:
            if class_id == 2:  # PAD token
                break
            if class_id in num_to_char:
                transcript += num_to_char[str(class_id)]

        # Calculate confidence
        non_pad_mask = predicted_classes != 2
        if np.sum(non_pad_mask) > 0:
            avg_confidence = np.mean(confidences[non_pad_mask]) * 100
        else:
            avg_confidence = 0.0

        transcript = transcript.strip()
        return transcript, avg_confidence, "Success"

    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return None, 0.0, str(e)

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    """API endpoint for transcription"""
    try:
        # Check if file is provided
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': f'File type not allowed. Allowed: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'}), 400

        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], timestamp + filename)
        file.save(filepath)

        logger.info(f"Processing file: {filepath} (size: {os.path.getsize(filepath)} bytes)")

        # Process audio
        transcript, confidence, message = process_audio_file(filepath)

        if transcript is None:
            return jsonify({
                'error': f'Processing failed: {message}',
                'success': False
            }), 500

        # Ensure confidence is valid (not NaN)
        if np.isnan(confidence) or confidence is None:
            confidence = 0.0

        # Add to history
        history_entry = history.add(filename, transcript, round(confidence, 2))

        # Silently add to learning queue for background training (admin learns without user knowing)
        admin_manager.add_to_learning_queue(filepath, transcript, confidence)

        # Ensure confidence is valid before returning
        confidence_value = round(confidence, 2) if not (np.isnan(confidence) or confidence is None) else 0.0

        response = {
            'success': True,
            'transcript': transcript,
            'confidence': confidence_value,
            'filename': filename,
            'timestamp': history_entry['timestamp'],
            'message': message
        }

        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in transcribe endpoint: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get transcription history"""
    try:
        limit = request.args.get('limit', 10, type=int)
        recent = history.get_recent(limit)
        total_count = len(history.history)
        return jsonify({
            'success': True,
            'history': recent,
            'count': total_count
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    """Clear transcription history"""
    try:
        history.clear()
        return jsonify({'success': True, 'message': 'History cleared'}), 200
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get model status and info"""
    return jsonify({
        'success': True,
        'model_loaded': model is not None,
        'language': 'kinyarwanda',
        'vocab_size': vocab_size,
        'model_parameters': '13.2M',
        'version': '1.0',
        'status': 'Ready for transcription'
    }), 200

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get usage statistics"""
    try:
        total_transcriptions = len(history.history)
        recent_7 = [h for h in history.history if
                   (datetime.fromisoformat(h['timestamp']).date() - datetime.now().date()).days <= 7]

        return jsonify({
            'success': True,
            'total_transcriptions': total_transcriptions,
            'recent_7_days': len(recent_7),
            'history_entries': total_transcriptions
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/download-history', methods=['GET'])
def download_history():
    """Download history as JSON"""
    try:
        return send_file(
            history.history_file,
            as_attachment=True,
            download_name=f'transcription_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/train', methods=['POST'])
def train_on_sample():
    """Train model on uploaded audio sample"""
    try:
        if not trainer:
            return jsonify({'error': 'Trainer not initialized', 'success': False}), 500

        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided', 'success': False}), 400

        if 'transcript' not in request.form:
            return jsonify({'error': 'No transcript provided', 'success': False}), 400

        file = request.files['audio']
        transcript = request.form.get('transcript', '').strip()

        if file.filename == '':
            return jsonify({'error': 'No selected file', 'success': False}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': f'File type not allowed. Allowed: {", ".join(app.config["ALLOWED_EXTENSIONS"])}', 'success': False}), 400

        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], timestamp + filename)
        file.save(filepath)

        logger.info(f"Training on file: {filepath} with transcript: {transcript}")

        # Get learning rate from request (optional)
        learning_rate = request.form.get('learning_rate', 0.0001, type=float)

        # Train on this sample
        success, message = trainer.train_on_audio(filepath, transcript, learning_rate)

        if success:
            # Save model after training
            trainer.save_model()

            response = {
                'success': True,
                'message': message,
                'training_state': trainer.get_training_state(),
                'timestamp': datetime.now().isoformat()
            }
        else:
            response = {
                'success': False,
                'error': message,
                'timestamp': datetime.now().isoformat()
            }

        # Clean up
        try:
            os.remove(filepath)
        except:
            pass

        return jsonify(response), 200 if success else 400

    except Exception as e:
        logger.error(f"Error in train endpoint: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/train/status', methods=['GET'])
def get_training_status():
    """Get current training status"""
    try:
        if not trainer:
            return jsonify({'error': 'Trainer not initialized', 'success': False}), 500

        state = trainer.get_training_state()
        return jsonify({
            'success': True,
            'training_state': state,
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error in training status endpoint: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/model/save', methods=['POST'])
def save_model():
    """Save current model state"""
    try:
        if not trainer:
            return jsonify({'error': 'Trainer not initialized', 'success': False}), 500

        success, message = trainer.save_model()
        return jsonify({
            'success': success,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }), 200 if success else 500

    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    """Get detailed model information"""
    try:
        model_info = {
            'name': 'LSTM Kinyarwanda Speech Recognition',
            'architecture': 'BiLSTM × 3 + Attention + Dense',
            'vocab_size': vocab_size,
            'input_shape': [300, 13],
            'output_shape': [300, 27],
            'parameters': '13.2M',
            'language': 'Kinyarwanda',
            'training_capable': trainer is not None,
            'version': '2.0'
        }
        if trainer:
            model_info['training_state'] = trainer.get_training_state()

        return jsonify({
            'success': True,
            'model_info': model_info,
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

# ===== ADMIN ROUTES =====

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Admin login page"""
    if request.method == 'POST':
        password = request.form.get('password', '')
        if check_admin_password(password):
            session['admin_logged_in'] = True
            session['admin_token'] = os.urandom(16).hex()
            flash('Admin access granted', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid password', 'danger')

    return render_template('admin_login.html')

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    """Admin control panel"""
    stats = admin_manager.get_statistics()
    return render_template('admin_dashboard.html', stats=stats)

@app.route('/admin/learning-queue')
@login_required
def admin_learning_queue():
    """View and manage learning queue (admin only)"""
    try:
        samples = admin_manager.queue
        return jsonify({
            'success': True,
            'samples': samples,
            'total': len(samples),
            'unlearned': sum(1 for s in samples if not s.get('learned', False))
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/admin/learning-stats')
@login_required
def admin_learning_stats():
    """Get background learning statistics (admin only)"""
    try:
        stats = admin_manager.get_statistics()
        return jsonify({
            'success': True,
            'stats': stats
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/admin/train-background', methods=['POST'])
@login_required
def train_background():
    """Manually trigger background learning (admin only)"""
    try:
        samples = admin_manager.get_unlearned_samples(limit=10)
        trained_count = 0

        for sample in samples:
            if os.path.exists(sample['audio_path']):
                success, _ = trainer.train_on_audio(
                    sample['audio_path'],
                    sample['transcript'],
                    learning_rate=0.00005  # Lower learning rate for background
                )
                if success:
                    admin_manager.mark_learned(sample['audio_path'])
                    trained_count += 1

        # Save model after batch training
        if trained_count > 0:
            trainer.save_model()

        return jsonify({
            'success': True,
            'trained': trained_count,
            'message': f'Trained on {trained_count} samples in background'
        }), 200

    except Exception as e:
        logger.error(f"Background training error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/admin/logout')
def admin_logout():
    """Logout admin"""
    session.clear()
    flash('Logged out', 'info')
    return redirect(url_for('index'))

@app.route('/api/learning-queue', methods=['GET'])
def get_learning_queue():
    """Get learning queue (for debugging/testing)"""
    try:
        stats = admin_manager.get_statistics()
        return jsonify({
            'success': True,
            'total_samples': stats.get('total_samples', 0),
            'learned_count': stats.get('learned_count', 0),
            'pending_count': stats.get('pending_count', 0),
            'learning_enabled': stats.get('learning_enabled', False)
        }), 200
    except Exception as e:
        logger.error(f"Error getting learning queue: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size: 50MB'}), 413

@app.errorhandler(404)
def not_found(error):
    """Handle 404 error"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 error"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

def create_app():
    """Create and configure the Flask app"""
    # Load model on startup
    if not load_model_and_vocab():
        logger.warning("Model not loaded. App will run but transcription may not work.")

    return app

if __name__ == '__main__':
    app = create_app()
    logger.info("Starting LSTM Speech Recognition Web App")
    logger.info("Access at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

