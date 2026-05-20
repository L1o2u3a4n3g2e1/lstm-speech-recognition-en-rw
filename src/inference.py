import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import os
import logging
from audio_processor import AudioProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class STTInference:
    """Speech-to-Text Inference Engine."""

    def __init__(self, model_path, language='english', vocab_dir=None):
        """
        Initialize inference engine.

        Args:
            model_path: Path to trained LSTM model
            language: 'english' or 'kinyarwanda'
            vocab_dir: Directory containing vocabulary JSON
        """
        self.model_path = model_path
        self.language = language.lower()
        self.model = None
        self.audio_processor = AudioProcessor(sr=16000, n_mfcc=13)
        self.char_to_num = {}
        self.num_to_char = {}

        self._load_model()
        self._load_vocabulary(vocab_dir)

    def _load_model(self):
        """Load trained model."""
        if not os.path.exists(self.model_path):
            logger.error(f"Model not found: {self.model_path}")
            return

        try:
            self.model = keras.models.load_model(self.model_path)
            logger.info(f"Model loaded: {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")

    def _load_vocabulary(self, vocab_dir=None):
        """Load character-to-number mapping."""
        if vocab_dir is None:
            vocab_dir = f'data/{self.language}'

        vocab_file = os.path.join(vocab_dir, 'vocabulary.json')

        if not os.path.exists(vocab_file):
            logger.warning(f"Vocabulary not found: {vocab_file}")
            logger.info("Using default character set...")
            self._create_default_vocabulary()
            return

        try:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)

            self.char_to_num = vocab_data.get('char_to_num', {})
            num_to_char_str = vocab_data.get('num_to_char', {})
            self.num_to_char = {int(k): v for k, v in num_to_char_str.items()}

            logger.info(f"Vocabulary loaded: {len(self.char_to_num)} characters")

        except Exception as e:
            logger.error(f"Error loading vocabulary: {e}")
            self._create_default_vocabulary()

    def _create_default_vocabulary(self):
        """Create default vocabulary."""
        if self.language == 'kinyarwanda':
            chars = set('abcdefghijklmnopqrstuvwxyzàâäçèéêëìîïñòôöœùûüœŵŷ ')
        else:
            chars = set('abcdefghijklmnopqrstuvwxyz ')

        chars.add('<PAD>')
        chars.add('<UNK>')

        self.char_to_num = {char: idx for idx, char in enumerate(sorted(chars))}
        self.num_to_char = {idx: char for char, idx in self.char_to_num.items()}

    def transcribe_audio_file(self, audio_path):
        """Transcribe audio file to text."""
        if self.model is None:
            logger.error("Model not loaded!")
            return None

        # Preprocess audio
        result = self.audio_processor.preprocess_audio(audio_path, remove_silence_flag=True)
        if result is None:
            return None

        # Extract MFCC features
        mfcc = result['mfcc']

        # Pad/truncate to fixed length
        if mfcc.shape[1] > 300:
            mfcc = mfcc[:, :300]
        else:
            pad_width = ((0, 0), (0, 300 - mfcc.shape[1]))
            mfcc = np.pad(mfcc, pad_width, mode='constant', constant_values=0)

        # Add batch dimension
        X = np.expand_dims(mfcc, axis=0)

        # Predict
        predictions = self.model.predict(X, verbose=0)[0]

        # Decode predictions
        transcript = self._decode_predictions(predictions)

        return transcript

    def transcribe_raw_audio(self, audio_data, sr=16000):
        """Transcribe raw audio data (numpy array)."""
        if self.model is None:
            logger.error("Model not loaded!")
            return None

        # Ensure correct sample rate
        if sr != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)

        # Extract features manually
        mfcc = self.audio_processor.extract_mfcc(audio_data)

        # Pad/truncate
        if mfcc.shape[1] > 300:
            mfcc = mfcc[:, :300]
        else:
            pad_width = ((0, 0), (0, 300 - mfcc.shape[1]))
            mfcc = np.pad(mfcc, pad_width, mode='constant', constant_values=0)

        # Add batch dimension
        X = np.expand_dims(mfcc, axis=0)

        # Predict
        predictions = self.model.predict(X, verbose=0)[0]

        # Decode
        transcript = self._decode_predictions(predictions)

        return transcript

    def _decode_predictions(self, predictions, confidence_threshold=0.5):
        """Decode model predictions to text."""
        # predictions shape: (time_steps, vocab_size)

        transcript = []
        last_char = None

        for frame_pred in predictions:
            # Get character with highest confidence
            char_idx = np.argmax(frame_pred)
            confidence = frame_pred[char_idx]

            if confidence < confidence_threshold:
                continue

            char = self.num_to_char.get(char_idx, '<UNK>')

            # Skip padding and unknown tokens
            if char in ['<PAD>', '<UNK>']:
                continue

            # Avoid consecutive duplicates (CTC-style)
            if char != last_char:
                transcript.append(char)
                last_char = char

        return ''.join(transcript).strip()

    def transcribe_with_confidence(self, audio_path):
        """Transcribe with per-frame confidence scores."""
        if self.model is None:
            logger.error("Model not loaded!")
            return None

        result = self.audio_processor.preprocess_audio(audio_path)
        if result is None:
            return None

        mfcc = result['mfcc']

        # Pad/truncate
        if mfcc.shape[1] > 300:
            mfcc = mfcc[:, :300]
        else:
            pad_width = ((0, 0), (0, 300 - mfcc.shape[1]))
            mfcc = np.pad(mfcc, pad_width, mode='constant', constant_values=0)

        X = np.expand_dims(mfcc, axis=0)
        predictions = self.model.predict(X, verbose=0)[0]

        # Extract character and confidence per frame
        frames = []
        for frame_pred in predictions:
            char_idx = np.argmax(frame_pred)
            confidence = frame_pred[char_idx]
            char = self.num_to_char.get(char_idx, '<UNK>')

            frames.append({
                'character': char,
                'confidence': float(confidence),
                'index': int(char_idx)
            })

        transcript = self._decode_predictions(predictions)

        return {
            'transcript': transcript,
            'frames': frames,
            'avg_confidence': np.mean([f['confidence'] for f in frames])
        }

    def batch_transcribe(self, audio_paths):
        """Transcribe multiple audio files."""
        results = {}

        for audio_path in audio_paths:
            logger.info(f"Transcribing: {audio_path}")
            transcript = self.transcribe_audio_file(audio_path)
            results[audio_path] = transcript

        return results


class RealTimeSTT:
    """Real-time Speech-to-Text using microphone input."""

    def __init__(self, model_path, language='english', sr=16000, chunk_duration=0.5):
        """
        Initialize real-time STT.

        Args:
            model_path: Path to trained model
            language: Language code
            sr: Sample rate
            chunk_duration: Duration of audio chunk in seconds
        """
        self.stt = STTInference(model_path, language)
        self.sr = sr
        self.chunk_size = int(sr * chunk_duration)
        self.audio_processor = AudioProcessor(sr=sr)

    def stream_transcribe(self, duration=10, update_callback=None):
        """
        Real-time transcription from microphone.

        Args:
            duration: Total recording duration in seconds
            update_callback: Function called with intermediate results
        """
        import pyaudio

        p = pyaudio.PyAudio()

        try:
            stream = p.open(format=pyaudio.paFloat32, channels=1, rate=self.sr,
                          input=True, frames_per_buffer=self.chunk_size)

            logger.info(f"Recording for {duration} seconds...")

            audio_chunks = []
            frames_recorded = int(self.sr * duration / self.chunk_size)

            for i in range(frames_recorded):
                data = stream.read(self.chunk_size)
                audio_data = np.frombuffer(data, dtype=np.float32)
                audio_chunks.append(audio_data)

                # Optional: Process intermediate results
                if update_callback and i % 2 == 0:
                    full_audio = np.concatenate(audio_chunks)
                    transcript = self.stt.transcribe_raw_audio(full_audio)
                    if transcript:
                        update_callback(transcript, i / frames_recorded)

            stream.stop_stream()
            stream.close()

            # Final transcription
            full_audio = np.concatenate(audio_chunks)
            final_transcript = self.stt.transcribe_raw_audio(full_audio)

            logger.info(f"Recording complete: {final_transcript}")
            return final_transcript

        finally:
            p.terminate()


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("LSTM Speech-to-Text Inference")
    print("=" * 60)

    language = input("\nLanguage (english/kinyarwanda) [english]: ").strip().lower() or "english"
    model_path = input("Model path [models/trained/english_bidirectional_final.h5]: ").strip()

    if not model_path:
        model_path = f"models/trained/{language}_bidirectional_final.h5"

    # Initialize inference
    stt = STTInference(model_path, language)

    mode = input("\nMode (file/realtime) [file]: ").strip().lower() or "file"

    if mode == 'file':
        audio_file = input("Audio file path: ").strip()
        if os.path.exists(audio_file):
            result = stt.transcribe_with_confidence(audio_file)
            if result:
                print(f"\nTranscript: {result['transcript']}")
                print(f"Avg Confidence: {result['avg_confidence']:.2%}")
        else:
            print(f"File not found: {audio_file}")

    elif mode == 'realtime':
        duration = int(input("Recording duration (seconds) [10]: ").strip() or "10")
        realtime_stt = RealTimeSTT(model_path, language)
        transcript = realtime_stt.stream_transcribe(duration)
        print(f"\nFinal Transcript: {transcript}")
