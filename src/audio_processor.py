import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, sr=16000, duration=5, n_mfcc=13):
        self.sr = sr
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.n_fft = 400
        self.hop_length = 160

    def load_audio(self, filepath):
        """Load audio file and resample to target sample rate."""
        try:
            audio, sr = librosa.load(filepath, sr=self.sr, mono=True)
            logger.info(f"Loaded {filepath}: {len(audio)} samples at {sr}Hz")
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return None, None

    def normalize_audio(self, audio):
        """Normalize audio to [-1, 1] range."""
        if len(audio) == 0:
            return audio
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio

    def remove_silence(self, audio, threshold_db=-40):
        """Remove silence from audio using energy thresholding."""
        S = librosa.feature.melspectrogram(y=audio, sr=self.sr)
        S_db = librosa.power_to_db(S, ref=np.max)

        energy = np.mean(S_db, axis=0)
        mask = energy > threshold_db

        frame_length = self.hop_length
        start_idx = np.where(mask)[0]
        if len(start_idx) == 0:
            return audio

        start = start_idx[0] * frame_length
        end = (start_idx[-1] + 1) * frame_length

        return audio[start:end]

    def pad_or_trim(self, audio, target_length=None):
        """Pad or trim audio to fixed length."""
        if target_length is None:
            target_length = self.sr * self.duration

        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            pad_amount = target_length - len(audio)
            audio = np.pad(audio, (0, pad_amount), mode='constant', constant_values=0)

        return audio

    def extract_mfcc(self, audio):
        """Extract MFCC features from audio."""
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=self.n_mfcc,
                                    n_fft=self.n_fft, hop_length=self.hop_length)
        mfcc = np.log(np.abs(mfcc) + 1e-9)
        return mfcc

    def extract_spectrogram(self, audio):
        """Extract mel-spectrogram features."""
        S = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_fft=self.n_fft,
                                           hop_length=self.hop_length, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        return S_db
      
    def preprocess_audio(self, filepath, remove_silence_flag=True):
        """Complete preprocessing pipeline."""
        audio, sr = self.load_audio(filepath)
        if audio is None:
            return None

        audio = self.normalize_audio(audio)

        if remove_silence_flag:
            audio = self.remove_silence(audio)

        audio = self.pad_or_trim(audio)
        mfcc = self.extract_mfcc(audio)
        spectrogram = self.extract_spectrogram(audio)

        return {
            'audio': audio,
            'mfcc': mfcc,
            'spectrogram': spectrogram,
            'length': len(audio)
        }

    def batch_preprocess(self, filepath_list, feature_type='mfcc'):
        """Process multiple audio files and extract features."""
        features = []
        valid_files = []

        for filepath in filepath_list:
            processed = self.preprocess_audio(filepath)
            if processed is not None:
                if feature_type == 'mfcc':
                    features.append(processed['mfcc'])
                elif feature_type == 'spectrogram':
                    features.append(processed['spectrogram'])
                valid_files.append(filepath)

        return np.array(features), valid_files

    def augment_audio(self, audio):
        """Data augmentation: pitch shift and time stretch."""
        audio_aug = audio.copy()

        # Pitch shift by -2 to +2 semitones
        shift = np.random.randint(-2, 3)
        if shift != 0:
            audio_aug = librosa.effects.pitch_shift(audio_aug, sr=self.sr, n_steps=shift)

        # Time stretch by 0.9x to 1.1x
        rate = np.random.uniform(0.9, 1.1)
        audio_aug = librosa.effects.time_stretch(audio_aug, rate=rate)

        return self.pad_or_trim(audio_aug)
