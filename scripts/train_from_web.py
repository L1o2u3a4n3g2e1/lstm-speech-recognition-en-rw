"""
Continuous Training Module for Web App Integration
Enables fine-tuning LSTM model from web interface
Handles single-sample training, batch training, and model checkpointing
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import librosa
import soundfile as sf

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.lstm_model import LSTMSpeechRecognition
from src.audio_processor import AudioProcessor


class ContinuousTrainer:
    """Handle model fine-tuning from web uploads"""

    def __init__(self, model_path=None, vocab_path=None):
        self.model = None
        self.vocab = None
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = model_path or os.path.join(base_dir, 'models', 'trained', 'kinyarwanda_masked_final.h5')
        self.vocab_path = vocab_path or os.path.join(base_dir, 'data', 'kinyarwanda', 'vocabulary.json')
        self.training_state = {
            'is_training': False,
            'epoch': 0,
            'loss': 0.0,
            'accuracy': 0.0,
            'samples_trained': 0
        }
        self.optimizer = None
        self.loss_fn = None
        self.audio_processor = AudioProcessor()

        # Load vocabulary
        self._load_vocab()

        # Load or initialize model
        self._load_model()

    def _load_vocab(self):
        """Load character vocabulary"""
        try:
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
                self.vocab = {int(k): v for k, v in vocab_data.items()}
        except FileNotFoundError:
            # Create default vocabulary if not found
            self.vocab = {
                0: ' ', 1: '.', 2: '<PAD>', 3: '<UNK>',
                4: 'a', 5: 'b', 6: 'c', 7: 'd', 8: 'e', 9: 'f',
                10: 'g', 11: 'h', 12: 'i', 13: 'j', 14: 'k', 15: 'l',
                16: 'm', 17: 'n', 18: 'o', 19: 'p', 20: 'q', 21: 'r',
                22: 's', 23: 't', 24: 'u', 25: 'v', 26: 'w'
            }

    def _load_model(self):
        """Load trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                # Set up optimizer and loss
                self._setup_training()
                return True
            else:
                print(f"Model not found at {self.model_path}")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def _setup_training(self):
        """Configure optimizer and loss for training"""
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.loss_fn = self._masked_categorical_crossentropy

    def _masked_categorical_crossentropy(self, y_true, y_pred):
        """Custom masked loss for padding tokens"""
        base_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        token_ids = tf.argmax(y_true, axis=-1)
        # Mask out padding (class 2)
        padding_mask = tf.equal(token_ids, 2)
        non_padding_mask = 1.0 - tf.cast(padding_mask, tf.float32)
        masked_loss = base_loss * non_padding_mask
        sum_loss = tf.reduce_sum(masked_loss, axis=1)
        count_non_padding = tf.reduce_sum(non_padding_mask, axis=1)
        count_non_padding = tf.maximum(count_non_padding, 1.0)
        mean_loss = sum_loss / count_non_padding
        return tf.reduce_mean(mean_loss)

    def train_on_audio(self, audio_path, transcript, learning_rate=0.0001):
        """Train model on single audio sample"""
        try:
            if not self.model:
                return False, "Model not loaded"

            # Process audio
            audio, sr = librosa.load(audio_path, sr=16000)
            mfcc = librosa.feature.mfcc(
                y=audio, sr=16000, n_mfcc=13, n_fft=512, hop_length=160
            )
            mfcc = np.transpose(mfcc)  # Shape: (time, 13)

            # Pad/truncate to 300 frames
            if mfcc.shape[0] < 300:
                mfcc = np.pad(mfcc, ((0, 300 - mfcc.shape[0]), (0, 0)))
            else:
                mfcc = mfcc[:300]

            # Convert transcript to labels
            labels = self._transcript_to_labels(transcript)
            if labels is None:
                return False, "Invalid transcript"

            # Train on this sample
            mfcc_batch = np.expand_dims(mfcc, axis=0)  # Shape: (1, 300, 13)
            labels_batch = np.expand_dims(labels, axis=0)  # Shape: (1, 300, 27)

            with tf.GradientTape() as tape:
                predictions = self.model(mfcc_batch, training=True)
                loss = self.loss_fn(labels_batch, predictions)

            # Update weights
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.learning_rate.assign(learning_rate)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            self.training_state['loss'] = float(loss.numpy())
            self.training_state['samples_trained'] += 1

            return True, f"Trained. Loss: {loss.numpy():.4f}"

        except Exception as e:
            return False, f"Error training: {str(e)}"

    def _transcript_to_labels(self, transcript):
        """Convert text transcript to one-hot encoded labels"""
        try:
            # Create reverse vocabulary mapping
            char_to_id = {v: k for k, v in self.vocab.items()}

            # Convert transcript to character IDs
            labels = []
            for char in transcript.lower():
                if char in char_to_id:
                    labels.append(char_to_id[char])
                else:
                    labels.append(char_to_id.get('<UNK>', 3))

            # Pad/truncate to 300
            if len(labels) < 300:
                labels = labels + [2] * (300 - len(labels))  # Pad with PAD token
            else:
                labels = labels[:300]

            # One-hot encode
            one_hot = np.zeros((300, 27))
            for i, label in enumerate(labels):
                if label < 27:
                    one_hot[i, label] = 1

            return one_hot

        except Exception as e:
            print(f"Error converting transcript: {e}")
            return None

    def batch_train(self, audio_files_with_transcripts, epochs=1, learning_rate=0.0001):
        """Train on multiple audio samples"""
        results = []

        for epoch in range(epochs):
            epoch_loss = 0
            for audio_path, transcript in audio_files_with_transcripts:
                success, message = self.train_on_audio(audio_path, transcript, learning_rate)
                if success:
                    epoch_loss += self.training_state['loss']
                results.append({'file': audio_path, 'transcript': transcript, 'success': success, 'message': message})

            avg_loss = epoch_loss / len(audio_files_with_transcripts)
            self.training_state['epoch'] = epoch + 1
            self.training_state['loss'] = avg_loss
            print(f"Epoch {epoch + 1}: Avg Loss = {avg_loss:.4f}")

        return results

    def save_model(self, output_path=None):
        """Save trained model"""
        try:
            output_path = output_path or self.model_path
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.model.save(output_path)
            return True, f"Model saved to {output_path}"
        except Exception as e:
            return False, f"Error saving model: {str(e)}"

    def get_training_state(self):
        """Get current training state"""
        return self.training_state


def train_on_uploaded_file(audio_path, transcript, model_path=None):
    """Quick function for web app to train on single file"""
    trainer = ContinuousTrainer(model_path)
    success, message = trainer.train_on_audio(audio_path, transcript)
    if success:
        trainer.save_model()
    return success, message


if __name__ == '__main__':
    # Example usage
    trainer = ContinuousTrainer()
    print("Continuous trainer initialized")
    print("Use train_on_audio() or batch_train() to train on new samples")
