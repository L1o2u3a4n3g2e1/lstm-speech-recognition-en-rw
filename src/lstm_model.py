import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMSpeechRecognition:
    def __init__(self, input_shape=(None, 13), vocab_size=128, embedding_dim=256,
                 lstm_units=512, dropout_rate=0.3, learning_rate=0.001):
        """
        Initialize LSTM Speech-to-Text model.

        Args:
            input_shape: Shape of input (time_steps, n_mfcc)
            vocab_size: Size of character vocabulary (for CTC loss)
            embedding_dim: Embedding dimension
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate
            learning_rate: Learning rate for optimizer
        """
        self.input_shape = input_shape
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.char_to_num = None
        self.num_to_char = None

    def build_encoder_decoder(self):
        """Build encoder-decoder LSTM architecture."""
        logger.info("Building Encoder-Decoder LSTM model...")

        inputs = layers.Input(shape=self.input_shape, name='audio_input')
        x = inputs

        # Encoder: Bidirectional LSTM layers
        x = layers.LSTM(self.lstm_units, return_sequences=True, activation='relu',
                       name='encoder_lstm_1')(x)
        x = layers.Dropout(self.dropout_rate)(x)

        x = layers.LSTM(self.lstm_units, return_sequences=True, activation='relu',
                       name='encoder_lstm_2')(x)
        x = layers.Dropout(self.dropout_rate)(x)

        # Bidirectional wrapper
        x = layers.Bidirectional(
            layers.LSTM(self.lstm_units // 2, return_sequences=True, activation='relu',
                       name='encoder_bilstm')
        )(x)
        x = layers.Dropout(self.dropout_rate)(x)

        # Decoder: LSTM layers
        x = layers.LSTM(self.lstm_units, return_sequences=True, activation='relu',
                       name='decoder_lstm_1')(x)
        x = layers.Dropout(self.dropout_rate)(x)

        x = layers.LSTM(self.lstm_units, return_sequences=True, activation='relu',
                       name='decoder_lstm_2')(x)
        x = layers.Dropout(self.dropout_rate)(x)

        # Output layer (character-level predictions)
        outputs = layers.Dense(self.vocab_size, activation='softmax',
                              name='character_output')(x)

        self.model = models.Model(inputs=inputs, outputs=outputs, name='LSTM_Speech_Recognition')
        logger.info("Model built successfully!")

        return self.model

    def build_bidirectional(self):
        """Build Bidirectional LSTM architecture."""
        logger.info("Building Bidirectional LSTM model...")

        inputs = layers.Input(shape=self.input_shape, name='audio_input')
        x = inputs

        # Bidirectional LSTM layers
        x = layers.Bidirectional(
            layers.LSTM(self.lstm_units, return_sequences=True, activation='relu'),
            name='bilstm_1'
        )(x)
        x = layers.Dropout(self.dropout_rate)(x)

        x = layers.Bidirectional(
            layers.LSTM(self.lstm_units, return_sequences=True, activation='relu'),
            name='bilstm_2'
        )(x)
        x = layers.Dropout(self.dropout_rate)(x)

        # Attention mechanism
        x = layers.MultiHeadAttention(num_heads=8, key_dim=64, dropout=self.dropout_rate)(x, x)

        x = layers.Bidirectional(
            layers.LSTM(self.lstm_units // 2, return_sequences=True, activation='relu'),
            name='bilstm_3'
        )(x)
        x = layers.Dropout(self.dropout_rate)(x)

        # Output layer
        outputs = layers.Dense(self.vocab_size, activation='softmax',
                              name='character_output')(x)

        self.model = models.Model(inputs=inputs, outputs=outputs, name='BiLSTM_Speech_Recognition')
        logger.info("Model built successfully!")

        return self.model

    def compile_model(self):
        """Compile the model."""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_encoder_decoder() or build_bidirectional() first.")

        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        logger.info("Model compiled!")

    def get_summary(self):
        """Print model summary."""
        if self.model is None:
            raise ValueError("Model not built yet.")
        self.model.summary()

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32,
             checkpoint_dir='models/checkpoints'):
        """
        Train the model.

        Args:
            X_train: Training audio features (num_samples, time_steps, n_mfcc)
            y_train: Training labels (num_samples, time_steps, vocab_size)
            X_val: Validation audio features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            checkpoint_dir: Directory to save checkpoints
        """
        if self.model is None:
            raise ValueError("Model not built yet.")

        import os
        os.makedirs(checkpoint_dir, exist_ok=True)

        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            callbacks.ModelCheckpoint(
                os.path.join(checkpoint_dir, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True
            ),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
        ]

        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )

        logger.info("Training complete!")
        return history

    def evaluate(self, X_test, y_test):
        """Evaluate model on test set."""
        if self.model is None:
            raise ValueError("Model not built yet.")

        results = self.model.evaluate(X_test, y_test, verbose=1)
        logger.info(f"Test Loss: {results[0]:.4f}, Test Accuracy: {results[1]:.4f}")
        return results

    def predict(self, X, return_probs=False):
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not built yet.")

        predictions = self.model.predict(X, verbose=0)

        if return_probs:
            return predictions
        else:
            return np.argmax(predictions, axis=-1)

    def save_model(self, filepath):
        """Save trained model."""
        if self.model is None:
            raise ValueError("Model not built yet.")
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load trained model."""
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")


class CTCLSTMModel:
    """LSTM model using CTC (Connectionist Temporal Classification) loss."""

    def __init__(self, input_shape=(None, 13), vocab_size=128, lstm_units=512,
                 dropout_rate=0.3, learning_rate=0.001):
        self.input_shape = input_shape
        self.vocab_size = vocab_size
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None

    def build(self):
        """Build CTC-based LSTM model."""
        logger.info("Building CTC LSTM model...")

        inputs = layers.Input(shape=self.input_shape, name='audio_input')
        x = inputs

        # Feature extraction
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)

        # Bidirectional LSTM layers
        x = layers.Bidirectional(
            layers.LSTM(self.lstm_units, return_sequences=True, activation='relu'),
            name='bilstm_1'
        )(x)
        x = layers.Dropout(self.dropout_rate)(x)

        x = layers.Bidirectional(
            layers.LSTM(self.lstm_units // 2, return_sequences=True, activation='relu'),
            name='bilstm_2'
        )(x)

        # Output layer (without softmax - CTC handles it)
        outputs = layers.Dense(self.vocab_size, activation='softmax',
                              name='character_output')(x)

        self.model = models.Model(inputs=inputs, outputs=outputs, name='CTC_LSTM')
        logger.info("CTC LSTM model built!")

        return self.model

    def compile_model(self):
        """Compile with CTC loss."""
        if self.model is None:
            raise ValueError("Model not built yet.")

        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        logger.info("CTC Model compiled!")

    def save_model(self, filepath):
        """Save model."""
        if self.model is None:
            raise ValueError("Model not built yet.")
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load model."""
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
