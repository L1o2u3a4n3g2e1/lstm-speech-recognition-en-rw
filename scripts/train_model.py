import os
import sys
import numpy as np
import logging
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lstm_model import LSTMSpeechRecognition, CTCLSTMModel
from preprocess_data import DataPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_dataset(language='english', processed_dir=None):
    """Load preprocessed dataset."""
    if processed_dir is None:
        processed_dir = f'data/{language}/processed'

    logger.info(f"Loading dataset from {processed_dir}")

    try:
        X_train = np.load(os.path.join(processed_dir, 'train', 'X_train.npy'))
        y_train = np.load(os.path.join(processed_dir, 'train', 'y_train.npy'))
        X_val = np.load(os.path.join(processed_dir, 'val', 'X_val.npy'))
        y_val = np.load(os.path.join(processed_dir, 'val', 'y_val.npy'))
        X_test = np.load(os.path.join(processed_dir, 'test', 'X_test.npy'))
        y_test = np.load(os.path.join(processed_dir, 'test', 'y_test.npy'))

        logger.info(f"Dataset loaded!")
        logger.info(f"Train: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"Val: X={X_val.shape}, y={y_val.shape}")
        logger.info(f"Test: X={X_test.shape}, y={y_test.shape}")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    except FileNotFoundError as e:
        logger.error(f"Dataset files not found: {e}")
        return None


def train_model(language='english', model_type='encoder_decoder', epochs=50, batch_size=32):
    """Train LSTM speech recognition model."""

    logger.info(f"Starting training for {language.upper()}")
    logger.info(f"Model type: {model_type}, Epochs: {epochs}, Batch size: {batch_size}")

    # Load data
    data = load_dataset(language)
    if data is None:
        logger.error("Failed to load dataset. Run preprocess_data.py first.")
        return

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data

    # Transpose X from (batch, features, time_steps) to (batch, time_steps, features)
    X_train = np.transpose(X_train, (0, 2, 1))
    X_val = np.transpose(X_val, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    # Load vocabulary to get actual vocab size
    import json
    vocab_path = f'data/{language}/vocabulary.json'
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
        vocab_size = len(vocab_data['char_to_num'])

    # Pad y to match input time dimension (300 frames)
    # Pad with PAD token (index 0)
    pad_idx = 0
    target_seq_len = X_train.shape[1]  # 300

    def pad_sequences(y, target_length, pad_idx=0):
        padded = np.full((y.shape[0], target_length), pad_idx, dtype=y.dtype)
        for i in range(y.shape[0]):
            seq_len = min(y.shape[1], target_length)
            padded[i, :seq_len] = y[i, :seq_len]
        return padded

    y_train = pad_sequences(y_train, target_seq_len, pad_idx)
    y_val = pad_sequences(y_val, target_seq_len, pad_idx)
    y_test = pad_sequences(y_test, target_seq_len, pad_idx)

    # One-hot encode y to match model output shape (batch, time_steps, vocab_size)
    y_train = keras.utils.to_categorical(y_train, num_classes=vocab_size)
    y_val = keras.utils.to_categorical(y_val, num_classes=vocab_size)
    y_test = keras.utils.to_categorical(y_test, num_classes=vocab_size)

    # Create model
    input_shape = (X_train.shape[1], X_train.shape[2])

    logger.info(f"Input shape: {input_shape}, Vocab size: {vocab_size}")

    if model_type == 'encoder_decoder':
        model = LSTMSpeechRecognition(
            input_shape=input_shape,
            vocab_size=vocab_size,
            embedding_dim=256,
            lstm_units=512,
            dropout_rate=0.3,
            learning_rate=0.001
        )
        model.build_encoder_decoder()

    elif model_type == 'bidirectional':
        model = LSTMSpeechRecognition(
            input_shape=input_shape,
            vocab_size=vocab_size,
            embedding_dim=256,
            lstm_units=512,
            dropout_rate=0.3,
            learning_rate=0.001
        )
        model.build_bidirectional()

    elif model_type == 'ctc':
        model = CTCLSTMModel(
            input_shape=input_shape,
            vocab_size=vocab_size,
            lstm_units=512,
            dropout_rate=0.3,
            learning_rate=0.001
        )
        model.build()

    else:
        logger.error(f"Unknown model type: {model_type}")
        return

    # Display model summary
    model.get_summary()

    # Compile model
    model.compile_model()

    # Create checkpoint directory
    checkpoint_dir = f'models/checkpoints/{language}_{model_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Train model
    logger.info(f"Training on {len(X_train)} samples...")
    history = model.train(
        X_train, y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir
    )

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    model.evaluate(X_test, y_test)

    # Save final model
    final_model_path = f'models/trained/{language}_{model_type}_final.h5'
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    model.save_model(final_model_path)

    logger.info(f"Training complete! Model saved to {final_model_path}")

    return model, history


def train_bilingual_model(epochs=50, batch_size=32):
    """Train separate models for English and Kinyarwanda."""

    logger.info("Starting bilingual training pipeline")

    results = {}

    for language in ['english', 'kinyarwanda']:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {language.upper()} model")
        logger.info(f"{'='*60}\n")

        try:
            model, history = train_model(
                language=language,
                model_type='bidirectional',
                epochs=epochs,
                batch_size=batch_size
            )
            results[language] = {'status': 'success', 'model': model, 'history': history}

        except Exception as e:
            logger.error(f"Error training {language} model: {e}")
            results[language] = {'status': 'failed', 'error': str(e)}

    logger.info(f"\n{'='*60}")
    logger.info("Bilingual training complete!")
    logger.info(f"{'='*60}")

    for lang, result in results.items():
        status = result['status'].upper()
        logger.info(f"{lang.upper()}: {status}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("LSTM Speech Recognition - Training Script")
    print("=" * 60)

    language = input("\nLanguage (english/kinyarwanda/both): ").strip().lower()
    model_type = input("Model type (encoder_decoder/bidirectional/ctc) [bidirectional]: ").strip().lower() or "bidirectional"
    epochs = int(input("Epochs [50]: ").strip() or "50")
    batch_size = int(input("Batch size [32]: ").strip() or "32")

    if language == 'both':
        train_bilingual_model(epochs=epochs, batch_size=batch_size)
    elif language in ['english', 'kinyarwanda']:
        train_model(language=language, model_type=model_type, epochs=epochs, batch_size=batch_size)
    else:
        logger.error("Invalid language selection")
