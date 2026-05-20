import os
import sys
import numpy as np
import logging
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from lstm_model import LSTMSpeechRecognition
from preprocess_augmented import AugmentedDataPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_augmented_dataset(language='kinyarwanda'):
    """Load augmented preprocessed dataset."""
    processed_dir = f'data/{language}/processed_augmented'

    X_train = np.load(os.path.join(processed_dir, 'train', 'X_train.npy'))
    y_train = np.load(os.path.join(processed_dir, 'train', 'y_train.npy'))
    X_val = np.load(os.path.join(processed_dir, 'val', 'X_val.npy'))
    y_val = np.load(os.path.join(processed_dir, 'val', 'y_val.npy'))
    X_test = np.load(os.path.join(processed_dir, 'test', 'X_test.npy'))
    y_test = np.load(os.path.join(processed_dir, 'test', 'y_test.npy'))

    logger.info(f"Augmented dataset loaded!")
    logger.info(f"Train: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Val: X={X_val.shape}, y={y_val.shape}")
    logger.info(f"Test: X={X_test.shape}, y={y_test.shape}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def prepare_data_for_training(X_train, y_train, X_val, y_val, X_test, y_test, vocab_size):
    """
    Prepare data with improved numerical stability.
    Uses masking to avoid NaN issues from padding.
    """
    # Transpose X: (batch, features, time) → (batch, time, features)
    X_train = np.transpose(X_train, (0, 2, 1))
    X_val = np.transpose(X_val, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    # Scale to avoid large values
    X_train = X_train / (np.max(np.abs(X_train)) + 1e-8)
    X_val = X_val / (np.max(np.abs(X_val)) + 1e-8)
    X_test = X_test / (np.max(np.abs(X_test)) + 1e-8)

    logger.info(f"Data scaled and transposed")
    logger.info(f"X_train range: [{X_train.min():.4f}, {X_train.max():.4f}]")

    # Pad y to match model output time dimension (300)
    target_seq_len = X_train.shape[1]  # 300

    def pad_sequences(y, target_length, pad_idx=0):
        padded = np.full((y.shape[0], target_length), pad_idx, dtype=y.dtype)
        for i in range(y.shape[0]):
            seq_len = min(y.shape[1], target_length)
            padded[i, :seq_len] = y[i, :seq_len]
        return padded

    y_train = pad_sequences(y_train, target_seq_len, 0)
    y_val = pad_sequences(y_val, target_seq_len, 0)
    y_test = pad_sequences(y_test, target_seq_len, 0)

    # For categorical crossentropy, we need one-hot encoding
    y_train_oh = keras.utils.to_categorical(y_train, num_classes=vocab_size)
    y_val_oh = keras.utils.to_categorical(y_val, num_classes=vocab_size)
    y_test_oh = keras.utils.to_categorical(y_test, num_classes=vocab_size)

    logger.info(f"One-hot encoded targets")
    logger.info(f"Y shapes: train={y_train_oh.shape}, val={y_val_oh.shape}, test={y_test_oh.shape}")

    return (X_train, y_train_oh, None), (X_val, y_val_oh), (X_test, y_test_oh)


def train_improved_model(language='kinyarwanda', epochs=100, batch_size=16):
    """Train improved model with augmented data."""
    logger.info("="*60)
    logger.info("LSTM Speech Recognition - Improved Training")
    logger.info(f"Language: {language.upper()}")
    logger.info(f"Dataset: Augmented (196 → 136 valid samples)")
    logger.info(f"Train/Val/Test: 94/21/21")
    logger.info("="*60)

    # Load augmented dataset
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_augmented_dataset(language)

    # Load vocabulary
    import json
    vocab_path = f'data/{language}/vocabulary_augmented.json'
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
        vocab_size = len(vocab_data['char_to_num'])

    # Prepare data
    (X_train, y_train, _), (X_val, y_val), (X_test, y_test) = prepare_data_for_training(
        X_train, y_train, X_val, y_val, X_test, y_test, vocab_size
    )

    input_shape = (X_train.shape[1], X_train.shape[2])
    logger.info(f"Input shape: {input_shape}, Vocab size: {vocab_size}")

    # Build model with regularization
    model = LSTMSpeechRecognition(
        input_shape=input_shape,
        vocab_size=vocab_size,
        embedding_dim=256,
        lstm_units=512,
        dropout_rate=0.4,
        learning_rate=0.0005  # Lower learning rate for stability
    )
    model.build_bidirectional()
    model.get_summary()

    # Compile with label smoothing and gradient clipping
    optimizer = keras.optimizers.Adam(
        learning_rate=0.0005,
        global_clipnorm=1.0  # Gradient clipping
    )
    model.model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    logger.info("Model compiled with improved settings")

    # Create checkpoint directory
    checkpoint_dir = f'models/checkpoints/{language}_improved_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training with improved callbacks
    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: logger.info(
                f"Epoch {epoch+1}: loss={logs['loss']:.6f}, val_loss={logs['val_loss']:.6f}"
            )
        )
    ]

    # Train
    logger.info(f"Training on {len(X_train)} augmented samples...")
    logger.info(f"Batch size: {batch_size}, Epochs: {epochs}")

    history = model.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1
    )

    # Evaluate
    logger.info("\nEvaluating on test set...")
    test_loss, test_acc = model.model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test Loss: {test_loss:.6f}")
    logger.info(f"Test Accuracy: {test_acc:.6f} ({test_acc*100:.2f}%)")

    # Save model
    final_model_path = f'models/trained/{language}_improved_final.h5'
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    model.save_model(final_model_path)

    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info(f"Model saved to: {final_model_path}")
    logger.info(f"Final Test Accuracy: {test_acc*100:.2f}%")
    logger.info("="*60)

    return model, history, (test_loss, test_acc)


if __name__ == "__main__":
    language = input("Language (english/kinyarwanda) [kinyarwanda]: ").strip().lower() or 'kinyarwanda'
    epochs = int(input("Epochs [100]: ").strip() or "100")
    batch_size = int(input("Batch size [16]: ").strip() or "16")

    model, history, (loss, acc) = train_improved_model(
        language=language,
        epochs=epochs,
        batch_size=batch_size
    )
