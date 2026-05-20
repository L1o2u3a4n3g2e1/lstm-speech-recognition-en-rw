"""Constants and configuration for LSTM Speech Recognition."""

# Audio Configuration
SAMPLE_RATE = 16000  # Hz - Standard for speech recognition
CHUNK_SIZE = 1024  # Frames per buffer
CHANNELS = 1  # Mono audio
AUDIO_FORMAT = 'float32'  # Audio format

# Feature Extraction
N_MFCC = 13  # Number of MFCC coefficients
N_FFT = 400  # FFT window size
HOP_LENGTH = 160  # Samples between frames
N_MELS = 128  # Number of mel bands

# Audio Processing
AUDIO_DURATION = 5  # Seconds per sample
TARGET_LENGTH = SAMPLE_RATE * AUDIO_DURATION  # 80,000 samples
SILENCE_THRESHOLD_DB = -40  # dB threshold for silence removal

# Feature Shapes
MAX_TIME_STEPS = 300  # Maximum time steps (padded/truncated)
FEATURE_DIM = N_MFCC  # Dimension of feature vector
MAX_TRANSCRIPT_LENGTH = 200  # Maximum character sequence length

# Model Configuration
VOCAB_SIZE = 128  # Character vocabulary size
EMBEDDING_DIM = 256  # Embedding dimension
LSTM_UNITS = 512  # LSTM units per layer
DROPOUT_RATE = 0.3  # Dropout rate
LEARNING_RATE = 0.001  # Initial learning rate

# Training Configuration
EPOCHS = 50  # Number of training epochs
BATCH_SIZE = 32  # Batch size for training
VALIDATION_SPLIT = 0.15  # Validation set ratio
TEST_SPLIT = 0.15  # Test set ratio
TRAIN_SPLIT = 0.70  # Training set ratio

# Early Stopping & Callbacks
EARLY_STOPPING_PATIENCE = 5  # Epochs without improvement before stopping
REDUCE_LR_PATIENCE = 3  # Epochs without improvement before reducing LR
REDUCE_LR_FACTOR = 0.5  # LR multiplication factor

# Data Augmentation
PITCH_SHIFT_RANGE = (-2, 3)  # Semitones
TIME_STRETCH_RANGE = (0.9, 1.1)  # Rate multiplier

# Languages
SUPPORTED_LANGUAGES = ['english', 'kinyarwanda']
LANGUAGE_CODES = {
    'english': 'en',
    'kinyarwanda': 'rw'
}

# Dataset Sources
DATASET_SOURCES = {
    'english': {
        'kaggle': 'librispeech',
        'mozilla': 'common_voice_en',
        'custom': 'custom_recordings'
    },
    'kinyarwanda': {
        'digital_umuganda': 'digital_umuganda_rw',
        'mozilla': 'common_voice_rw',
        'custom': 'custom_recordings'
    }
}

# Directory Paths
DATA_ROOT = 'data'
MODELS_ROOT = 'models'
LOGS_ROOT = 'logs'

# Special Tokens
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
START_TOKEN = '<START>'
END_TOKEN = '<END>'

# Character Set (Kinyarwanda & English)
KINYARWANDA_CHARS = set('abcdefghijklmnopqrstuvwxyzàâäçèéêëìîïñòôöœùûüœŵŷ ')
ENGLISH_CHARS = set('abcdefghijklmnopqrstuvwxyz ')

# Model Names
MODEL_TYPES = {
    'encoder_decoder': 'Encoder-Decoder LSTM',
    'bidirectional': 'Bidirectional LSTM',
    'ctc': 'CTC LSTM'
}

# Inference Configuration
INFERENCE_CHUNK_SIZE = CHUNK_SIZE  # Real-time inference chunk size
INFERENCE_BUFFER_SIZE = SAMPLE_RATE  # 1 second buffer
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for prediction

# Performance Targets
PERFORMANCE_TARGETS = {
    'english': {
        'accuracy': 0.90,
        'latency_ms': 100,
        'training_time_hours': 4
    },
    'kinyarwanda': {
        'accuracy': 0.80,
        'latency_ms': 100,
        'training_time_hours': 4
    }
}

# Logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'

# Device Configuration
USE_GPU = True  # Try to use GPU if available
GPU_MEMORY_FRACTION = 0.8  # Fraction of GPU memory to use

# Version
VERSION = '1.0.0'
PROJECT_NAME = 'LSTM Speech Recognition - English & Kinyarwanda'
