import os
import json
import csv
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from audio_processor import AudioProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, language='english', sr=16000, n_mfcc=13):
        self.language = language.lower()
        self.sr = sr
        self.audio_processor = AudioProcessor(sr=sr, n_mfcc=n_mfcc)
        self.data_dir = f"data/{language}"
        self.raw_dir = os.path.join(self.data_dir, 'raw')
        self.processed_dir = os.path.join(self.data_dir, 'processed')

        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(os.path.join(self.processed_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.processed_dir, 'val'), exist_ok=True)
        os.makedirs(os.path.join(self.processed_dir, 'test'), exist_ok=True)

    def scan_audio_files(self, source_dir):
        """Scan directory for audio files."""
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
        audio_files = []

        for ext in audio_extensions:
            audio_files.extend(Path(source_dir).rglob(f'*{ext}'))

        logger.info(f"Found {len(audio_files)} audio files in {source_dir}")
        return audio_files

    def process_kaggle_dataset(self, dataset_path):
        """Process Kaggle speech dataset (e.g., Librispeech, Common Voice)."""
        logger.info(f"Processing Kaggle dataset from {dataset_path}")

        audio_files = self.scan_audio_files(dataset_path)
        processed_data = []

        for idx, audio_file in enumerate(audio_files):
            if idx % 100 == 0:
                logger.info(f"Processing file {idx}/{len(audio_files)}: {audio_file.name}")

            result = self.audio_processor.preprocess_audio(str(audio_file), remove_silence_flag=True)
            if result is not None:
                processed_data.append({
                    'source': 'kaggle',
                    'filename': audio_file.name,
                    'filepath': str(audio_file),
                    'mfcc': result['mfcc'].tolist(),
                    'audio_length': result['length'],
                    'transcript': 'UNKNOWN'  # Kaggle datasets need external transcripts
                })

        return processed_data

    def process_custom_dataset(self, custom_dir):
        """Process custom recorded dataset with transcripts CSV."""
        logger.info(f"Processing custom dataset from {custom_dir}")

        if not os.path.exists(custom_dir):
            logger.warning(f"Directory not found: {custom_dir}")
            return []

        # Load transcripts from CSV
        transcript_csv = os.path.join(custom_dir, 'transcripts.csv')
        transcripts = {}

        if os.path.exists(transcript_csv):
            with open(transcript_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    transcripts[row['filename']] = row['transcript']
        else:
            logger.warning(f"No transcripts.csv found in {custom_dir}")

        audio_files = self.scan_audio_files(custom_dir)
        processed_data = []

        for idx, audio_file in enumerate(audio_files):
            if idx % 50 == 0:
                logger.info(f"Processing {idx}/{len(audio_files)}: {audio_file.name}")

            result = self.audio_processor.preprocess_audio(str(audio_file), remove_silence_flag=True)
            if result is not None:
                transcript = transcripts.get(audio_file.name, 'UNKNOWN')
                processed_data.append({
                    'source': 'custom',
                    'filename': audio_file.name,
                    'filepath': str(audio_file),
                    'mfcc': result['mfcc'].tolist(),
                    'audio_length': result['length'],
                    'transcript': transcript
                })

        logger.info(f"Processed {len(processed_data)} custom samples")
        return processed_data

    def process_digital_umuganda(self, dataset_path):
        """Process Digital Umuganda Kinyarwanda dataset."""
        logger.info(f"Processing Digital Umuganda dataset from {dataset_path}")

        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset path not found: {dataset_path}")
            return []

        audio_files = self.scan_audio_files(dataset_path)
        processed_data = []

        for idx, audio_file in enumerate(audio_files):
            if idx % 100 == 0:
                logger.info(f"Processing {idx}/{len(audio_files)}")

            result = self.audio_processor.preprocess_audio(str(audio_file), remove_silence_flag=True)
            if result is not None:
                processed_data.append({
                    'source': 'digital_umuganda',
                    'filename': audio_file.name,
                    'filepath': str(audio_file),
                    'mfcc': result['mfcc'].tolist(),
                    'audio_length': result['length'],
                    'transcript': 'UNKNOWN'
                })

        return processed_data

    def process_huggingface_dataset(self, dataset_path):
        """Process Hugging Face speech dataset."""
        logger.info(f"Processing Hugging Face dataset from {dataset_path}")

        audio_files = self.scan_audio_files(dataset_path)
        processed_data = []

        for idx, audio_file in enumerate(audio_files):
            if idx % 100 == 0:
                logger.info(f"Processing {idx}/{len(audio_files)}")

            result = self.audio_processor.preprocess_audio(str(audio_file), remove_silence_flag=True)
            if result is not None:
                processed_data.append({
                    'source': 'huggingface',
                    'filename': audio_file.name,
                    'filepath': str(audio_file),
                    'mfcc': result['mfcc'].tolist(),
                    'audio_length': result['length'],
                    'transcript': 'UNKNOWN'
                })

        return processed_data

    def normalize_features(self, mfcc_features_list):
        """Normalize MFCC features across dataset."""
        all_features = np.array([f for f in mfcc_features_list if f is not None])

        mean = np.mean(all_features)
        std = np.std(all_features)

        logger.info(f"Normalization stats - Mean: {mean:.4f}, Std: {std:.4f}")

        return mean, std

    def pad_features(self, features, max_length=300):
        """Pad or truncate MFCC features to fixed length."""
        padded = []

        for feat in features:
            if feat.shape[1] > max_length:
                feat = feat[:, :max_length]
            else:
                pad_width = ((0, 0), (0, max_length - feat.shape[1]))
                feat = np.pad(feat, pad_width, mode='constant', constant_values=0)

            padded.append(feat)

        return np.array(padded)

    def create_vocabulary(self, transcripts_list):
        """Create character-level vocabulary from transcripts."""
        unique_chars = set()

        for transcript in transcripts_list:
            if transcript and transcript != 'UNKNOWN':
                unique_chars.update(transcript.lower())

        # Add special characters
        unique_chars.add(' ')
        unique_chars.add('<PAD>')
        unique_chars.add('<UNK>')

        char_to_num = {char: idx for idx, char in enumerate(sorted(unique_chars))}
        num_to_char = {idx: char for char, idx in char_to_num.items()}

        logger.info(f"Vocabulary size: {len(char_to_num)}")

        return char_to_num, num_to_char

    def encode_transcripts(self, transcripts, char_to_num, max_length=200):
        """Encode transcripts to numeric sequences."""
        encoded = []

        for transcript in transcripts:
            if transcript == 'UNKNOWN':
                seq = [char_to_num.get('<UNK>', 0)] * max_length
            else:
                seq = [char_to_num.get(c, char_to_num.get('<UNK>', 0))
                      for c in transcript.lower()]
                seq = seq[:max_length]
                seq += [char_to_num.get('<PAD>', 0)] * (max_length - len(seq))

            encoded.append(seq)

        return np.array(encoded)

    def split_data(self, X, y, train_ratio=0.7, val_ratio=0.15):
        """Split data into train/val/test sets."""
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=1 - train_ratio - val_ratio, random_state=42
        )

        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio_adjusted, random_state=42
        )

        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def save_dataset(self, X_train, y_train, X_val, y_val, X_test, y_test,
                    char_to_num, num_to_char, split_name=''):
        """Save preprocessed dataset and vocabulary."""
        np.save(os.path.join(self.processed_dir, 'train', 'X_train.npy'), X_train)
        np.save(os.path.join(self.processed_dir, 'train', 'y_train.npy'), y_train)
        np.save(os.path.join(self.processed_dir, 'val', 'X_val.npy'), X_val)
        np.save(os.path.join(self.processed_dir, 'val', 'y_val.npy'), y_val)
        np.save(os.path.join(self.processed_dir, 'test', 'X_test.npy'), X_test)
        np.save(os.path.join(self.processed_dir, 'test', 'y_test.npy'), y_test)

        # Save vocabulary
        vocab_file = os.path.join(self.data_dir, 'vocabulary.json')
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump({
                'char_to_num': char_to_num,
                'num_to_char': {str(k): v for k, v in num_to_char.items()}
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"Dataset saved to {self.processed_dir}")
        logger.info(f"Vocabulary saved to {vocab_file}")

    def load_preprocessed_data(self):
        """Load preprocessed train/val/test data."""
        X_train = np.load(os.path.join(self.processed_dir, 'train', 'X_train.npy'))
        y_train = np.load(os.path.join(self.processed_dir, 'train', 'y_train.npy'))
        X_val = np.load(os.path.join(self.processed_dir, 'val', 'X_val.npy'))
        y_val = np.load(os.path.join(self.processed_dir, 'val', 'y_val.npy'))
        X_test = np.load(os.path.join(self.processed_dir, 'test', 'X_test.npy'))
        y_test = np.load(os.path.join(self.processed_dir, 'test', 'y_test.npy'))

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def preprocess_pipeline(self, sources_config):
        """
        Complete preprocessing pipeline.

        Args:
            sources_config: Dict with paths to different data sources
                {
                    'kaggle': '/path/to/kaggle/dataset',
                    'custom': '/path/to/custom/recordings',
                    'digital_umuganda': '/path/to/digital_umuganda',
                    'huggingface': '/path/to/huggingface/dataset'
                }
        """
        all_data = []

        # Process each source
        if sources_config.get('kaggle'):
            data = self.process_kaggle_dataset(sources_config['kaggle'])
            all_data.extend(data)

        if sources_config.get('custom'):
            data = self.process_custom_dataset(sources_config['custom'])
            all_data.extend(data)

        if sources_config.get('digital_umuganda'):
            data = self.process_digital_umuganda(sources_config['digital_umuganda'])
            all_data.extend(data)

        if sources_config.get('huggingface'):
            data = self.process_huggingface_dataset(sources_config['huggingface'])
            all_data.extend(data)

        logger.info(f"Total samples processed: {len(all_data)}")

        if len(all_data) == 0:
            logger.error("No data processed! Check source paths.")
            return

        # Extract features and transcripts
        X = self.pad_features([np.array(d['mfcc']) for d in all_data], max_length=300)
        transcripts = [d['transcript'] for d in all_data]

        # Create vocabulary and encode
        char_to_num, num_to_char = self.create_vocabulary(transcripts)
        y = self.encode_transcripts(transcripts, char_to_num, max_length=200)

        # Split data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.split_data(X, y)

        # Save dataset
        self.save_dataset(X_train, y_train, X_val, y_val, X_test, y_test,
                         char_to_num, num_to_char)

        logger.info("Preprocessing pipeline complete!")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":
    language = input("Language (english/kinyarwanda): ").strip().lower()
    if language not in ['english', 'kinyarwanda']:
        language = 'english'

    preprocessor = DataPreprocessor(language=language)

    # Configure data sources
    sources_config = {
        'custom': f'data/{language}/raw/custom_recordings',
        # 'kaggle': 'data/kaggle/librispeech',  # Uncomment when available
        # 'digital_umuganda': 'data/kinyarwanda/digital_umuganda',  # For Kinyarwanda
        # 'huggingface': 'data/huggingface/common_voice',
    }

    preprocessor.preprocess_pipeline(sources_config)
