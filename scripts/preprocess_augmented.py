import os
import sys
import numpy as np
import json
import csv
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from audio_processor import AudioProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AugmentedDataPreprocessor:
    """Preprocess augmented dataset with improved handling."""

    def __init__(self, language='kinyarwanda'):
        self.language = language.lower()
        self.data_dir = f'data/{language}'
        self.raw_dir = os.path.join(self.data_dir, 'raw')
        self.augmented_dir = os.path.join(self.raw_dir, 'augmented')
        self.processed_dir = os.path.join(self.data_dir, 'processed_augmented')

        os.makedirs(self.processed_dir, exist_ok=True)
        for subdir in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.processed_dir, subdir), exist_ok=True)

        self.processor = AudioProcessor(sr=16000, duration=5, n_mfcc=13)

    def process_augmented_dataset(self):
        """Process augmented audio files."""
        logger.info(f"Processing augmented dataset from {self.augmented_dir}")

        # Load augmented transcripts
        aug_csv = os.path.join(self.augmented_dir, 'transcripts_augmented.csv')
        transcripts_map = {}
        with open(aug_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                transcripts_map[row['filename']] = row['transcript']

        # Process all augmented files
        wav_files = sorted(Path(self.augmented_dir).glob('*.wav'))
        logger.info(f"Found {len(wav_files)} augmented audio files")

        mfcc_features = []
        transcripts = []
        valid_count = 0

        for idx, wav_file in enumerate(wav_files):
            filename = wav_file.name
            transcript = transcripts_map.get(filename, 'UNKNOWN')

            if transcript == 'UNKNOWN':
                logger.warning(f"No transcript for {filename}, skipping")
                continue

            # Process audio
            audio, sr = self.processor.load_audio(str(wav_file))
            if audio is None:
                logger.warning(f"Failed to load {filename}, skipping")
                continue

            audio = self.processor.normalize_audio(audio)
            audio = self.processor.remove_silence(audio)
            audio = self.processor.pad_or_trim(audio)

            mfcc = self.processor.extract_mfcc(audio)
            mfcc = np.log(np.abs(mfcc) + 1e-9)

            mfcc_features.append(mfcc)
            transcripts.append(transcript.lower())
            valid_count += 1

            if (idx + 1) % 20 == 0:
                logger.info(f"Processed {idx + 1}/{len(wav_files)} files")

        logger.info(f"Successfully processed {valid_count} augmented samples")

        # Pad MFCC features
        X = self._pad_features(mfcc_features, max_length=300)
        logger.info(f"Feature shape: {X.shape}")

        # Create vocabulary
        char_to_num, num_to_char = self._create_vocabulary(transcripts)
        logger.info(f"Vocabulary size: {len(char_to_num)}")

        # Encode transcripts
        y, actual_lengths = self._encode_transcripts(transcripts, char_to_num, max_length=200)
        logger.info(f"Target shape: {y.shape}")

        # Split data
        (X_train, y_train, len_train), (X_val, y_val, len_val), (X_test, y_test, len_test) = self._split_data(
            X, y, actual_lengths
        )

        # Save dataset
        self._save_dataset(X_train, y_train, len_train, X_val, y_val, len_val, X_test, y_test, len_test,
                          char_to_num, num_to_char)

        logger.info("Preprocessing augmented dataset complete!")

    def _pad_features(self, features, max_length=300):
        """Pad MFCC features to fixed length."""
        padded = []
        for feat in features:
            if feat.shape[1] > max_length:
                feat = feat[:, :max_length]
            else:
                pad_width = ((0, 0), (0, max_length - feat.shape[1]))
                feat = np.pad(feat, pad_width, mode='constant', constant_values=0)
            padded.append(feat)
        return np.array(padded)

    def _create_vocabulary(self, transcripts):
        """Create character vocabulary."""
        unique_chars = set()
        for transcript in transcripts:
            if transcript and transcript != 'UNKNOWN':
                unique_chars.update(transcript.lower())

        unique_chars.add(' ')
        unique_chars.add('<PAD>')
        unique_chars.add('<UNK>')

        char_to_num = {char: idx for idx, char in enumerate(sorted(unique_chars))}
        num_to_char = {idx: char for char, idx in char_to_num.items()}

        return char_to_num, num_to_char

    def _encode_transcripts(self, transcripts, char_to_num, max_length=200):
        """Encode transcripts with actual length tracking."""
        encoded = []
        actual_lengths = []

        for transcript in transcripts:
            seq = [char_to_num.get(c, char_to_num.get('<UNK>', 0)) for c in transcript.lower()]

            # Track actual length
            actual_len = min(len(seq), max_length)
            actual_lengths.append(actual_len)

            # Pad
            seq = seq[:max_length]
            seq += [char_to_num.get('<PAD>', 0)] * (max_length - len(seq))

            encoded.append(seq)

        return np.array(encoded), np.array(actual_lengths)

    def _split_data(self, X, y, lengths, train_ratio=0.7, val_ratio=0.15):
        """Split data with length tracking."""
        X_temp, X_test, y_temp, y_test, len_temp, len_test = train_test_split(
            X, y, lengths, test_size=1 - train_ratio - val_ratio, random_state=42
        )

        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val, len_train, len_val = train_test_split(
            X_temp, y_temp, len_temp, test_size=val_ratio_adjusted, random_state=42
        )

        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        return (X_train, y_train, len_train), (X_val, y_val, len_val), (X_test, y_test, len_test)

    def _save_dataset(self, X_train, y_train, len_train, X_val, y_val, len_val, X_test, y_test, len_test,
                     char_to_num, num_to_char):
        """Save preprocessed dataset."""
        # Save features
        np.save(os.path.join(self.processed_dir, 'train', 'X_train.npy'), X_train)
        np.save(os.path.join(self.processed_dir, 'train', 'y_train.npy'), y_train)
        np.save(os.path.join(self.processed_dir, 'train', 'lengths_train.npy'), len_train)

        np.save(os.path.join(self.processed_dir, 'val', 'X_val.npy'), X_val)
        np.save(os.path.join(self.processed_dir, 'val', 'y_val.npy'), y_val)
        np.save(os.path.join(self.processed_dir, 'val', 'lengths_val.npy'), len_val)

        np.save(os.path.join(self.processed_dir, 'test', 'X_test.npy'), X_test)
        np.save(os.path.join(self.processed_dir, 'test', 'y_test.npy'), y_test)
        np.save(os.path.join(self.processed_dir, 'test', 'lengths_test.npy'), len_test)

        # Save vocabulary
        vocab_file = os.path.join(self.data_dir, 'vocabulary_augmented.json')
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump({
                'char_to_num': char_to_num,
                'num_to_char': {str(k): v for k, v in num_to_char.items()}
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"Dataset saved to {self.processed_dir}")
        logger.info(f"Vocabulary saved to {vocab_file}")


if __name__ == "__main__":
    language = input("Language (english/kinyarwanda) [kinyarwanda]: ").strip().lower() or 'kinyarwanda'

    preprocessor = AugmentedDataPreprocessor(language=language)
    preprocessor.process_augmented_dataset()
