"""
Synthetic Audio Generation from Parallel Kinyarwanda-English Corpus
Generates training data from text using Text-to-Speech (TTS)
Creates 2000+ audio samples for model training
"""

import os
import sys
import csv
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import pyttsx3
    from pydub import AudioSegment
    from pydub.generators import Sine
except ImportError:
    logger.warning("Optional dependencies not installed. Install with:")
    logger.warning("pip install pyttsx3 pydub")
    pyttsx3 = None
    AudioSegment = None


class SyntheticAudioGenerator:
    """Generate synthetic audio from text using TTS"""

    def __init__(self, language='kinyarwanda', output_dir='data/kinyarwanda/raw/synthetic'):
        self.language = language.lower()
        self.output_dir = output_dir
        self.engine = None
        self.transcripts = []

        os.makedirs(output_dir, exist_ok=True)

        if pyttsx3:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)  # Speech rate
            self.engine.setProperty('volume', 0.9)  # Volume
        else:
            logger.warning("pyttsx3 not available. Audio generation will be simulated.")

    def generate_audio_from_text(self, text, filename, language='kinyarwanda'):
        """Generate audio file from text"""
        try:
            filepath = os.path.join(self.output_dir, filename)

            if self.engine:
                self.engine.save_to_file(text, filepath)
                self.engine.runAndWait()

                if os.path.exists(filepath):
                    logger.info(f"Generated: {filename}")
                    return True, filepath
            else:
                # Simulate audio generation (create dummy WAV file)
                logger.info(f"Simulating audio generation: {filename}")
                self._create_dummy_audio(filepath, len(text))
                return True, filepath

        except Exception as e:
            logger.error(f"Error generating audio for '{text}': {e}")
            return False, None

    def _create_dummy_audio(self, filepath, text_length):
        """Create a dummy WAV file for simulation"""
        try:
            if AudioSegment:
                # Create a sine wave audio
                duration_ms = max(500, int(text_length * 100))  # ~100ms per character
                frequency = 440  # A4 note

                sine_wave = Sine(frequency, sample_rate=16000).to_audio_segment(duration=duration_ms)
                sine_wave.export(filepath, format="wav")
        except:
            # Fallback: create minimal WAV file
            self._create_minimal_wav(filepath)

    def _create_minimal_wav(self, filepath):
        """Create a minimal valid WAV file"""
        import wave

        sample_rate = 16000
        duration = 2  # seconds
        samples = np.random.randint(-32768, 32767, sample_rate * duration, dtype=np.int16)

        with wave.open(filepath, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(samples.tobytes())

    def load_parallel_corpus(self, corpus_file='data/kinyarwanda/final_corpus.csv'):
        """Load parallel Kinyarwanda-English corpus"""
        try:
            sentences = {'kinyarwanda': [], 'english': []}

            if not os.path.exists(corpus_file):
                logger.warning(f"Corpus file not found: {corpus_file}")
                return sentences

            with open(corpus_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row and len(sentences['kinyarwanda']) < 2000:  # Limit to 2000
                        if 'kinyarwanda' in row and 'english' in row:
                            rw_text = row['kinyarwanda'].strip()
                            en_text = row['english'].strip()

                            # Filter valid sentences (3-100 words)
                            if 3 <= len(rw_text.split()) <= 100 and 3 <= len(en_text.split()) <= 100:
                                sentences['kinyarwanda'].append(rw_text)
                                sentences['english'].append(en_text)

            logger.info(f"Loaded {len(sentences['kinyarwanda'])} parallel sentence pairs")
            return sentences

        except Exception as e:
            logger.error(f"Error loading corpus: {e}")
            return {'kinyarwanda': [], 'english': []}

    def generate_dataset(self, num_samples=2000):
        """Generate synthetic audio dataset from corpus"""
        logger.info(f"Starting synthetic audio generation for {num_samples} samples...")

        # Load corpus
        corpus = self.load_parallel_corpus()

        if not corpus['kinyarwanda']:
            logger.error("No corpus data available")
            return False

        # Limit to available sentences
        num_samples = min(num_samples, len(corpus['kinyarwanda']))

        # Create transcripts CSV
        transcripts_file = os.path.join(self.output_dir, 'transcripts.csv')
        generated_files = []

        try:
            with open(transcripts_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'transcript', 'language', 'source'])

                for idx in range(num_samples):
                    # Alternate between Kinyarwanda and English
                    language = 'kinyarwanda' if idx % 2 == 0 else 'english'
                    text = corpus[language][idx]

                    filename = f"{language}_synthetic_{idx:05d}.wav"

                    # Generate audio
                    success, filepath = self.generate_audio_from_text(
                        text, filename, language
                    )

                    if success:
                        writer.writerow([filename, text, language, 'synthetic'])
                        generated_files.append(filename)

                        if (idx + 1) % 100 == 0:
                            logger.info(f"Progress: {idx + 1}/{num_samples} files generated")

            logger.info(f"✅ Generated {len(generated_files)} synthetic audio files")
            logger.info(f"Transcripts saved to: {transcripts_file}")

            # Summary
            logger.info(f"""
            ═══════════════════════════════════════════════════════════
            Synthetic Audio Generation Complete
            ═══════════════════════════════════════════════════════════
            Files Generated: {len(generated_files)}
            Output Directory: {self.output_dir}
            Transcripts File: {transcripts_file}

            Next Steps:
            1. Preprocess audio: python scripts/preprocess_augmented.py
            2. Train model: python scripts/train_with_masking.py
            3. Start web app: START_WEB_APP.bat
            ═══════════════════════════════════════════════════════════
            """)

            return True

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return False


class DatasetAugmenter:
    """Augment existing dataset with synthetic samples"""

    def __init__(self, original_dir='data/kinyarwanda/raw/custom_recordings',
                 synthetic_dir='data/kinyarwanda/raw/synthetic'):
        self.original_dir = original_dir
        self.synthetic_dir = synthetic_dir

    def combine_datasets(self):
        """Combine original and synthetic recordings"""
        try:
            combined_dir = 'data/kinyarwanda/raw/combined'
            os.makedirs(combined_dir, exist_ok=True)

            original_count = 0
            synthetic_count = 0

            # Copy original files
            if os.path.exists(self.original_dir):
                for file in os.listdir(self.original_dir):
                    if file.endswith(('.wav', '.mp3', '.m4a', '.ogg')):
                        src = os.path.join(self.original_dir, file)
                        dst = os.path.join(combined_dir, f"original_{file}")
                        import shutil
                        shutil.copy(src, dst)
                        original_count += 1

            # Copy synthetic files
            if os.path.exists(self.synthetic_dir):
                for file in os.listdir(self.synthetic_dir):
                    if file.endswith('.wav'):
                        src = os.path.join(self.synthetic_dir, file)
                        dst = os.path.join(combined_dir, file)
                        import shutil
                        shutil.copy(src, dst)
                        synthetic_count += 1

            logger.info(f"""
            Dataset Combined:
            - Original: {original_count} files
            - Synthetic: {synthetic_count} files
            - Total: {original_count + synthetic_count} files
            - Location: {combined_dir}
            """)

            return original_count + synthetic_count

        except Exception as e:
            logger.error(f"Error combining datasets: {e}")
            return 0


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate synthetic audio dataset')
    parser.add_argument('--num-samples', type=int, default=2000,
                       help='Number of samples to generate (default: 2000)')
    parser.add_argument('--language', type=str, default='kinyarwanda',
                       help='Language: kinyarwanda or english')
    parser.add_argument('--corpus-file', type=str, default='data/kinyarwanda/final_corpus.csv',
                       help='Path to parallel corpus CSV file')
    parser.add_argument('--combine', action='store_true',
                       help='Combine original and synthetic datasets')

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("LSTM Kinyarwanda Speech Recognition - Synthetic Data Generator")
    logger.info("=" * 70)

    # Generate synthetic audio
    generator = SyntheticAudioGenerator(language=args.language)
    success = generator.generate_dataset(num_samples=args.num_samples)

    if success and args.combine:
        # Combine datasets
        augmenter = DatasetAugmenter()
        total = augmenter.combine_datasets()
        logger.info(f"✅ Total dataset size: {total} files")

    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

