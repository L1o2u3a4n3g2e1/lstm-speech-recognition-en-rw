import os
import sys
import numpy as np
import librosa
import soundfile as sf
import logging
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from audio_processor import AudioProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def augment_and_save_dataset(language='kinyarwanda', num_augmentations=3):
    """
    Augment audio dataset using pitch shift and time stretch.

    Args:
        language: 'english' or 'kinyarwanda'
        num_augmentations: Number of augmented versions per original sample
    """
    raw_dir = f'data/{language}/raw/custom_recordings'
    augmented_dir = f'data/{language}/raw/augmented'

    os.makedirs(augmented_dir, exist_ok=True)

    # Load transcripts
    import csv
    transcripts = {}
    csv_path = os.path.join(raw_dir, 'transcripts.csv')
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            transcripts[row['filename']] = row['transcript']

    # Initialize processor
    processor = AudioProcessor(sr=16000, duration=5, n_mfcc=13)

    # Get all WAV files
    wav_files = sorted(Path(raw_dir).glob('*.wav'))
    logger.info(f"Found {len(wav_files)} original audio files")

    augmented_count = 0
    augmented_records = []

    for original_file in wav_files:
        filename = original_file.name
        transcript = transcripts.get(filename, 'UNKNOWN')

        logger.info(f"Augmenting: {filename}")

        # Load original audio
        audio, sr = librosa.load(str(original_file), sr=16000)

        # Save original with marker
        original_augmented_path = os.path.join(augmented_dir, f"aug0_{filename}")
        sf.write(original_augmented_path, audio, sr)
        augmented_records.append({
            'filename': f"aug0_{filename}",
            'transcript': transcript,
            'language': language,
            'speaker_id': 'user',
            'duration': len(audio) / sr,
            'timestamp': original_file.stat().st_mtime
        })
        augmented_count += 1

        # Generate augmentations
        for i in range(1, num_augmentations + 1):
            # Random pitch shift (-2 to +2 semitones)
            pitch_shift = np.random.randint(-2, 3)
            if pitch_shift != 0:
                audio_aug = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)
            else:
                audio_aug = audio.copy()

            # Random time stretch (0.9x to 1.1x)
            time_stretch = np.random.uniform(0.9, 1.1)
            audio_aug = librosa.effects.time_stretch(audio_aug, rate=time_stretch)

            # Pad/trim to original length
            audio_aug = processor.pad_or_trim(audio_aug)

            # Save augmented file
            aug_filename = f"aug{i}_{filename}"
            aug_path = os.path.join(augmented_dir, aug_filename)
            sf.write(aug_path, audio_aug, sr)

            augmented_records.append({
                'filename': aug_filename,
                'transcript': transcript,
                'language': language,
                'speaker_id': 'user',
                'duration': len(audio_aug) / sr,
                'timestamp': original_file.stat().st_mtime
            })
            augmented_count += 1

            logger.info(f"  Generated: aug{i} (pitch:{pitch_shift:+d}, time:{time_stretch:.2f}x)")

    # Save augmented transcripts
    aug_csv_path = os.path.join(augmented_dir, 'transcripts_augmented.csv')
    with open(aug_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'transcript', 'language', 'speaker_id', 'duration', 'timestamp'])
        writer.writeheader()
        writer.writerows(augmented_records)

    logger.info(f"\n{'='*60}")
    logger.info(f"Augmentation Complete!")
    logger.info(f"Original samples: {len(wav_files)}")
    logger.info(f"Total augmented samples: {augmented_count}")
    logger.info(f"Expansion ratio: {augmented_count / len(wav_files):.1f}x")
    logger.info(f"Saved to: {augmented_dir}")
    logger.info(f"Transcripts saved to: {aug_csv_path}")
    logger.info(f"{'='*60}\n")

    return augmented_count


if __name__ == "__main__":
    language = input("Language (english/kinyarwanda) [kinyarwanda]: ").strip().lower() or 'kinyarwanda'
    num_aug = int(input("Number of augmentations per sample [3]: ").strip() or "3")

    total = augment_and_save_dataset(language=language, num_augmentations=num_aug)
    print(f"✓ Generated {total} augmented samples")
