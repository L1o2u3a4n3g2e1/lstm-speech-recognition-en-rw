import pyaudio
import wave
import numpy as np
import os
import json
import csv
from datetime import datetime
import threading
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioRecorder:
    def __init__(self, output_dir, language='english', sr=16000, chunk=1024, format=pyaudio.paInt16, channels=1):
        self.output_dir = output_dir
        self.language = language.lower()
        self.sr = sr
        self.chunk = chunk
        self.format = format
        self.channels = channels
        self.p = pyaudio.PyAudio()
        self.is_recording = False
        self.frames = []

        os.makedirs(output_dir, exist_ok=True)

    def list_devices(self):
        """List available audio input devices."""
        logger.info("Available Audio Devices:")
        info = self.p.get_host_api_info_by_index(0)
        for i in range(0, info.get('deviceCount')):
            device_info = self.p.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxInputChannels') > 0:
                logger.info(f"Device {i}: {device_info.get('name')} "
                          f"(Channels: {device_info.get('maxInputChannels')})")

    def record_audio(self, duration=5, device_index=None):
        """Record audio for specified duration."""
        try:
            stream = self.p.open(format=self.format, channels=self.channels, rate=self.sr,
                               input=True, input_device_index=device_index, frames_per_buffer=self.chunk)
            self.frames = []
            self.is_recording = True

            logger.info(f"Recording for {duration} seconds... Speak now!")
            for _ in range(0, int(self.sr / self.chunk * duration)):
                if not self.is_recording:
                    break
                data = stream.read(self.chunk)
                self.frames.append(data)

            stream.stop_stream()
            stream.close()
            logger.info("Recording complete!")
            return True

        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            return False

    def stop_recording(self):
        """Stop recording prematurely."""
        self.is_recording = False

    def save_audio(self, filename, transcript, speaker_id="user"):
        """Save recorded audio to WAV file."""
        if not self.frames:
            logger.warning("No audio frames to save!")
            return None

        filepath = os.path.join(self.output_dir, filename)
        try:
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.p.get_sample_size(self.format))
                wf.setframerate(self.sr)
                wf.writeframes(b''.join(self.frames))

            logger.info(f"Saved: {filepath}")

            metadata = {
                'filename': filename,
                'transcript': transcript,
                'language': self.language,
                'speaker_id': speaker_id,
                'duration': len(self.frames) * self.chunk / self.sr,
                'sample_rate': self.sr,
                'timestamp': datetime.now().isoformat(),
                'channels': self.channels
            }

            return filepath, metadata

        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            return None

    def save_metadata(self, metadata_list, csv_filename='transcripts.csv'):
        """Save metadata to CSV file."""
        csv_path = os.path.join(self.output_dir, csv_filename)
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['filename', 'transcript', 'language', 'speaker_id', 'duration', 'timestamp']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for meta in metadata_list:
                    writer.writerow({
                        'filename': meta['filename'],
                        'transcript': meta['transcript'],
                        'language': meta['language'],
                        'speaker_id': meta['speaker_id'],
                        'duration': f"{meta['duration']:.2f}",
                        'timestamp': meta['timestamp']
                    })

            logger.info(f"Metadata saved to {csv_path}")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")

    def cleanup(self):
        """Close PyAudio instance."""
        self.p.terminate()


def interactive_recording_session(language='english', duration=5, num_samples=10):
    """Interactive recording session with prompts."""

    output_dir = f"data/{language}/raw/custom_recordings"
    recorder = AudioRecorder(output_dir, language=language)

    logger.info(f"Starting {language.upper()} recording session")
    logger.info(f"Recording {num_samples} samples, {duration} seconds each")

    # Sample sentences for recording (from EN_RW_DICTIONARY)
    if language.lower() == 'kinyarwanda':
        sentences = [
            # Greetings (5)
            "Muraho neza.",
            "Mwaramutse neza.",
            "Habari yako?",
            "Nzi neza.",
            "Mwakunze.",

            # Books & Reading (8)
            "Ndagukunda ibitabo.",
            "Nshaka gusoma.",
            "Soma igitabo.",
            "Fungura isomero.",
            "Reka ntubuke inkuru.",
            "Iki gitabo nkabyo.",
            "Igitabo.",
            "Soma.",

            # Categories (8)
            "Categori.",
            "Ubuhinzi.",
            "Ubuvuzi.",
            "Amateka.",
            "Amasomo.",
            "Igihembe.",
            "Umuziki.",
            "Siporo.",

            # Time & Duration (5)
            "Ni ikihe kigero?",
            "Umezute gake?",
            "Komeza.",
            "Subira inyuma.",
            "Nyuma.",

            # Actions & Commands (10)
            "Nsomera.",
            "Hindura mukinyarwanda.",
            "Hindura mucyongereza.",
            "Mpindurira.",
            "Komeza gusoma.",
            "Gushaka.",
            "Tegeka.",
            "Humeka.",
            "Kubika.",
            "Kuroga.",

            # Confirmation & Questions (8)
            "Yego.",
            "Oya.",
            "Iki?",
            "Hehe?",
            "Ryari?",
            "Gute?",
            "Waba afite?",
            "Yego, nimpeta.",

            # Emotions & Reactions (6)
            "Ndakwiyubaka.",
            "Nkabyo.",
            "Bibi.",
            "Nziza.",
            "Nziza cyane.",
            "Ayaba.",

            # Digital Library (7)
            "Ikirango cy'igitabo.",
            "Ikirango.",
            "Isomero.",
            "Gushaka igitabo.",
            "Ibikumbuzo.",
            "Bisoma vuba.",
            "Bisoma cyane."
        ]
    else:  # English
        sentences = [
            "Hello, how are you today?",
            "What time is it now?",
            "I love reading books.",
            "Let me not forget the story.",
            "I want to read later.",
            "This book is amazing.",
            "The path is beautiful.",
            "Knowledge is power.",
            "Have no worries, be free.",
            "We are happy to share."
        ]

    metadata_list = []
    speaker_id = input("Enter speaker ID (or press Enter for 'user'): ").strip() or "user"

    try:
        for i in range(num_samples):
            sentence = sentences[i % len(sentences)]
            print(f"\n[{i+1}/{num_samples}] Read this sentence:")
            print(f">>> {sentence}")
            input("Press ENTER when ready to record...")

            if recorder.record_audio(duration=duration):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{language}_{speaker_id}_{timestamp}_{i+1}.wav"

                result = recorder.save_audio(filename, sentence, speaker_id)
                if result:
                    filepath, metadata = result
                    metadata_list.append(metadata)

            response = input("Continue? (y/n): ").strip().lower()
            if response != 'y':
                break

        recorder.save_metadata(metadata_list)
        logger.info(f"Session complete! Recorded {len(metadata_list)} samples.")

    finally:
        recorder.cleanup()


if __name__ == "__main__":
    print("=" * 60)
    print("LSTM Speech Dataset Recorder")
    print("=" * 60)

    language = input("Language (english/kinyarwanda): ").strip().lower()
    if language not in ['english', 'kinyarwanda']:
        language = 'english'

    duration = int(input("Duration per sample (seconds, default 5): ").strip() or "5")
    num_samples = int(input("Number of samples to record (default 10): ").strip() or "10")

    interactive_recording_session(language=language, duration=duration, num_samples=num_samples)
