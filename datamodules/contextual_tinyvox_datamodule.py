import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from utils.constant import CHARS_TO_REMOVE_REGEX
from utils.logger import init_logger

from datasets import Dataset


class ContextualTinyVoxDataModule(LightningDataModule):
    def __init__(self, dataset_param):
        super().__init__()
        self.config = dataset_param
        self.logger = init_logger('ContextualTinyVoxDataModule', 'INFO')

        # Context parameters
        self.context_duration = dataset_param.context_duration
        if self.context_duration is None or self.context_duration < 0:
            raise ValueError("context_duration must be a non-negative number")
        self.context_duration_ms = self.context_duration * 1000

        self.sampling_rate = 16000
        self.n_debug = 100
        self.processor = None
        self.config.dataset_path = Path(self.config.dataset_path)
        self.dataset_name = self.config.dataset_path.stem.lower()

        self.logger.info(f'Loading Contextual Dataset from: {self.config.dataset_path}')
        self.logger.info(f'Context duration: {self.context_duration}s')
        self.logger.info(f'Using VAD timing: {self.config.use_vad}')

    def _load_split(self, split):
        """Load and create contextual metadata (no caching)"""
        # Load CSV metadata
        csv_path = self.config.dataset_path / f'{split}.csv'
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if self.config.debug_dataset:
            df = df.iloc[:min(len(df), self.n_debug)]

        # Original recordings are not needed in no-context mode. Apart from
        # saving disk space, delaying this conversion keeps audio-only datasets
        # independent of the naming convention used by original recordings.
        if self.context_duration == 0:
            audio_dir = self.config.dataset_path / 'audio'
            if not audio_dir.is_dir():
                raise FileNotFoundError(f"TinyVox audio directory not found: {audio_dir}")
        else:
            df['original_filename'] = df['audio_filename'].map(
                lambda filename: '_'.join(filename.split('_')[:-2]) + '.wav'
            )

        self.logger.info(f"Loaded {len(df)} utterances for {split} split")

        # Remove word boundaries (ML: this should be removed from tinyvox altogether)
        df['phones'] = df['phones'].str.replace('|', '').str.replace(r'\s+', ' ', regex=True).str.strip()

        # Filter out rows with missing data
        na_phones = df['phones'].isna()
        self.logger.info(f"Removed {na_phones.sum()} samples with NA phones.")
        df = df[~na_phones]

        # Create contextual samples metadata
        contextual_samples = self._create_contextual_metadata(df)

        self.logger.info(f"Created {len(contextual_samples)} contextual samples from CSV")

        # Create dataset from metadata only
        dataset = Dataset.from_list(contextual_samples)
        return dataset

    def _create_contextual_metadata(self, df):
        """Create training sample metadata.

        With context_duration == 0, use the already-segmented TinyVox WAV files
        from audio/ directly.

        With context_duration > 0, preserve the original BabAR contextual
        behaviour based on original/ recordings.
        """
        samples = []

        # No-context mode: TinyVox utterance WAVs are already segmented.
        if self.context_duration == 0:
            for _, row in df.iterrows():
                audio_path = self.config.dataset_path / 'audio' / row['audio_filename']

                phonemes = row['phones'].strip() if pd.notna(row['phones']) else ""
                sentence = row['sentence'] if pd.notna(row['sentence']) else ""
                cleaned_sentence = re.sub(
                    CHARS_TO_REMOVE_REGEX, '', sentence
                ).lower().strip()

                samples.append({
                    'audio_path': str(audio_path),
                    'target_phonemes': phonemes,
                    'target_sentence': cleaned_sentence,
                    'audio_filename': row['audio_filename'],
                })

            return samples

        # Contextual mode: preserve the original implementation.
        grouped = df.groupby('original_filename')

        for original_filename, group in grouped:
            original_audio_path = (
                self.config.dataset_path
                / 'original'
                / original_filename
            )

            group = group.sort_values('onset')

            for _, row in group.iterrows():
                sample = self._create_context_metadata_for_utterance(
                    row,
                    str(original_audio_path)
                )

                if sample:
                    samples.append(sample)

        return samples

    def _create_context_metadata_for_utterance(self, target_row, original_audio_path):
        """Create metadata for a contextual sample centered around a target utterance"""

        if self.config.use_vad and pd.notna(target_row['with_vad_onset']):
            target_onset = target_row['with_vad_onset']
            target_offset = target_row['with_vad_offset']
        else:
            target_onset = target_row['onset']
            target_offset = target_row['offset']

        if pd.isna(target_onset) or pd.isna(target_offset):
            return None

        # Calculate desired context window (centered on target utterance)
        target_center = (target_onset + target_offset) / 2
        desired_start = target_center - self.context_duration_ms / 2
        desired_end = target_center + self.context_duration_ms / 2

        # Ensure the context always includes the full target utterance
        # This may expand the context beyond the requested duration for long utterances
        context_start = max(0, min(desired_start, target_onset))
        context_end = max(desired_end, target_offset)

        # Calculate actual duration needed (may be > requested duration)
        context_duration_ms = context_end - context_start

        # Calculate target position within the (possibly expanded) context
        target_start_in_context = target_onset - context_start
        target_end_in_context = target_offset - context_start

        # Precompute frame boundaries (for CTC loss optimization)
        estimated_frame_rate = 50.0  # frames per second
        target_start_frame = round(target_start_in_context * estimated_frame_rate / 1000.0)
        target_end_frame = round(target_end_in_context * estimated_frame_rate / 1000.0)
        target_start_frame = max(0, target_start_frame)
        target_end_frame = max(target_start_frame + 1, target_end_frame)

        # Clean up the phoneme and sentence strings
        phonemes = target_row['phones'].strip() if pd.notna(target_row['phones']) else ""
        sentence = target_row['sentence'] if pd.notna(target_row['sentence']) else ""
        cleaned_sentence = re.sub(CHARS_TO_REMOVE_REGEX, '', sentence).lower().strip()

        return {
            'original_audio_path': original_audio_path,
            'target_phonemes': phonemes,
            'target_sentence': cleaned_sentence,  # Pre-cleaned
            # these fields indicate where in the whole audio, one can extract the context window
            'context_start_ms': float(context_start),
            'context_duration_ms': float(context_duration_ms),
            # these fields indicate where the target utterance start and end within the context window
            'target_start_ms': float(target_start_in_context),
            'target_end_ms': float(target_end_in_context),
            'target_start_frame': target_start_frame,
            'target_end_frame': target_end_frame,
            'audio_filename': target_row['audio_filename'],
        }

    def set_processor(self, processor):
        self.processor = processor

    def setup(self, stage):
        """Load and setup datasets"""
        if self.processor is None:
            raise ValueError("Processor must be set before calling setup().")

        if stage == 'fit':
            self.train_dataset = self._load_split('train')
            self.val_dataset = self._load_split('val')
        elif stage == 'test':
            self.test_dataset = self._load_split('test')
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def _load_audio_segment(self, audio_path, offset_ms, duration_ms):
        """Load audio segment using soundfile"""
        offset_samples = int(offset_ms * self.sampling_rate / 1000.0)
        duration_samples = int(duration_ms * self.sampling_rate / 1000.0)

        # Load audio segment with soundfile
        audio, sr = sf.read(
            audio_path,
            start=offset_samples,
            stop=offset_samples + duration_samples,
            dtype='float32'
        )

        # Handle mono conversion if needed
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Convert to mono

        # Verify sample rate (should already be 16kHz but check)
        if sr != self.sampling_rate:
            raise ValueError(f"Sample rate mismatch in {audio_path}: expected {self.sampling_rate}, got {sr}")

        return audio

    def _load_audio_file(self, audio_path):
        """Load an entire already-segmented TinyVox WAV file."""
        audio, sr = sf.read(audio_path, dtype='float32')

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != self.sampling_rate:
            raise ValueError(
                f"Sample rate mismatch in {audio_path}: "
                f"expected {self.sampling_rate}, got {sr}"
            )

        return audio

    def collate_fn(self, batch):
        """Load audio on-demand and create batch."""

        context_audios = []
        valid_samples = []

        # TinyVox audio/*.wav already contains the target utterance.
        if self.context_duration == 0:
            for sample in batch:
                audio = self._load_audio_file(sample['audio_path'])

                context_audios.append(audio)
                valid_samples.append(sample)

            if not context_audios:
                raise ValueError("No valid audio samples in batch")

            # Dynamic padding is performed by the processor.
            processed = self.processor(
                context_audios,
                sampling_rate=self.sampling_rate,
                padding=True,
                return_tensors="pt",
            )

            # In no-context mode, the whole WAV corresponds to the target.
            target_frame_starts = [0] * len(context_audios)

            target_frame_ends = [
                max(
                    1,
                    round(len(audio) * 50.0 / self.sampling_rate)
                )
                for audio in context_audios
            ]

            target_start_ms = [0.0] * len(context_audios)

            target_end_ms = [
                len(audio) * 1000.0 / self.sampling_rate
                for audio in context_audios
            ]

            cleaned_sentences = [
                sample.get('target_sentence', '')
                for sample in valid_samples
            ]

            return {
                "array": processed["input_values"],
                "path": [
                    sample["audio_path"]
                    for sample in valid_samples
                ],
                "phonemes": [
                    sample["target_phonemes"]
                    for sample in valid_samples
                ],
                "sentence": cleaned_sentences,
                "target_frame_start": target_frame_starts,
                "target_frame_end": target_frame_ends,
                "target_start_ms": target_start_ms,
                "target_end_ms": target_end_ms,
                "audio_filename": [
                    sample["audio_filename"]
                    for sample in valid_samples
                ],
            }

        max_duration_ms = max(
            sample['context_duration_ms']
            for sample in batch
        )

        expected_length = int(
            self.sampling_rate * max_duration_ms / 1000.0
        )

        for sample in batch:
            audio = self._load_audio_segment(
                sample['original_audio_path'],
                sample['context_start_ms'],
                sample['context_duration_ms']
            )

            if len(audio) < expected_length:
                audio = np.pad(
                    audio,
                    (0, expected_length - len(audio)),
                    mode='constant',
                    constant_values=0.0
                )

            context_audios.append(audio)
            valid_samples.append(sample)

        if not context_audios:
            raise ValueError("No valid audio samples in batch")

        processed = self.processor(
            context_audios,
            sampling_rate=self.sampling_rate,
            padding=True,
            return_tensors="pt",
        )

        target_frame_starts = [
            sample["target_start_frame"]
            for sample in valid_samples
        ]

        target_frame_ends = [
            sample["target_end_frame"]
            for sample in valid_samples
        ]

        cleaned_sentences = [
            sample.get('target_sentence', '')
            for sample in valid_samples
        ]

        return {
            "array": processed["input_values"],
            "path": [
                sample["original_audio_path"]
                for sample in valid_samples
            ],
            "phonemes": [
                sample["target_phonemes"]
                for sample in valid_samples
            ],
            "sentence": cleaned_sentences,
            "target_frame_start": target_frame_starts,
            "target_frame_end": target_frame_ends,
            "target_start_ms": [
                sample["target_start_ms"]
                for sample in valid_samples
            ],
            "target_end_ms": [
                sample["target_end_ms"]
                for sample in valid_samples
            ],
            "audio_filename": [
                sample["audio_filename"]
                for sample in valid_samples
            ],
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
