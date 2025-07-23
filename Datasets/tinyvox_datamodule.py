import pickle
import re
import shutil
import os
import numpy as np
import utils.agent_utils as ag_u
import wandb
from datasets import Dataset, Audio
from librosa.effects import trim
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from utils.constant import CHARS_TO_REMOVE_REGEX
from utils.dataset_utils import coll_fn
from utils.logger import init_logger
import pandas as pd
from pathlib import Path

class TinyVoxDataModule(LightningDataModule):
    def __init__(self, dataset_param):
        super().__init__()
        self.config = dataset_param
        self.logger = init_logger('TinyVoxDataModule', 'INFO')
        self.audio_folder = 'audio_with_vad' if self.config.use_vad else 'audio'
        self.sampling_rate = 16000
        self.n_debug = 100
        self.config.dataset_path = Path(self.config.dataset_path)

        self.logger.info(f'Loading Dataset : {self.config.dataset_path / self.audio_folder}')


    def setup(self, stage=None):
        """ Load and setup datasets """
        if stage in (None, 'fit'):
            self.train_dataset = self._load_split('train')
            self.val_dataset = self._load_split('val')
            self._process_split('train')
            self._process_split('val')

        if stage == 'test':
            self.test_dataset = self._load_split('test')
            self._process_split('test')

    def _load_split(self, split):
        """Load a dataset split from CSV and audio files"""
        save_path = Path('assets') / 'datasets' / f'{split}_tinyvox'
        processed_file = save_path / f'tinyvox_{split}_processed.pkl'

        # A. Load pickle file if it exists (if create_dataset = False)
        if processed_file.is_file() and not self.config.create_dataset:
            self.logger.info(f"Loading cached {split} dataset from {processed_file}")
            with open(processed_file, 'rb') as f:
                return pickle.load(f)

        save_path.mkdir(exist_ok=True, parents=True)

        csv_path = self.config.dataset_path / f'{split}.csv'
        audio_dir = self.config.dataset_path / self.audio_folder

        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        if not audio_dir.is_dir():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

        # B.1 Load CSV
        df = pd.read_csv(csv_path)
        if self.config.debug_training:
            df = df.iloc[:min(len(df), self.n_debug)]


        self.logger.info(f"Loaded {len(df)} samples for {split} split")
        na_phones = df['phones'].isna()
        self.logger.info(f"Removed {na_phones.sum()} samples with NA phones.")
        df = df[~na_phones]

        # B.2 Verify required columns
        required_cols = ['audio_filename', 'phones']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")

        # B.3 Create full audio paths
        df['full_audio_path'] = df['audio_filename'].apply(lambda x: str(audio_dir / x))

        # B.4 Verify audio files exist
        missing_files = []
        for idx, row in df.iterrows():
            if not os.path.exists(row['full_audio_path']):
                missing_files.append(row['full_audio_path'])

        if missing_files:
            self.logger.warning(f"Found {len(missing_files)} missing audio files")
            # Remove missing files
            df = df[~df['full_audio_path'].isin(missing_files)]
            self.logger.info(f"After filtering: {len(df)} samples remaining")

        # B.5 Convert to HuggingFace Dataset format
        dataset_dict = {
            'audio': df['full_audio_path'].tolist(),
            'path': df['full_audio_path'].tolist(),
            'phonemes': df['phones'].tolist(),
        }

        dataset = Dataset.from_dict(dataset_dict)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
        dataset = dataset.map( self._extract_audio_array, num_proc=self.config.num_proc, load_from_cache_file=False)

        # C. Save dataset
        self.logger.info(f"Saving processed {split} dataset to {processed_file}")
        with open(processed_file, 'wb') as f:
            pickle.dump(dataset, f)

        return dataset

    def _extract_audio_array(self, sample):
        """Extract audio array from HuggingFace Audio format"""
        return {
            'audio': sample['audio']['array'],  # Extract just the numpy array
            'path': sample['path'],
            'phonemes': sample['phonemes'],
        }

    def _process_split(self, split):
        """Process dataset split - format phonemes"""
        dataset = getattr(self, f"{split}_dataset")

        # Format phonemes to match expected format
        dataset = dataset.map(
            self._format_phonemes,
            num_proc=self.config.num_proc,
            load_from_cache_file=False,
        )

        setattr(self, f"{split}_dataset", dataset)
        self.logger.info(f"Processed {split} dataset: {len(dataset)} samples")

    def _format_phonemes(self, data_sample):
        """Add space between individual phonemes and | for end of word token"""
        """ meʒərɪŋ kʌp -->  m e ʒ ə r ɪ ŋ | k ʌ p |"""
        raw_phones = data_sample['phonemes'].strip()

        # Convert "meʒərɪŋ kʌp" → "m e ʒ ə r ɪ ŋ | k ʌ p |"
        formatted_phones = ' | '.join(
            ' '.join(word) for word in raw_phones.split()
        ) + ' |'

        return {'phonemes': formatted_phones}

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=coll_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=coll_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=coll_fn,
        )
