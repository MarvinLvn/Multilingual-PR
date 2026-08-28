import os
from pickle import FALSE
import random
from dataclasses import dataclass, field
from os import path as osp
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional
from simple_parsing.helpers import Serializable, choice, dict_field, list_field
import multiprocessing
import pytorch_lightning as pl
import simple_parsing
import torch
import torch.optim

################################## Global parameters ##################################


@dataclass
class Hparams:
    """Hyperparameters of for the run"""

    # wandb
    wandb_entity: str = "phorec"
    wandb_project: str = "dev_run" # name of the project/experiment
    debug_pytorch: bool = False  # if activated, allow pytorch debugging features (slower training)

    root_dir: str = os.getcwd()  # root_dir

    # basic params
    seed_everything: Optional[int] = None  # seed for the 5e whole run
    gpu: int = 1  # number or gpu
    max_epochs: int = 40  # maximum number of epochs
    weights_path: str = osp.join(os.getcwd(), "weights")
    precision: int = 32  # 16 for mixed precision, 32 for full precision

    # modes
    tune_lr: bool = False  # tune the model on first run
    dev_run: bool = False

    best_model: str = ""

    log_freq_audio: int = 3             # log audio examples every N epochs
    log_nb_audio: int = 8

    # trainer params
    val_check_interval: float = 1.0     # How often within one training epoch to check the validation set
                                        # (e.g., if set to .25 will validate 4 times during a training epoch)
    limit_train_batches: float = 1.0    # Run through, say 25% of the training set each epoch
    limit_val_batches: float = 1.0      # Run through, say 25% of the validation set each epoch
    enable_progress_bar: bool = True

    # The early stopping callback runs at the end of every validation epoch by default
    # Consequently, it is affected by check_val_every_n_epoch and val_check_interval
    early_stopping: bool = True
    early_stopping_params: Dict[str, Any] = dict_field(
        dict(monitor="val/per", patience=10, mode="min", verbose=True)
    )


@dataclass
class NetworkParams:
    network_name: str = "WavLM"  # Hubert, Wav2Vec2, WavLM
    pretrained_name: Optional[str] = ""

    freeze: bool = True
    freeze_transformer: bool = True

    # Dynamic unfreezing
    conditional_transformer_unfreezing: bool = False  # Enable dynamic unfreezing
    transformer_unfreeze_step: int = 10000  # Step to unfreeze at

    # Phoneme Tokenizer
    eos_token: str = "<blank>"
    bos_token: str = "<blank>"
    unk_token: str = "<blank>"
    pad_token: str = "<blank>"
    word_delimiter_token: str = "<blank>" # blank token

    # Decoder parameters
    decoder_type: str = "greedy"  # greedy, beam_search
    beam_size: int = 5 # the 4 next arguments are use only if decoder_type == beam_search
    language_model_path: Optional[str] = None
    lm_weight: float = 0.5
    word_score: float = 0.0

@dataclass
class DatasetParams:
    """Dataset Parameters
    ! The batch_size and number of crops should be defined here
    """

    # TinyVox Dataset Parameters
    # Configure paths through command-line arguments or environment variables;
    # do not commit machine-specific paths to the repository.
    dataset_path: Optional[str] = field(default_factory=lambda: os.environ.get(
        "TINYVOX_DATASET_PATH"
    ))
    inventory_path: Optional[str] = field(default_factory=lambda: os.environ.get(
        "TINYVOX_INVENTORY_PATH"
    ))
    use_vad: bool = False  # Use audio_with_vad folder instead of audio
    debug_dataset: bool = False # If activated, will only load 1000 training samples
    cache_dir: str = osp.join(os.getcwd(), "assets") # Where dataset files will be stored
    create_dataset: bool = False # Whether to recreate datasets even if they already exists

    # Dataloader parameters
    num_workers: int = multiprocessing.cpu_count() // 2
    batch_size: int = 128

    # Dataset processing parameters
    num_proc: int = 4
    # Window duration in seconds. A value of 0 loads utterance-level files
    # directly from audio/; positive values load windows from original/.
    context_duration: Optional[int] = 0

    def __post_init__(self):
        self.resolve_paths()

    def resolve_paths(self):
        """Normalize configured paths and derive the default inventory path."""
        if self.dataset_path is not None:
            self.dataset_path = str(Path(self.dataset_path).expanduser())

        # Unless explicitly configured, keep the inventory next to the TinyVox
        # metadata. This makes --dataset_path the only path normally required.
        if self.inventory_path is None and self.dataset_path is not None:
            self.inventory_path = str(
                Path(self.dataset_path) / "unique_phonemes.json"
            )
        elif self.inventory_path is not None:
            self.inventory_path = str(Path(self.inventory_path).expanduser())

    def validate_paths(self):
        # Resolve again in case an argument parser updated a nested dataclass
        # after its initial construction.
        self.resolve_paths()

        if self.dataset_path is None:
            raise ValueError(
                "TinyVox dataset path is not configured. Pass --dataset_path "
                "or set TINYVOX_DATASET_PATH."
            )

        dataset_path = Path(self.dataset_path)
        if not dataset_path.is_dir():
            raise FileNotFoundError(
                f"TinyVox dataset not found: {dataset_path}. "
                "Pass --dataset_path or set TINYVOX_DATASET_PATH."
            )

        if self.inventory_path is None:
            raise ValueError(
                "Phoneme inventory path is not configured. Pass "
                "--inventory_path or set TINYVOX_INVENTORY_PATH."
            )

        inventory_path = Path(self.inventory_path)
        if not inventory_path.is_file():
            raise FileNotFoundError(
                f"Phoneme inventory not found: {inventory_path}. Run "
                f"'python utils/create_phoneme_inventory.py {dataset_path}'."
            )


@dataclass
class OptimizerParams:
    """Optimization parameters"""

    optimizer: str = "AdamW"
    lr: float = 1e-4
    weight_decay: float = 1e-2
    accumulate_grad_batches: int = 8

    # Scheduler parameters (all step-based except ReduceLROnPlateau)
    scheduler: Optional[str] = None # TriStage, Cosine, StepLR, MultiStepLR, ReduceLROnPlateau

    # Cosine scheduler (step-based)
    # Phase1: linear warmup from <warm_start_lr> to <lr> over <warmup_steps>
    # Phase2: cosine decay from base <lr> to <eta_min> over remaining epochs
    #    /-------\
    #   /         \
    #  /           \____
    # /                 \___
    max_steps: int = 260000
    warmup_steps: int = 10000
    warmup_start_lr: float = 0.0
    eta_min: float = 0.0

    # StepLR scheduler (step-based)
    # Multiplies <lr> by <gamma> every <step_size_steps>
    # __
    #   __
    #     __
    #        __
    step_size_steps: int = 50000
    gamma: float = 0.1

    # MultiStepLR scheduler (step-based)
    # Reduces <lr> by <gamma> at <milestone_steps>
    # ____
    #     ____
    #         ____
    milestones_steps: List[Any] = list_field(50000, 100000, 150000)

    # ReduceLROnPlateau scheduler (epoch-based)
    min_lr: float = 5e-9
    patience: int = 10 # in number of epochs

    # Tri-stage scheduler parameters
    #      / -------- \
    #     /            \
    #    /              \
    #   /                \____
    total_training_steps: int = 100000
    tri_stage_warmup_ratio: float = 0.1  # 10% warmup
    tri_stage_constant_ratio: float = 0.4 # 40% constant lr
    # Decay for the remaining steps (calculated automatically)
    # /!\ Careful to plan max_epochs accordingly (otherwise you'll be training with lr = 0)

@dataclass
class Parameters:
    """base options."""

    hparams: Hparams = Hparams()
    data_param: DatasetParams = DatasetParams()
    network_param: NetworkParams = NetworkParams()
    optim_param: OptimizerParams = OptimizerParams()

    def __post_init__(self):
        """Post-initialization code"""
        if self.hparams.seed_everything is None:
            self.hparams.seed_everything = random.randint(1, 10000)

        if self.hparams.precision == 16:
            self.hparams.precision = '16-mixed'
        random.seed(self.hparams.seed_everything)
        torch.manual_seed(self.hparams.seed_everything)
        pl.seed_everything(self.hparams.seed_everything)

        if self.network_param.pretrained_name == "":
            if self.network_param.network_name == "Wav2Vec2":
                self.network_param.pretrained_name = "facebook/wav2vec2-base-960h"
            elif self.network_param.network_name == "WavLM":
                self.network_param.pretrained_name = "microsoft/wavlm-base"
            elif self.network_param.network_name == "Hubert":
                self.network_param.pretrained_name = "facebook/hubert-base-ls960"
            elif self.network_param.network_name == "WavLMplus":
                self.network_param.network_name = "WavLM"
                self.network_param.pretrained_name = "microsoft/wavlm-base-plus"
            elif self.network_param.network_name == "Wav2Vec2XLSR":
                self.network_param.network_name = "Wav2Vec2"
                self.network_param.pretrained_name = "facebook/wav2vec2-large-xlsr-53"
            elif self.network_param.network_name == "BabyHubert":
                self.network_param.pretrained_name = "weights/babyhubert_pretrained"
            elif self.network_param.network_name == "W2VLB":
                self.network_param.pretrained_name = "weights/w2vlb_pretrained"
            else:
                raise NotImplementedError(
                    "Only Wav2Vec2, WavLM, Hubert, WavLMplus, Wav2Vec2XLSR, BabyHubert, and W2VLB are available."
                )
        print(f"Pretrained model: {self.network_param.pretrained_name}")

        self.data_param.wandb_project = self.hparams.wandb_project
        self.hparams.accumulate_grad_batches = self.optim_param.accumulate_grad_batches

    @classmethod
    def parse(cls):
        parser = simple_parsing.ArgumentParser()
        parser.add_arguments(cls, dest="parameters")
        args = parser.parse_args()
        instance: Parameters = args.parameters
        return instance
