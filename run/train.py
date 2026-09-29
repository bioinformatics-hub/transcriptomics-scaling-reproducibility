import signal
import time
from contextlib import contextmanager

import torch

from core.config import ScalingConfig
from core.datamodule import CensusDataModule
from core.models import load_model_from_config
from core.trainer import (
    ScalingTrainer,
    request_graceful_training_stop,
    reset_graceful_training_stop,
)
from lightning.pytorch.callbacks import Callback
from tools.config_files import list_config_files

torch.set_float32_matmul_precision("medium")

MEASURED_COST_STEPS = 25


@contextmanager
def _graceful_training_signals():
    """Turn scheduler termination signals into a clean end-of-batch stop."""

    reset_graceful_training_stop()
    handled = [signal.SIGTERM]
    if hasattr(signal, "SIGUSR1"):
        handled.append(signal.SIGUSR1)
    previous = {signum: signal.getsignal(signum) for signum in handled}

    def request_stop(signum, frame):
        print(
            f"[train] Received signal {signum}; stopping after the current batch.",
            flush=True,
        )
        request_graceful_training_stop()

    for signum in handled:
        signal.signal(signum, request_stop)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        reset_graceful_training_stop()


class CostSplitTimer(Callback):
    def __init__(self):
        super().__init__()
        self.train_time_s = 0.0
        self.val_time_s = 0.0
        self.train_batches = 0
        self.val_batches = 0
        self._batch_start = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._batch_start = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._batch_start is not None:
            self.train_time_s += time.perf_counter() - self._batch_start
            self.train_batches += 1

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        self._batch_start = time.perf_counter()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if self._batch_start is not None:
            self.val_time_s += time.perf_counter() - self._batch_start
            self.val_batches += 1


def train(config: str, fast_trainer: bool = False, resume: bool = False):
    with _graceful_training_signals():
        # Parse path and open config
        config = ScalingConfig(config)
        if resume:
            config.trainer.resume_training = True
            config.trainer.resume_dataloader = True

        # Validate Census-dependent configuration before initializing Lightning/CUDA.
        datamodule = CensusDataModule(config)

        # Scaling trainer set ups logger, checkpointer etc.
        trainer = ScalingTrainer(config, fast_trainer=fast_trainer)

        # Load model
        model = load_model_from_config(config)

        # Train
        trainer.fit(model, datamodule)


def train_cost(config: str):
    # Parse path and open config
    print(f"[Config]: {config}")
    config = ScalingConfig(config)
    original_n_steps = config.trainer.n_steps
    original_n_epochs = config.trainer.n_epochs
    original_val_check_interval = config.trainer.val_check_interval

    config.trainer.n_steps = MEASURED_COST_STEPS  # small train sample
    config.trainer.checkpoint = False  # dont save checkpoints
    config.trainer.val_check_interval = (
        MEASURED_COST_STEPS  # trigger one validation pass after sampled train steps
    )
    config.trainer.limit_val_batches = (
        MEASURED_COST_STEPS  # measure at most this many validation batches
    )
    config.paths.path_to_aimrepo = None  # so it does not log things

    # Validate Census-dependent configuration before initializing Lightning/CUDA.
    datamodule = CensusDataModule(config)

    # Scaling trainer set ups logger, checkpointer etc.
    trainer = ScalingTrainer(config, fast_trainer=True)
    timer = CostSplitTimer()
    trainer.trainer.callbacks.append(timer)

    # Load model
    model = load_model_from_config(config)

    datamodule.setup()

    train_batches_per_epoch = len(datamodule.train_dataloader())
    val_batches_per_epoch = len(datamodule.val_dataloader())

    # Train
    trainer.fit(model, datamodule)

    print(f"[TrainTimer]: {timer.train_time_s}")
    print(f"[ValTimer]: {timer.val_time_s}")
    print(f"[TrainStepsMeasured]: {timer.train_batches}")
    print(f"[ValStepsMeasured]: {timer.val_batches}")
    print(f"[TrainBatchesPerEpoch]: {train_batches_per_epoch}")
    print(f"[ValBatchesPerEpoch]: {val_batches_per_epoch}")
    print(f"[PlannedTrainSteps]: {original_n_steps}")
    print(f"[PlannedEpochs]: {original_n_epochs}")
    print(f"[PlannedValCheckInterval]: {original_val_check_interval}")
    print(f"[MeasuredCostSteps]: {MEASURED_COST_STEPS}")
    # [Memory] logging
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"[Memory]: {peak_vram_mb:.2f} MB")


def train_from_dir(config_dir: str, id: int, cost: bool = False, resume: bool = False):
    # create Path object for robustness
    # Add the config file from the directory based on the id
    config_files = list_config_files(config_dir)
    config_file = config_files[id]
    # switch behaviour
    if cost:
        train_cost(str(config_file))
    else:
        train(str(config_file), resume=resume)
