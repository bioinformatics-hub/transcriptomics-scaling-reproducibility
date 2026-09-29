import os
import shutil
import threading
import time
import warnings

import lightning as pl
import torch
from aim.pytorch_lightning import AimLogger
from lightning.pytorch.callbacks import (
    Callback,
    ModelCheckpoint,
    Timer,
    TQDMProgressBar,
)

from core.config import ScalingConfig
from tools.utils import human_format, format_seconds_dhms


_GRACEFUL_STOP_REQUESTED = threading.Event()


def request_graceful_training_stop() -> None:
    _GRACEFUL_STOP_REQUESTED.set()


def reset_graceful_training_stop() -> None:
    _GRACEFUL_STOP_REQUESTED.clear()


def _training_data_cursor(trainer):
    datamodule = getattr(trainer, "datamodule", None)
    train_dataset = getattr(datamodule, "train_dataset", None)
    return getattr(train_dataset, "data_cursor", None)


class SegmentTimer(Timer):
    """A per-process training timer additionally bounded by a shared job deadline.

    Lightning's standard Timer restores its elapsed time from a checkpoint. That is
    useful for a lifetime budget, but it makes a resumed Slurm segment immediately
    consume the previous segment's time. This timer deliberately starts from zero on
    every process and also respects ``SCALING_JOB_DEADLINE_UNIX`` when a packed job
    provides one.
    """

    def __init__(self, duration: str | None):
        super().__init__(duration=duration, interval="step")
        raw_deadline = os.environ.get("SCALING_JOB_DEADLINE_UNIX")
        self._job_deadline_unix = float(raw_deadline) if raw_deadline else None
        self._stop_logged = False

    def state_dict(self) -> dict:
        # Segment elapsed time must never carry into the next resumed process.
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        self._offset = 0

    def _deadline_reached(self) -> bool:
        return (
            self._job_deadline_unix is not None
            and time.time() >= self._job_deadline_unix
        )

    def _check_time_remaining(self, trainer) -> None:
        duration_reached = (
            self._duration is not None and self.time_elapsed() >= self._duration
        )
        deadline_reached = self._deadline_reached()
        signal_received = _GRACEFUL_STOP_REQUESTED.is_set()
        should_stop = duration_reached or deadline_reached or signal_received
        should_stop = trainer.strategy.broadcast(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop and not self._stop_logged:
            reasons = []
            if duration_reached:
                reasons.append("segment max_time")
            if deadline_reached:
                reasons.append("packed-job deadline")
            if signal_received:
                reasons.append("termination signal")
            print(
                "[SegmentTimer] Graceful stop requested by " + ", ".join(reasons),
                flush=True,
            )
            self._stop_logged = True

    def on_fit_start(self, trainer, *args, **kwargs) -> None:
        # Always allow one optimizer step so even a process that reaches training
        # after its deadline can produce a valid model/cursor checkpoint pair.
        return None

    def on_train_batch_end(self, trainer, *args, **kwargs) -> None:
        self._check_time_remaining(trainer)


class TrainingDataCursorCallback(Callback):
    """Commit batches after optimization and serialize the cursor in checkpoints."""

    CHECKPOINT_KEY = "training_data_cursor"

    def __init__(self):
        super().__init__()
        self._last_global_step = 0

    def on_train_start(self, trainer, pl_module) -> None:
        self._last_global_step = int(trainer.global_step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        cursor = _training_data_cursor(trainer)
        global_step = int(trainer.global_step)
        if cursor is not None and global_step > self._last_global_step:
            # Commit the whole accumulation window only after its optimizer step.
            # Checkpoints do not preserve partially accumulated gradients.
            cursor.commit_pending_batches()
        self._last_global_step = global_step

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        cursor = _training_data_cursor(trainer)
        if cursor is not None:
            checkpoint[self.CHECKPOINT_KEY] = cursor.state_dict()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        cursor = _training_data_cursor(trainer)
        state = checkpoint.get(self.CHECKPOINT_KEY)
        if cursor is not None and state is not None:
            cursor.load_state_dict(state)


class ResumableModelCheckpoint(ModelCheckpoint):
    """Write the lightweight resume marker only after ``last.ckpt`` is durable."""

    def _save_last_checkpoint(self, trainer, monitor_candidates) -> None:
        super()._save_last_checkpoint(trainer, monitor_candidates)
        cursor = _training_data_cursor(trainer)
        if cursor is not None and self.last_model_path:
            cursor.save_resume_marker(
                checkpoint_path=self.last_model_path,
                global_step=trainer.global_step,
            )


class StepTimeLogger(Callback):
    """
    Logs the runtime of each training, validation, and test step as 'train_step_time_s', 'val_step_time_s', 'test_step_time_s'.
    """

    def __init__(self):
        super().__init__()
        self._train_wall_start = None
        self._train_step_total = 0.0
        self._train_interstep_total = 0.0
        self._val_wall_total = 0.0
        self._val_step_total = 0.0
        self._checkpoint_total = 0.0

    def on_train_start(self, trainer, pl_module):
        self._train_wall_start = time.time()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = time.time()
        # Inter-step time: time since last batch end (i.e., dataloader wait + any idle)
        last_end = getattr(pl_module, "_step_end_time", None)
        if last_end is not None:
            interstep = now - last_end
            self._train_interstep_total += interstep
            pl_module.log(
                "train_interstep_time_s",
                interstep,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )
        pl_module._step_start_time = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        now = time.time()
        elapsed = now - getattr(pl_module, "_step_start_time", now)
        self._train_step_total += elapsed
        pl_module.log(
            "train_step_time_s",
            elapsed,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        pl_module._step_end_time = now

    def on_validation_start(self, trainer, pl_module):
        self._val_wall_start = time.time()

    def on_validation_end(self, trainer, pl_module):
        val_wall = time.time() - getattr(self, "_val_wall_start", time.time())
        self._val_wall_total += val_wall

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        pl_module._step_start_time = time.time()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        elapsed = time.time() - getattr(pl_module, "_step_start_time", time.time())
        self._val_step_total += elapsed
        pl_module.log(
            "val_step_time_s",
            elapsed,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )

    def on_test_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        pl_module._step_start_time = time.time()

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        elapsed = time.time() - getattr(pl_module, "_step_start_time", time.time())
        pl_module.log(
            "test_step_time_s",
            elapsed,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        self._checkpoint_start = time.time()

    def on_after_save_checkpoint(self, trainer, pl_module):
        if hasattr(self, "_checkpoint_start"):
            elapsed = time.time() - self._checkpoint_start
            self._checkpoint_total += elapsed

    def on_train_end(self, trainer, pl_module):
        train_wall = time.time() - getattr(self, "_train_wall_start", time.time())
        logger = getattr(trainer, "logger", None)

        if (
            isinstance(logger, AimLogger)
            and hasattr(logger, "experiment")
            and hasattr(logger.experiment, "set")
        ):
            logger.experiment.set("train_wall_time_s", train_wall)
            logger.experiment.set(
                "train_step_total_time_s", getattr(self, "_train_step_total", 0.0)
            )
            logger.experiment.set(
                "train_interstep_total_time_s",
                getattr(self, "_train_interstep_total", 0.0),
            )
            logger.experiment.set(
                "val_wall_total_time_s", getattr(self, "_val_wall_total", 0.0)
            )
            logger.experiment.set(
                "val_step_total_time_s", getattr(self, "_val_step_total", 0.0)
            )
            logger.experiment.set(
                "checkpoint_total_time_s", getattr(self, "_checkpoint_total", 0.0)
            )
            logger.experiment.set(
                "train_total_time_dhms", format_seconds_dhms(train_wall)
            )
            logger.experiment.set(
                "val_total_time_dhms",
                format_seconds_dhms(getattr(self, "_val_wall_total", 0.0)),
            )


class EmbeddingCheckpoint(Callback):
    def __init__(self, every_n_train_steps: int):
        super().__init__()
        self.every_n_train_steps = every_n_train_steps

    def on_validation_start(self, trainer, pl_module):
        step = int(trainer.global_step)
        should_save = (
            not trainer.sanity_checking
            and self.every_n_train_steps > 0
            and step > 0
            and step % self.every_n_train_steps == 0
        )
        pl_module._save_persistent_embedding_checkpoint = should_save

    def on_validation_end(self, trainer, pl_module):
        if getattr(pl_module, "_save_persistent_embedding_checkpoint", False):
            if hasattr(torch.cuda, "synchronize") and torch.cuda.is_available():
                torch.cuda.synchronize()


class ScalingTrainer:
    def __init__(self, config: ScalingConfig, fast_trainer: bool = False):
        # We don't store each attribute singularly as
        # we don't need to serialize and save the trainer
        self.config = config
        self.fast_trainer = fast_trainer
        self.val_check_interval = config.trainer.val_check_interval
        self.find_lr = config.trainer.find_lr

        # If we are not testing
        if not self.fast_trainer:
            # Checkpointer, pbar etc.
            self._init_callbacks()

            # Logger (aim)
            self._init_logger()

        # Init the trainer itself
        self._init_trainer()

    def _init_checkpointer(self) -> Callback:
        # Built the path to ckpt dir
        ckpt_dir = os.path.join(
            self.config.paths.path_to_ckpt_dir,
            self.config.metadata.title,  # needed when we want to collect multiple runs where same params vary intra run and other inter run
            self.config.metadata.run_name,
        )

        # Remove if already exists
        if os.path.exists(ckpt_dir):
            if getattr(self.config.trainer, "resume_training", False):
                warnings.warn("Dir already exists, preserving for resume")
            else:
                warnings.warn("Dir already exists, deleting")
                shutil.rmtree(ckpt_dir)

        # Create dirs
        os.makedirs(ckpt_dir, exist_ok=True)

        # Copy the config
        config_path = os.path.join(ckpt_dir, "config.yml")
        if not os.path.exists(config_path):
            self.config.save(config_path)

        return EmbeddingCheckpoint(
            every_n_train_steps=self.config.trainer.checkpoint_steps,
        )

    def _init_training_state_checkpointer(self) -> Callback:
        ckpt_dir = os.path.join(
            self.config.paths.path_to_ckpt_dir,
            self.config.metadata.title,
            self.config.metadata.run_name,
            "training_checkpoints",
        )
        os.makedirs(ckpt_dir, exist_ok=True)
        every_n_train_steps = getattr(
            self.config.trainer,
            "training_state_checkpoint_steps",
            getattr(self.config.trainer, "checkpoint_steps", 0),
        )
        return ResumableModelCheckpoint(
            dirpath=ckpt_dir,
            filename="step_{step}",
            every_n_train_steps=every_n_train_steps,
            save_last=True,
            save_top_k=0,
            auto_insert_metric_name=False,
            enable_version_counter=False,
        )

    def _init_logger(self) -> AimLogger:
        if self.config.paths.path_to_aimrepo:
            self.logger = AimLogger(
                repo=self.config.paths.path_to_aimrepo,
                experiment=self.config.metadata.run_name,
            )
            self.logger.experiment.set_artifacts_uri(
                f"file://{self.config.paths.path_to_aimrepo}/artifacts/"
            )
            self.logger.experiment["config"] = self.config.to_dict()
        else:
            self.logger = None

    def _init_pbar(self) -> TQDMProgressBar:
        return TQDMProgressBar(
            refresh_rate=1,
            leave=True,
        )

    def _init_callbacks(self):
        self.callbacks = []

        # Checkpointer if required
        if self.config.trainer.checkpoint:
            self.callbacks.append(self._init_checkpointer())
        self.callbacks.append(TrainingDataCursorCallback())
        if getattr(self.config.trainer, "training_state_checkpoint", False):
            self.callbacks.append(self._init_training_state_checkpointer())
        self.callbacks.append(
            SegmentTimer(getattr(self.config.trainer, "max_time", None))
        )
        # Always init logger
        self.callbacks.append(self._init_pbar())

        # Add step time logger
        self.callbacks.append(StepTimeLogger())

    def _init_trainer(self):
        precision_config = getattr(self.config.trainer, "precision", None)
        precision = precision_config.value if precision_config is not None else "32-true"
        if precision == "full":
            precision = "32-true"
        if self.fast_trainer:
            self.trainer = pl.Trainer(
                max_epochs=self.config.trainer.n_epochs,
                max_steps=self.config.trainer.n_steps,
                accelerator="auto",
                devices="auto",
                accumulate_grad_batches=self.config.trainer.accumulate_grad,
                gradient_clip_val=1,
                log_every_n_steps=0,
                enable_checkpointing=False,
                enable_progress_bar=False,
                num_sanity_val_steps=0,
                precision=precision,
                val_check_interval=self.val_check_interval,
                check_val_every_n_epoch=getattr(
                    self.config.trainer, "check_val_every_n_epoch", 1
                ),
                limit_val_batches=getattr(
                    self.config.trainer, "limit_val_batches", 1.0
                ),
                max_time=None,
            )
        else:
            self.trainer = pl.Trainer(
                max_epochs=self.config.trainer.n_epochs,
                max_steps=self.config.trainer.n_steps,
                accelerator="auto",  # recognizes device
                devices="auto",  # how many devices to use
                accumulate_grad_batches=self.config.trainer.accumulate_grad,
                gradient_clip_val=1,
                logger=self.logger,
                log_every_n_steps=1,
                val_check_interval=self.val_check_interval,
                enable_checkpointing=getattr(
                    self.config.trainer, "training_state_checkpoint", False
                ),
                callbacks=self.callbacks,
                enable_progress_bar=True,
                num_sanity_val_steps=0,
                precision=precision,
                check_val_every_n_epoch=getattr(
                    self.config.trainer, "check_val_every_n_epoch", 1
                ),
                limit_val_batches=getattr(
                    self.config.trainer, "limit_val_batches", 1.0
                ),
                max_time=None,
            )

    def get_param_count(self, model):
        param_total = sum(p.numel() for p in model.parameters())
        param_embedding = sum(
            p.numel()
            for name, p in model.named_parameters()
            if name.startswith("embedder")
        )
        param_encoder = sum(
            p.numel()
            for name, p in model.named_parameters()
            if name.startswith("encoder")
        )
        param_decoder = sum(
            p.numel()
            for name, p in model.named_parameters()
            if name.startswith("decoder")
        )
        return {
            "parameters_total": param_total,
            "parameters_total_human": human_format(param_total),
            "parameters_embedding": param_embedding,
            "parameters_embedding_human": human_format(param_embedding),
            "parameters_encoder": param_encoder,
            "parameters_encoder_human": human_format(param_encoder),
            "parameters_decoder": param_decoder,
            "parameters_decoder_human": human_format(param_decoder),
        }

    def _find_lr(
        self, model: pl.LightningModule, datamodule: pl.LightningDataModule
    ) -> None:
        import polars as pl
        from lightning.pytorch.tuner import Tuner

        # Perform LR Finding Procedure
        tuner = Tuner(self.trainer)
        lr_finder = tuner.lr_find(
            model, train_dataloaders=datamodule, update_attr=False, num_training=300
        )
        df = pl.from_dict(lr_finder.results)
        df.write_ipc("lr_finder_metrics.arrow")
        if self.logger:
            self.logger.experiment.log_artifact(
                "lr_finder_metrics.arrow", name="lr_finder_metrics"
            )
        os.remove("lr_finder_metrics.arrow")

    def fit(
        self, model: pl.LightningModule, datamodule: pl.LightningDataModule
    ) -> None:
        if self.find_lr:
            self._find_lr(model, datamodule)

        if not self.fast_trainer and self.logger:
            self.logger.log_hyperparams(self.get_param_count(model))

        if not self.fast_trainer and self.config.trainer.checkpoint:
            # Persist a step-0 embedding snapshot before any optimizer update.
            model._save_persistent_embedding_checkpoint = True
            self.trainer.validate(model=model, datamodule=datamodule, verbose=False)

        # Train the model
        ckpt_path = getattr(
            getattr(self.config, "paths", None), "path_to_resume_checkpoint", None
        )
        if not getattr(self.config.trainer, "resume_training", False):
            ckpt_path = None
        elif not ckpt_path:
            candidate = os.path.join(
                self.config.paths.path_to_ckpt_dir,
                self.config.metadata.title,
                self.config.metadata.run_name,
                "training_checkpoints",
                "last.ckpt",
            )
            if os.path.exists(candidate):
                ckpt_path = candidate
        self.trainer.fit(model, datamodule, ckpt_path=str(ckpt_path) if ckpt_path else None)

    def test(
        self,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        verbose: bool = True,
    ):
        result = self.trainer.test(model=model, datamodule=datamodule, verbose=verbose)
        if hasattr(self, "timer"):
            test_time = self.timer.time_elapsed("test")
            print(
                f"[Timer] Total test time: {format_seconds_dhms(test_time)} (raw: {test_time:.2f}s)"
            )
            if self.logger:
                self.logger.experiment.set(
                    "test_total_time_dhms", format_seconds_dhms(test_time)
                )
                self.logger.experiment.set("test_total_time_s", test_time)
        return result
