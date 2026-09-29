import argparse
import os
import signal
import shutil
import subprocess
import sys
import time
import warnings
from enum import Enum
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl-cache")
)

from core.config import ScalingConfig
from run.downstream_tasks import downstream_tasks
from tools.config_files import list_config_files
from tools.coarse_labels import DEFAULT_K, save_cell_type_to_coarse_map
from run.train import train, train_from_dir, train_cost
from tools.hvg import compute_hvgs
from tools.means import compute_means

warnings.filterwarnings("ignore", category=FutureWarning)


class CommandEnum(str, Enum):
    TRAIN = "train"
    MEANS = "means"
    HVG = "hvg"
    DOWNSTREAM_TASKS = "downstream_tasks"
    COARSE_LABELS = "coarse_labels"
    TRAIN_WITH_DOWNSTREAM = "train_with_downstream"


CHOICES = [e.value for e in CommandEnum]


def parse_command():
    cmd_parser = argparse.ArgumentParser(
        description="Take a command as option and perform the action.\n",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    cmd_parser.add_argument(
        "command",
        type=str,
        choices=CHOICES,
        help=f"Command to execute: {', '.join(CHOICES)}.",
    )
    args_cmd, remaining_args = cmd_parser.parse_known_args()
    return args_cmd, remaining_args


def add_config_arg(parser: argparse.ArgumentParser, required: bool = False):
    parser.add_argument(
        "--config",
        type=str,
        default="local_config.yml",
        required=required,
        help="Path to the base YAML configuration file (default: local_config.yml).",
    )


def _build_main_command(*args: str) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), *args]


def _stop_process(proc: subprocess.Popen, label: str, grace_s: float = 10.0) -> None:
    if proc.poll() is not None:
        return

    print(f"[orchestrator] Stopping {label} (pid={proc.pid})...")
    try:
        proc.send_signal(signal.SIGTERM)
        deadline = time.time() + grace_s
        while time.time() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.2)
        proc.kill()
    except ProcessLookupError:
        return


def _clear_run_dir_for_config(config_path: str) -> None:
    config = ScalingConfig(config_path)
    if getattr(config.trainer, "resume_training", False):
        print("[orchestrator] Resume requested; preserving run directory")
        return
    run_dir = _run_dir_for_config(config)
    if run_dir.exists():
        print(f"[orchestrator] Removing stale run directory: {run_dir}")
        shutil.rmtree(run_dir)


def _run_dir_for_config(config: ScalingConfig) -> Path:
    return (
        Path(config.paths.path_to_ckpt_dir)
        / config.metadata.title
        / config.metadata.run_name
    )


def _is_training_complete(config_path: str) -> bool:
    config = ScalingConfig(config_path)
    return (_run_dir_for_config(config) / "DONE").exists()


def _set_resume_flag(config_path: str) -> None:
    config = ScalingConfig(config_path)
    config.trainer.resume_training = True
    config.trainer.resume_dataloader = True
    config.save(config_path, overwrite=True)














def run_means(remaining_args):
    parser = argparse.ArgumentParser(
        description="Compute means for the given configuration.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="local_config.yml",
        help="Path to the YAML configuration file (default: local_config.yml).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for the stats batch size.",
    )
    args = parser.parse_args(remaining_args)
    compute_means(args.config, batch_size_override=args.batch_size)


def run_hvg(remaining_args):
    parser = argparse.ArgumentParser(
        description="Compute highly variable genes (HVGs) for the given configuration.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    add_config_arg(parser)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for the stats batch size.",
    )
    args = parser.parse_args(remaining_args)
    compute_hvgs(args.config, batch_size_override=args.batch_size)


def run_train(remaining_args, cost=False):
    parser = argparse.ArgumentParser(
        description="Train a model using the given configuration.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    add_config_arg(parser)
    parser.add_argument(
        "--config_dir", "--config-dir",
        type=str,
        default=None,
        help="Directory containing configuration files (default: None, used for jobarrays).",
    )
    parser.add_argument(
        "--id",
        type=int,
        default=None,
        help="ID of the job (default: None, used for jobarrays).",
    )
    parser.add_argument(
        "-A",
        action="store_true",
        help="Whether to also create hvg.csv, mean.csv, celltypes.csv",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume optimizer and dataloader state for this run.",
    )
    args = parser.parse_args(remaining_args)
    selected_config = args.config
    if args.config_dir is not None:
        if args.id is None:
            parser.error("--id is required with --config_dir")
        selected_config = str(list_config_files(args.config_dir)[args.id])
    if args.A:
        compute_hvgs(selected_config, overwrite=True)
        compute_means(selected_config, overwrite=True)

    # If config_dir is provided, train from that directory; otherwise, train from the config file.
    if args.config_dir is not None:
        train_from_dir(args.config_dir, args.id, cost, resume=args.resume)
    else:
        if cost:
            train_cost(args.config)
        else:
            train(args.config, resume=args.resume)






def run_downstream_tasks(remaining_args):
    parser = argparse.ArgumentParser(
        description="Consume saved embedding artifacts and compute downstream metrics.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    add_config_arg(parser)
    args = parser.parse_args(remaining_args)
    downstream_tasks(path_to_config=args.config)


def run_coarse_labels(remaining_args):
    parser = argparse.ArgumentParser(
        description="Build and save the fine-to-coarse label mapping from validation data.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    add_config_arg(parser)
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Validate and reuse an existing mapping without loading Census data.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional output path for the fine-to-coarse mapping CSV.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Target number of coarse anchors.",
    )
    parser.add_argument(
        "--ontology-path",
        type=str,
        default="cl.owl",
        help="Path to the ontology OWL file.",
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Skip tree visualizations and summary CSV outputs.",
    )
    args = parser.parse_args(remaining_args)

    config = ScalingConfig(args.config)
    config_k = getattr(getattr(config, "downstream", None), "coarse_k", DEFAULT_K)
    effective_k = args.k if args.k is not None else config_k
    if args.output_csv is not None:
        output_csv = args.output_csv
    else:
        config_dir = getattr(
            getattr(config, "paths", None),
            "path_to_coarse_labels_dir",
            None,
        )
        if config_dir is None:
            config_dir = config.paths.path_to_ckpt_dir
        output_csv = str(Path(config_dir) / f"cell_type_to_coarse_k{effective_k}.csv")

    save_cell_type_to_coarse_map(
        config_path=args.config,
        output_csv=output_csv,
        k=int(effective_k),
        ontology_path=args.ontology_path,
        save_visualizations=not args.no_visualizations,
        skip_if_exists=args.skip_if_exists,
    )


def run_train_with_downstream(remaining_args):
    parser = argparse.ArgumentParser(
        description="Run training and downstream metric consumption concurrently.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    add_config_arg(parser)
    parser.add_argument(
        "--config_dir", "--config-dir",
        type=str,
        default=None,
        help="Directory containing configuration files (used for jobarrays).",
    )
    parser.add_argument(
        "--id",
        type=int,
        default=None,
        help="ID of the config inside config_dir (used for jobarrays).",
    )
    parser.add_argument(
        "-A",
        action="store_true",
        help="Whether to also create hvg.csv, mean.csv, celltypes.csv before training.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume optimizer and dataloader state for this run.",
    )
    args = parser.parse_args(remaining_args)

    downstream_config = args.config
    if args.config_dir is not None:
        if args.id is None:
            raise ValueError("--id is required when --config_dir is provided.")
        config_files = list_config_files(args.config_dir)
        downstream_config = str(config_files[args.id])

    if args.resume:
        _set_resume_flag(downstream_config)

    train_cmd = ["train"]
    if args.config_dir is not None:
        train_cmd.extend(["--config_dir", args.config_dir])
        train_cmd.extend(["--id", str(args.id)])
    else:
        train_cmd.extend(["--config", args.config])
    if args.A:
        train_cmd.append("-A")
    if args.resume:
        train_cmd.append("--resume")

    downstream_cmd = [
        "downstream_tasks",
        "--config",
        downstream_config,
    ]

    _clear_run_dir_for_config(downstream_config)

    print(f"[orchestrator] Starting downstream consumer: {' '.join(downstream_cmd)}")
    downstream_proc = subprocess.Popen(_build_main_command(*downstream_cmd))
    print(f"[orchestrator] Starting training: {' '.join(train_cmd)}")
    train_proc = subprocess.Popen(_build_main_command(*train_cmd))

    def request_clean_stop(signum, frame):
        print(
            f"[orchestrator] Received signal {signum}; forwarding a graceful stop to training.",
            flush=True,
        )
        if train_proc.poll() is None:
            try:
                train_proc.send_signal(signal.SIGUSR1)
            except ProcessLookupError:
                pass

    handled_signals = [signal.SIGTERM]
    if hasattr(signal, "SIGUSR1"):
        handled_signals.append(signal.SIGUSR1)
    previous_handlers = {
        signum: signal.signal(signum, request_clean_stop)
        for signum in handled_signals
    }

    try:
        while True:
            train_rc = train_proc.poll()
            downstream_rc = downstream_proc.poll()

            if train_rc is not None and train_rc != 0:
                _stop_process(downstream_proc, "downstream consumer")
                raise SystemExit(train_rc)

            if downstream_rc is not None and downstream_rc != 0:
                _stop_process(train_proc, "training")
                raise SystemExit(downstream_rc)

            if train_rc is not None and downstream_rc is None:
                if _is_training_complete(downstream_config):
                    time.sleep(2.0)
                    continue
                print(
                    "[orchestrator] Training segment ended before DONE; stopping downstream consumer."
                )
                _stop_process(downstream_proc, "downstream consumer")
                _set_resume_flag(downstream_config)
                return

            if train_rc is not None and downstream_rc is not None:
                if train_rc != 0:
                    raise SystemExit(train_rc)
                if downstream_rc != 0:
                    raise SystemExit(downstream_rc)
                return

            time.sleep(2.0)
    except KeyboardInterrupt:
        _stop_process(train_proc, "training")
        _stop_process(downstream_proc, "downstream consumer")
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)










if __name__ == "__main__":
    args_cmd, remaining_args = parse_command()
    commands = {
        CommandEnum.TRAIN.value: run_train,
        CommandEnum.MEANS.value: run_means,
        CommandEnum.HVG.value: run_hvg,
        CommandEnum.DOWNSTREAM_TASKS.value: run_downstream_tasks,
        CommandEnum.COARSE_LABELS.value: run_coarse_labels,
        CommandEnum.TRAIN_WITH_DOWNSTREAM.value: run_train_with_downstream,
    }
    commands[args_cmd.command](remaining_args)
