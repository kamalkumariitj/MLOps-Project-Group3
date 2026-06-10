import argparse
import subprocess
import sys
import os,certifi

from pathlib import Path
from typing import List

from config import load_config

os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

def parse_args() -> argparse.Namespace:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Entry program for ANLI pipeline: data -> train -> eval.")
    parser.add_argument("--run-mode", choices=["SMALL_RUN", "FULL_RUN"], default=cfg.run_mode)
    parser.add_argument("--experiment-version", choices=sorted(cfg.experiment_configs.keys()), default=cfg.experiment_version)
    parser.add_argument("--stage", choices=["all", "data", "train", "eval"], default="all")
    parser.add_argument("--python-bin", default=sys.executable)

    parser.add_argument("--data-path", default=cfg.data_pickle_path)
    parser.add_argument("--label-map-path", default=cfg.label_map_path)
    parser.add_argument("--output-dir", default=cfg.output_dir)
    parser.add_argument("--logging-dir", default=cfg.logging_dir)
    parser.add_argument("--report-path", default=cfg.eval_report_path)

    parser.add_argument("--train-max-steps", type=int, default=None)
    parser.add_argument("--run-name", default=cfg.wandb_run_name)

    parser.add_argument("--enable-wandb", dest="enable_wandb", action="store_true", default=None)
    parser.add_argument("--disable-wandb", dest="enable_wandb", action="store_false")
    parser.add_argument("--push-to-hub", dest="push_to_hub", action="store_true", default=None)
    parser.add_argument("--no-push-to-hub", dest="push_to_hub", action="store_false")
    return parser.parse_args()


def run_step(name: str, command: List[str]) -> None:
    print(f"\n[{name}] Running: {' '.join(command)}")
    subprocess.run(command, check=True)
    print(f"[{name}] Completed.")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent

    run_data = args.stage in {"all", "data"}
    run_train = args.stage in {"all", "train"}
    run_eval = args.stage in {"all", "eval"}

    if run_data:
        data_cmd = [
            args.python_bin,
            str(root / "data.py"),
            "--run-mode",
            args.run_mode,
            "--output-pickle",
            args.data_path,
            "--label-map-path",
            args.label_map_path,
        ]
        run_step("data", data_cmd)

    if run_train:
        train_cmd = [
            args.python_bin,
            str(root / "train.py"),
            "--run-mode",
            args.run_mode,
            "--experiment-version",
            args.experiment_version,
            "--data-path",
            args.data_path,
            "--output-dir",
            args.output_dir,
            "--logging-dir",
            args.logging_dir,
        ]
        if args.train_max_steps is not None:
            train_cmd.extend(["--max-steps", str(args.train_max_steps)])
        if args.run_name:
            train_cmd.extend(["--run-name", args.run_name])
        if args.enable_wandb is not None:
            train_cmd.append("--enable-wandb" if args.enable_wandb else "--disable-wandb")
        if args.push_to_hub is not None:
            train_cmd.append("--push-to-hub" if args.push_to_hub else "--no-push-to-hub")
        run_step("train", train_cmd)

    if run_eval:
        eval_cmd = [
            args.python_bin,
            str(root / "eval.py"),
            "--run-mode",
            args.run_mode,
            "--data-path",
            args.data_path,
            "--model-dir",
            args.output_dir,
            "--report-path",
            args.report_path,
        ]
        if args.enable_wandb is not None:
            eval_cmd.append("--enable-wandb" if args.enable_wandb else "--disable-wandb")
        run_step("eval", eval_cmd)

    print("\nPipeline execution finished.")


if __name__ == "__main__":
    main()


# main command to run locally
#     python3 main.py --run-mode SMALL_RUN --disable-wandb --no-push-to-hub
