from __future__ import annotations

import argparse as ap

from hydra import compose, initialize
from omegaconf import DictConfig

from ._cli import (
    add_arguments_emb,
    add_arguments_tx,
    run_emb_fit,
    run_emb_transform,
    run_emb_query,
    run_emb_preprocess,
    run_emb_eval,
    run_tx_infer,
    run_tx_predict,
    run_tx_preprocess_train,
    run_tx_train,
)


def get_args() -> tuple[ap.Namespace, list[str]]:
    """Parse known args and return remaining args for Hydra overrides"""
    parser = ap.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest="command")
    add_arguments_emb(subparsers.add_parser("emb"))
    add_arguments_tx(subparsers.add_parser("tx"))

    # Use parse_known_args to get both known args and remaining args
    return parser.parse_args()


def load_hydra_config(method: str, overrides: list[str] = None) -> DictConfig:
    """Load Hydra config with optional overrides"""
    if overrides is None:
        overrides = []

    # Initialize Hydra with the path to your configs directory
    # Adjust the path based on where this file is relative to configs/
    with initialize(version_base=None, config_path="configs"):
        if method == "emb":
            cfg = compose(config_name="state-defaults", overrides=overrides)
        elif method == "tx":
            cfg = compose(config_name="config", overrides=overrides)
        else:
            raise ValueError(f"Unknown method: {method}")
    return cfg


def show_hydra_help(method: str):
    """Show Hydra configuration help with all parameters"""
    from omegaconf import OmegaConf

    # Load the default config to show structure
    cfg = load_hydra_config(method)

    print("Hydra Configuration Help")
    print("=" * 50)
    print(f"Configuration for method: {method}")
    print()
    print("Full configuration structure:")
    print(OmegaConf.to_yaml(cfg))
    print()
    print("Usage examples:")
    print("  Override single parameter:")
    print("    uv run state tx train data.batch_size=64")
    print()
    print("  Override nested parameter:")
    print("    uv run state tx train model.kwargs.hidden_dim=512")
    print()
    print("  Override multiple parameters:")
    print("    uv run state tx train data.batch_size=64 training.lr=0.001")
    print()
    print("  Change config group:")
    print("    uv run state tx train data=custom_data model=custom_model")
    print()
    print("Available config groups:")

    # Show available config groups
    from pathlib import Path

    config_dir = Path(__file__).parent / "configs"
    if config_dir.exists():
        for item in config_dir.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                configs = [f.stem for f in item.glob("*.yaml")]
                if configs:
                    print(f"  {item.name}: {', '.join(configs)}")

    exit(0)


def main():
    args = get_args()

    if args.command == "emb":
        if args.subcommand == "fit":
            cfg = load_hydra_config("emb", args.hydra_overrides)
            run_emb_fit(cfg, args)
        elif args.subcommand == "transform":
            run_emb_transform(args)
        elif args.subcommand == "query":
            run_emb_query(args)
        elif args.subcommand == "preprocess":
            run_emb_preprocess(args)
        elif args.subcommand == "eval":
            run_emb_eval(args)
    elif args.command == "tx":
        if args.subcommand == "train":
            if hasattr(args, "help") and args.help:
                show_hydra_help("tx")
            else:
                cfg = load_hydra_config("tx", args.hydra_overrides)
                run_tx_train(cfg)
        elif args.subcommand == "predict":
            run_tx_predict(args)
        elif args.subcommand == "infer":
            run_tx_infer(args)
        elif args.subcommand == "preprocess_train":
            run_tx_preprocess_train(args.adata, args.output, args.num_hvgs)


if __name__ == "__main__":
    main()
