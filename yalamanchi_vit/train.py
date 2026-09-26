"""Train the Yalamanchi-inspired model with the shared forecast objective."""
import argparse
from pathlib import Path
from .engine import train
from .config import load_config
from .model import ARCHITECTURE, YalamanchiViT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    args = parser.parse_args(argv)
    train(load_config(args.config), YalamanchiViT, ARCHITECTURE)


if __name__ == "__main__":
    main()
