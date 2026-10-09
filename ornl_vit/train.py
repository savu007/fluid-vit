"""Train the approved four-frame/five-step MATEY adaptation."""
import argparse
from .config import load_config
from .engine import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == '__main__':
    main()
