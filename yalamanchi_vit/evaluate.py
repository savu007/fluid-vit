"""Evaluate the model using shared one-step and rollout metrics."""
from forecasting.evaluate import main as evaluate_main


def main(argv=None):
    evaluate_main("yalamanchi_vit", argv)


if __name__ == "__main__":
    main()
