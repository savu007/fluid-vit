"""Train the matched model with MSE plus energy-spectrum regularization."""
from forecasting.train_physical import main as train_main


def main(argv=None):
    train_main("baseline_vit", argv)


if __name__ == "__main__":
    main()
