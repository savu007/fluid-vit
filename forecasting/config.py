"""Validate the supported temporal objectives identically for both models."""


def validate_objective(training):
    horizon = training["horizon"]
    if type(horizon) is not int or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if training["objective"] not in ("single_step", "rollout"):
        raise ValueError("objective must be single_step or rollout")
    if training["objective"] == "single_step" and horizon != 1:
        raise ValueError("single_step requires horizon=1")
    if training["objective"] == "rollout" and horizon < 2:
        raise ValueError("rollout requires horizon>=2")
