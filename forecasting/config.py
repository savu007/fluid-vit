"""Validate the supported temporal objectives identically for both models."""


def validate_objective(training):
    horizon = training["horizon"]
    if type(horizon) is not int or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if training["objective"] not in ("single_step", "rollout", "pushforward"):
        raise ValueError("objective must be single_step, rollout, or pushforward")
    if training["objective"] == "single_step" and horizon != 1:
        raise ValueError("single_step requires horizon=1")
    if training["objective"] in ("rollout", "pushforward") and horizon < 2:
        raise ValueError("rollout and pushforward require horizon>=2")
