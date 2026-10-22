"""Recursive four-frame context: future targets never enter the model input."""
import torch


def forecast_five(model, history, times, memory_callback=None):
    """Physical-valued [B,4,C,X,Y,Z] -> [B,5,C,X,Y,Z].

    times is [B,9] in recorded simulation-time units. Autograd is retained;
    callers use inference_mode for evaluation. Target field values are not
    arguments, so teacher forcing cannot occur in this function.
    """
    if history.ndim != 6 or history.shape[1] != 4:
        raise ValueError('Expected history [B,4,C,X,Y,Z]')
    if times.shape != (history.shape[0], 9):
        raise ValueError('Expected recorded times [B,9]')
    if not torch.isfinite(history).all() or not torch.isfinite(times).all():
        raise ValueError('Inputs must be finite')
    if not torch.all(times[:, 1:] > times[:, :-1]):
        raise ValueError('Times must increase strictly')
    state = history
    predictions = []
    for step in range(5):
        prediction = model(state, times[:, step:step + 4], times[:, step + 4])
        if prediction.shape != history[:, 0].shape:
            raise ValueError('One-step model returned the wrong shape')
        if not torch.isfinite(prediction).all():
            raise FloatingPointError(f'Nonfinite prediction at lead {step + 1}')
        if memory_callback is not None:
            memory_callback(step + 1)
        predictions.append(prediction)
        state = torch.cat((state[:, 1:], prediction[:, None]), dim=1)
    return torch.stack(predictions, dim=1)
