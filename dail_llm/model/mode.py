"""Preserve caller mode around model evaluation helpers."""

from functools import wraps


def preserve_model_mode(function):
    @wraps(function)
    def wrapped(model, *args, **kwargs):
        previous = model.training
        try:
            return function(model, *args, **kwargs)
        finally:
            model.train(previous)

    return wrapped
