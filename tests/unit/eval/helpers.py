from types import SimpleNamespace


def make_eval_result(metrics: dict) -> SimpleNamespace:
    return SimpleNamespace(metrics=metrics)
