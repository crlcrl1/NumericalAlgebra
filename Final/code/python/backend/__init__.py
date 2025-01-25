from .equation import StrokesEquation


def get_equation(backend: str, n: int) -> StrokesEquation:
    if backend.lower() == "torch":
        from .pytorch.equation import TorchStrokesEquation
        return TorchStrokesEquation(n)
    elif backend.lower() == "jax":
        from .jax.equation import JaxStrokesEquation
        return JaxStrokesEquation(n)
    else:
        raise ValueError("Unsupported backend")
