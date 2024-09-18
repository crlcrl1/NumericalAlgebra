from numpy.typing import NDArray

def forward_substitution(a: NDArray, b: NDArray) -> None:
    n = a.shape[0]
    for i in range(n - 1):
        b[i] = b[i] / a[i, i]
        b[i + 1:] = b[i + 1:] - b[i] * a[i + 1:, i]
    b[n - 1] = b[n - 1] / a[n - 1, n - 1]


def backward_substitution(a: NDArray, b: NDArray) -> None:
    n = a.shape[0]
    for i in range(n - 1, 0, -1):
        b[i] = b[i] / a[i, i]
        b[:i] = b[:i] - b[i] * a[:i, i]
    b[0] = b[0] / a[0, 0]