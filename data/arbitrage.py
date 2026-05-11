import numpy as np
from numpy.typing import NDArray

def _d2_dk2_nonuniform(
        w: NDArray[np.floating],
        k_grid: NDArray[np.floating],
        ) -> NDArray[np.floating]:
    """
    Discrete second derivative of w(k, tau) along k at interior points,
    accurate on non-uniform grids. Returns shape (n_k - 2, n_tau).

    Composing np.gradient twice gives a wider-stencil approximation that
    amplifies noise where spacing jumps (e.g., at our ATM/wing segment join).
    The explicit non-uniform 3-point formula below is exact for quadratics
    regardless of spacing.
    """
    h = np.diff(k_grid)              # (n_k - 1,)
    h_minus = h[:-1, None]           # (n_k - 2, 1) for broadcasting against (n_k, n_tau)
    h_plus = h[1:, None]             # (n_k - 2, 1)
    df_minus = (w[1:-1, :] - w[:-2, :]) / h_minus
    df_plus = (w[2:, :] - w[1:-1, :]) / h_plus
    return 2.0 / (h_minus + h_plus) * (df_plus - df_minus)


def is_arbitrage_free(iv: NDArray[np.floating], taus: NDArray[np.floating],
                      k_grid: NDArray[np.floating],
                      atol_cal: float = 1e-4,
                      atol_btf: float = 1e-1) -> bool:
    '''
    Check that the IV surface has no arbitrage.
    Conceptually: positivity, calendar (total variance non-decreasing in tau),
    and butterfly (total variance convex in k). Heston is arbitrage-free
    analytically — the gate exists to catch numerical pathologies.

    Tolerances are split: butterfly is much looser than calendar because the
    discrete second derivative on the non-uniform k-grid is noisier than the
    first difference in tau.
    '''
    # Reject if NaNs are present or negative ivs
    if np.any(~np.isfinite(iv)) or np.any(iv <= 0):
        return False
    # Convert implied volatility to total variance
    # Also flatten to 1 axis using the second term
    w = (iv ** 2) * taus[None, :]

    # Calendar: reject when total variance drops by more than the tolerance.
    if np.any(np.diff(w, axis=1) < -atol_cal):
        return False

    # Butterfly: w must be convex in k at each maturity.
    d2w_dk2 = _d2_dk2_nonuniform(w, k_grid)
    if np.any(d2w_dk2 < -atol_btf):
        return False

    return True


def audit_dataset(
        surfaces: NDArray[np.floating],
        k_grid: NDArray[np.floating],
        tau_grid: NDArray[np.floating],
        atol_cal: float = 1e-4,
        atol_btf: float = 1e-1,
        ) -> dict[str, int]:
    '''
    Post-hoc audit, checks whole dataset for any type of arbitrage violation.
    Calendar and butterfly tolerances are split because the discrete second
    derivative used in the butterfly check is much noisier on the non-uniform
    k-grid than the first difference in tau.
    '''
    n_total = len(surfaces)
    n_neg = 0
    n_cal = 0
    n_btf = 0

    for iv in surfaces:
        if np.any(~np.isfinite(iv)) or np.any(iv <= 0):
            n_neg += 1
            continue

        w = (iv ** 2) * tau_grid[None, :]

        if np.any(np.diff(w, axis=1) < -atol_cal):
            n_cal += 1

        d2w_dk2 = _d2_dk2_nonuniform(w, k_grid)
        if np.any(d2w_dk2 < -atol_btf):
            n_btf += 1

    return {
        "n_total": int(n_total),
        "negative_iv": int(n_neg),
        "calendar_violations": int(n_cal),
        "butterfly_violations": int(n_btf),
    }