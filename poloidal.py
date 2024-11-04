from __future__ import annotations
import numpy as np
from scipy.special import jv, gamma, hyp2f1
from numba import njit


@njit
def phi(
    z: np.float64,
    p: np.ndarray,
    q: np.ndarray,
    z0: np.float64,
    deltaz: np.float64,
):
    """
    Returns solution of asymptotic approximated version of ODE (22)
    in Neukirch and Wiegelmann (2019) which defines the poloidal component
    of the magnetic field vector, in the case C-, C+ > 0 (for definitions
    see L Nadol PhD thesis).

    Vectorisation possible for p and q, which have to be passed as arrays of the
    same size. Vectorisation for z not possible due to differentiation between
    z < z0 and z > z0. Returns array of size p.shape = q.shape which is (nf, nf,).
    """

    rplus = p / deltaz
    rminus = q / deltaz

    r = rminus / rplus

    d = np.cosh(2.0 * rplus * z0) + np.multiply(r, np.sinh(2.0 * rplus * z0))

    if z - z0 < 0.0:
        return (
            np.cosh(2.0 * rplus * (z0 - z))
            + np.multiply(r, np.sinh(2.0 * rplus * (z0 - z)))
        ) / d

    else:
        return np.exp(-2.0 * rminus * (z - z0)) / d


@njit
def phi_complex(
    z: np.float64,
    v: np.ndarray,
    q: np.ndarray,
    z0: np.float64,
    deltaz: np.float64,
):
    """
    Returns solution of asymptotic approximated version of ODE (22)
    in Neukirch and Wiegelmann (2019) which defines the poloidal component
    of the magnetic field vector, in the case C-<0 or 1-2a<0 and k^2<k_critical
    (for definitions see L Nadol PhD thesis).

    Vectorisation possible for p and q, which have to be passed as arrays of the
    same size. Vectorisation for z not possible due to differentiation between
    z < z0 and z > z0. Returns array of size p.shape = q.shape which is (nf, nf,).
    """

    rplus = v / deltaz
    rminus = q / deltaz

    r = rminus / rplus

    d = np.cos(2.0 * rplus * z0) + np.multiply(r, np.sin(2.0 * rplus * z0))

    if z - z0 < 0.0:
        return (
            np.cos(2.0 * rplus * (z0 - z))
            + np.multiply(r, np.sin(2.0 * rplus * (z0 - z)))
        ) / d

    else:
        return np.exp(-2.0 * rminus * (z - z0)) / d


@njit
def dphidz(
    z: np.float64,
    p: np.ndarray,
    q: np.ndarray,
    z0: np.float64,
    deltaz: np.float64,
) -> np.ndarray[np.float64, np.dtype[np.float64]]:
    """
    Returns z-derivative of solution of asymptotic approximated version of ODE (22)
    in Neukirch and Wiegelmann (2019) which defines the poloidal component
    of the magnetic field vector, in the case C-, C+ > 0 (for definitions
    see L Nadol PhD thesis).

    Vectorisation possible for p and q, which have to be passed as arrays of the
    same size. Vectorisation for z not possible due to differentiation between
    z < z0 and z > z0. Returns array of size (nf, nf, nz,) whereas p.shape = q.shape = (nf, nf,).
    """

    rplus = p / deltaz
    rminus = q / deltaz

    r = rminus / rplus
    d = np.cosh(2.0 * rplus * z0) + np.multiply(r, np.sinh(2.0 * rplus * z0))

    if z - z0 < 0.0:
        return (
            -2.0
            * np.multiply(
                rplus,
                (
                    np.sinh(2.0 * rplus * (z0 - z))
                    + np.multiply(r, np.cosh(2.0 * rplus * (z0 - z)))
                ),
            )
            / d
        )

    else:
        return -2.0 * np.multiply(rminus, np.exp(-2.0 * rminus * (z - z0))) / d


@njit
def dphidz_complex(
    z: np.float64,
    v: np.ndarray,
    q: np.ndarray,
    z0: np.float64,
    deltaz: np.float64,
):
    """
    Returns z-derivative of solution of asymptotic approximated version of ODE (22)
    in Neukirch and Wiegelmann (2019) which defines the poloidal component
    of the magnetic field vector, in the case C-<0 or 1-2a<0 and k^2<k_critical
    (for definitions see L Nadol PhD thesis).

    Vectorisation possible for v and q, which have to be passed as arrays of the
    same size. Vectorisation for z not possible due to differentiation between
    z < z0 and z > z0. Returns array of size (nf, nf, nz,) whereas v.shape = q.shape = (nf, nf,).
    """

    rplus = v / deltaz
    rminus = q / deltaz

    r = rminus / rplus

    d = np.cos(2.0 * rplus * z0) + np.multiply(r, np.sin(2.0 * rplus * z0))

    if z - z0 < 0.0:
        return (
            2.0
            * np.multiply(
                rplus,
                (
                    np.sin(2.0 * rplus * (z0 - z))
                    - np.multiply(r, np.cos(2.0 * rplus * (z0 - z)))
                ),
            )
            / d
        )

    else:
        return -2.0 * np.multiply(rminus, np.exp(-2.0 * rminus * (z - z0))) / d


# @njit
def phi_low(
    z: np.float64, p: np.float64, q: np.float64, kappa: np.float64
) -> np.float64:
    """
    Returns solution of ODE (18) in Neukirch and Wiegelmann (2019)
    using the exponential switch function f(z) = a exp(-kappa z)
    introduced in Low (1991).

    Vectorisation should be possible for z, p and q, of which p and q have to be
    passed as arrays of the same size. Returns array of size (nf, nf, nz,) whereas
    p.shape = q.shape = (nf, nf,).
    """

    return jv(p, q * np.exp(-z * kappa / 2.0)) / jv(p, q)


# @njit
def dphidz_low(
    z: np.float64, p: np.float64, q: np.float64, kappa: np.float64
) -> np.float64:
    """
    Returns z-derivative of solution of ODE (18) in Neukirch and Wiegelmann (2019)
    using the exponential switch function f(z) = a exp(-kappa z)
    introduced in Low (1991).

    Vectorisation should be possible for z, p and q, of which p and q have to be
    passed as arrays of the same size. Returns array of size (nf, nf, nz,) whereas
    p.shape = q.shape = (nf, nf,).
    """

    return (
        (
            q * np.exp(-z * kappa / 2.0) * jv(p + 1.0, q * np.exp(-z * kappa / 2.0))
            - p * jv(p, q * np.exp(-z * kappa / 2.0))
        )
        * kappa
        / (2.0 * jv(p, q))
    )


# @njit
def phi_hypgeo(
    z: np.float64,
    p: np.ndarray,
    q: np.ndarray,
    z0: np.float64,
    deltaz: np.float64,
):
    """
    Returns solution of ODE (22) in Neukirch and Wiegelmann (2019)
    which defines the poloidal component of the magnetic field vector,
    in the case C1, C2 > 0 (for definitions see N+W (2019)).

    Vectorisation possible for p and q, which have to be passed as arrays of the
    same size. Vectorisation for z not possible due to differentiation between
    z < z0 and z > z0. Returns array of size p.shape = q.shape which is (nf, nf,).
    """

    w = (z - z0) / deltaz
    eta_d = 1.0 / (1.0 + np.exp(2.0 * w))

    if z - z0 < 0.0:
        phi = (1.0 / (1.0 + np.exp(2.0 * w))) ** (q + p) * np.exp(2.0 * p * w) * gamma(
            2 * q + 1
        ) * gamma(-2 * p) / (gamma(q - p) * gamma(q - p + 1)) * hyp2f1(
            p + q + 1, p + q, 2 * p + 1, 1 - eta_d
        ) + (
            1.0 / (1.0 + np.exp(2.0 * w))
        ) ** (
            q - p
        ) * np.exp(
            -2.0 * p * w
        ) * gamma(
            2 * q + 1
        ) * gamma(
            2 * p
        ) / (
            gamma(p + q + 1) * gamma(p + q)
        ) * hyp2f1(
            q - p, q - p + 1, -2 * p + 1, 1 - eta_d
        )
    else:
        phi = eta_d**q * (1 - eta_d) ** p * hyp2f1(p + q + 1, p + q, 2 * q + 1, eta_d)

    w0 = -z0 / deltaz
    eta0 = 1.0 / (1.0 + np.exp(2.0 * w0))

    phi0 = (1.0 / (1.0 + np.exp(2.0 * w0))) ** (q + p) * np.exp(2.0 * p * w0) * gamma(
        2 * q + 1
    ) * gamma(-2 * p) / (gamma(q - p + 1) * gamma(q - p)) * hyp2f1(
        p + q + 1, p + q, 2 * p + 1, 1 - eta0
    ) + (
        1.0 / (1.0 + np.exp(2.0 * w0))
    ) ** (
        q - p
    ) * np.exp(
        -2.0 * p * w0
    ) * gamma(
        2 * q + 1
    ) * gamma(
        2 * p
    ) / (
        gamma(p + q + 1) * gamma(p + q)
    ) * hyp2f1(
        q - p, q - p + 1, -2 * p + 1, 1 - eta0
    )

    return phi / phi0


# @njit
def dphidz_hypgeo(
    z: np.float64,
    p: np.ndarray,
    q: np.ndarray,
    z0: np.float64,
    deltaz: np.float64,
):
    """
    Returns z-derivative of solution of ODE (22) in Neukirch and Wiegelmann (2019)
    which defines the poloidal component of the magnetic field vector,
    in the case C1, C2 > 0 (for definitions see N+W (2019)).

    Vectorisation possible for p and q, which have to be passed as arrays of the
    same size. Vectorisation for z not possible due to differentiation between
    z < z0 and z > z0. Returns array of size p.shape = q.shape which is (nf, nf,).
    """
    w = (z - z0) / deltaz
    eta_d = 1.0 / (1.0 + np.exp(2.0 * w))

    if z - z0 < 0.0:
        term1 = (
            gamma(2 * q + 1)
            * gamma(-2 * p)
            / (gamma(q - p) * gamma(q - p + 1))
            * (1.0 / (1.0 + np.exp(2.0 * w))) ** (q + p + 1)
            * (q * np.exp(2.0 * (p + 1) * w) - p * np.exp(2 * p * w))
            * hyp2f1(p + q + 1, p + q, 2 * p + 1, 1 - eta_d)
        )
        term2 = (
            gamma(2 * q + 1)
            * gamma(2 * p)
            / (gamma(q + p) * gamma(q + p + 1))
            * (1.0 / (1.0 + np.exp(2.0 * w))) ** (q - p + 1)
            * (q * np.exp(-2.0 * (p - 1) * w) - p * np.exp(-2 * p * w))
            * hyp2f1(q - p, q - p + 1, -2 * p + 1, 1 - eta_d)
        )
        term3 = (
            gamma(2 * q + 2)
            * gamma(-2 * p - 1)
            / (gamma(q - p) * gamma(q - p + 1))
            * (p + q + 1)
            * (p + q)
            / (2 * q + 1)
            * (1.0 / (1.0 + np.exp(2.0 * w))) ** (q + p + 2)
            * np.exp(2.0 * (p + 1) * w)
            * hyp2f1(q + p + 2, q + p + 1, 2 * p + 2, 1 - eta_d)
        )
        term4 = (
            gamma(2 * q + 2)
            * gamma(2 * p + 1)
            / (gamma(q + p + 1) * gamma(q + p + 2))
            * (p + q + 1)
            * (p + q)
            / (2 * q + 1)
            * (1.0 / (1.0 + np.exp(2.0 * w))) ** (q - p + 1)
            * np.exp(-2.0 * p * w)
            * hyp2f1(q - p, q - p + 1, -2 * p, 1 - eta_d)
        )
        dphi = term1 + term2 + term3 + term4
    else:
        dphi = (
            q * eta_d**q * (1 - eta_d) ** (p + 1)
            - p * eta_d ** (q + 1) * (1 - eta_d) ** p
        ) * hyp2f1(p + q + 1, p + q, 2 * q + 1, eta_d) + (p + q + 1) * (p + q) / (
            2 * q + 1
        ) * eta_d ** (
            q + 1
        ) * (
            1 - eta_d
        ) ** (
            p + 1
        ) * hyp2f1(
            p + q + 2, p + q + 1, 2 * q + 2, eta_d
        )

    w0 = -z0 / deltaz
    eta0 = 1.0 / (1.0 + np.exp(2.0 * w0))

    phi0 = (1.0 / (1.0 + np.exp(2.0 * w0))) ** (q + p) * np.exp(2.0 * p * w0) * gamma(
        2 * q + 1
    ) * gamma(-2 * p) / (gamma(q - p + 1) * gamma(q - p)) * hyp2f1(
        p + q + 1, p + q, 2 * p + 1, 1 - eta0
    ) + (
        1.0 / (1.0 + np.exp(2.0 * w0))
    ) ** (
        q - p
    ) * np.exp(
        -2.0 * p * w0
    ) * gamma(
        2 * q + 1
    ) * gamma(
        2 * p
    ) / (
        gamma(p + q + 1) * gamma(p + q)
    ) * hyp2f1(
        q - p, q - p + 1, -2 * p + 1, 1 - eta0
    )

    return -2.0 / deltaz * dphi / phi0


"""
def phi_vectorized(
    z: np.ndarray, p: np.ndarray, q: np.ndarray, z0: float, deltaz: float
) -> np.ndarray:
    
    # Vectorized version of the phi function that operates on NumPy arrays.

    rplus = p / deltaz
    rminus = q / deltaz

    r = rminus / rplus
    d = np.cosh(2.0 * rplus * z0) + r * np.sinh(2.0 * rplus * z0)

    result = np.empty_like(z)
    mask = z < z0
    result[mask] = (
        np.cosh(2.0 * rplus * (z0 - z[mask]))
        + r[mask] * np.sinh(2.0 * rplus * (z0 - z[mask]))
    ) / d[mask]
    result[~mask] = np.exp(-2.0 * rminus[~mask] * (z[~mask] - z0)) / d[~mask]

    return result


def dphidz_vectorized(
    z: np.ndarray, p: np.ndarray, q: np.ndarray, z0: float, deltaz: float
) -> np.ndarray:
    
    # Vectorized version of the dphidz function that operates on NumPy arrays.
    
    rplus = p / deltaz
    rminus = q / deltaz

    r = rminus / rplus
    d = np.cosh(2.0 * rplus * z0) + r * np.sinh(2.0 * rplus * z0)

    result = np.empty_like(z)
    mask = z < z0
    result[mask] = (
        -2.0
        * rplus
        * (
            np.sinh(2.0 * rplus * (z0 - z[mask]))
            + r[mask] * np.cosh(2.0 * rplus * (z0 - z[mask]))
        )
    ) / d[mask]
    result[~mask] = (
        -2.0 * rminus[~mask] * np.exp(-2.0 * rminus[~mask] * (z[~mask] - z0))
    ) / d[~mask]
"""
