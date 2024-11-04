from __future__ import annotations
import numpy as np
from mhsflex.b3d import get_phi_dphi

from mhsflex.field2d import Field2dData
from typing import Tuple

from dataclasses import dataclass

import pickle

from functools import cached_property

from mhsflex.field2d import Field2dData

from mhsflex.switch import f, dfdz, f_low, dfdz_low

T_PHOTOSPHERE = 5600.0  # Photospheric temperature
T_CORONA = 2.0 * 10.0**6  # Coronal temperature

G_SOLAR = 272.2  # m/s^2
KB = 1.380649 * 10**-23  # Boltzmann constant in Joule/ Kelvin = kg m^2/(Ks^2)
MBAR = 1.67262 * 10**-27  # mean molecular weight (proton mass)
RHO0 = 2.7 * 10**-4  # plasma density at z = 0 in kg/(m^3)
P0 = T_PHOTOSPHERE * KB * RHO0 / MBAR  # plasma pressure in kg/(s^2 m)
MU0 = 1.25663706 * 10**-6  # permeability of free space in mkg/(s^2A^2)

L = 10**6  # Lengthscale Mm


@dataclass
class Field3dData:
    nx: np.int32
    ny: np.int32
    nz: np.int32
    nf: np.int32
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    bz: np.ndarray
    field: np.ndarray
    dfield: np.ndarray

    a: float
    b: float
    alpha: float
    z0: np.float64
    deltaz: np.float64

    tanh: bool

    def save(self, path):
        for name, attribute in self.__dict__.items():
            name = ".".join((name, "pkl"))
            with open("/".join((path, name)), "wb") as f:
                pickle.dump(attribute, f)

    @classmethod
    def load(cls, path):
        my_model = {}
        for name in cls.__annotations__:
            file_name = ".".join((name, "pkl"))
            with open("/".join((path, file_name)), "rb") as f:
                my_model[name] = pickle.load(f)
        return cls(**my_model)

    @cached_property
    def btemp(self) -> np.ndarray:

        T0 = (T_PHOTOSPHERE + T_CORONA * np.tanh(self.z0 / self.deltaz)) / (
            1.0 + np.tanh(self.z0 / self.deltaz)
        )
        T1 = (T_CORONA - T_PHOTOSPHERE) / (1.0 + np.tanh(self.z0 / self.deltaz))

        return T0 + T1 * np.tanh((self.z - self.z0) / self.deltaz)

    @cached_property
    def bpressure(self) -> np.ndarray:

        T0 = (T_PHOTOSPHERE + T_CORONA * np.tanh(self.z0 / self.deltaz)) / (
            1.0 + np.tanh(self.z0 / self.deltaz)
        )  # in Kelvin
        T1 = (T_CORONA - T_PHOTOSPHERE) / (
            1.0 + np.tanh(self.z0 / self.deltaz)
        )  # in Kelvin
        H = KB * T0 / (MBAR * G_SOLAR) / L  # in m

        q1 = self.deltaz / (2.0 * H * (1.0 + T1 / T0))
        q2 = self.deltaz / (2.0 * H * (1.0 - T1 / T0))
        q3 = self.deltaz * (T1 / T0) / (H * (1.0 - (T1 / T0) ** 2))

        p1 = (
            2.0
            * np.exp(-2.0 * (self.z - self.z0) / self.deltaz)
            / (1.0 + np.exp(-2.0 * (self.z - self.z0) / self.deltaz))
            / (1.0 + np.tanh(self.z0 / self.deltaz))
        )
        p2 = (1.0 - np.tanh(self.z0 / self.deltaz)) / (
            1.0 + np.tanh((self.z - self.z0) / self.deltaz)
        )
        p3 = (1.0 + T1 / T0 * np.tanh((self.z - self.z0) / self.deltaz)) / (
            1.0 - T1 / T0 * np.tanh(self.z0 / self.deltaz)
        )

        return (p1**q1) * (p2**q2) * (p3**q3)

    @cached_property
    def bdensity(self) -> np.ndarray:

        T0 = (T_PHOTOSPHERE + T_CORONA * np.tanh(self.z0 / self.deltaz)) / (
            1.0 + np.tanh(self.z0 / self.deltaz)
        )  # in Kelvin
        T1 = (T_CORONA - T_PHOTOSPHERE) / (
            1.0 + np.tanh(self.z0 / self.deltaz)
        )  # in Kelvin

        temp0 = T0 - T1 * np.tanh(self.z0 / self.deltaz)  # in Kelvin
        dummypres = self.bpressure  # normalised
        dummytemp = self.btemp / temp0  # normalised

        return dummypres / dummytemp

    @cached_property
    def dpressure(self) -> np.ndarray:

        bz_matrix = self.field[:, :, :, 2]  # in Gauss
        z_matrix = np.zeros_like(bz_matrix)
        z_matrix[:, :, :] = self.z

        B0 = self.field[:, :, 0, 2].max()  # in Gauss

        if self.tanh:
            return (
                -f(z_matrix, self.z0, self.deltaz, self.a, self.b)  # normalised
                / 2.0
                * bz_matrix**2.0
                / B0**2.0
            )
        else:
            kappa = self.deltaz
            a = self.a
            return -f_low(z_matrix, a, kappa) / 2.0 * bz_matrix**2.0 / B0**2.0

    @cached_property
    def ddensity(self) -> np.ndarray:

        bz_matrix = self.field[:, :, :, 2]  # in Gauss
        z_matrix = np.zeros_like(bz_matrix)
        z_matrix[:, :, :] = self.z

        bdotbz_matrix = np.zeros_like(bz_matrix)

        bdotbz_matrix = (
            self.field[:, :, :, 0] * self.dfield[:, :, :, 0]
            + self.field[:, :, :, 1] * self.dfield[:, :, :, 1]
            + self.field[:, :, :, 2] * self.dfield[:, :, :, 2]
        )  # in Gauss**2

        B0 = self.field[:, :, 0, 2].max()  # in Gauss

        if self.tanh:
            return (
                dfdz(z_matrix, self.z0, self.deltaz, self.a, self.b)  # normalised
                / 2.0
                * bz_matrix**2
                / B0**2
                + f(z_matrix, self.z0, self.deltaz, self.a, self.b)  # normalised
                * bdotbz_matrix  # normalised
                / B0**2
            )
        else:
            kappa = self.deltaz
            a = self.a
            return (
                dfdz_low(z_matrix, a, kappa) / 2.0 * bz_matrix**2 / B0**2
                + f_low(z_matrix, a, kappa) * bdotbz_matrix / B0**2
            )

    @cached_property
    def fpressure(self) -> np.ndarray:

        bp_matrix = np.zeros_like(self.dpressure)
        bp_matrix[:, :, :] = self.bpressure

        B0 = self.field[
            :, :, 0, 2
        ].max()  # Gauss background magnetic field strength in 10^-4 kg/(s^2A) = 10^-4 T
        PB0 = (B0 * 10**-4) ** 2 / (
            2 * MU0
        )  # magnetic pressure b0**2 / 2mu0 in kg/(s^2m)
        BETA0 = P0 / PB0  # Plasma Beta, ration plasma to magnetic pressure

        return BETA0 / 2.0 * bp_matrix + self.dpressure  # * (B0 * 10**-4) ** 2.0 / MU0

    @cached_property
    def fdensity(self) -> np.ndarray:

        T0 = (T_PHOTOSPHERE + T_CORONA * np.tanh(self.z0 / self.deltaz)) / (
            1.0 + np.tanh(self.z0 / self.deltaz)
        )
        H = KB * T0 / (MBAR * G_SOLAR) / L
        B0 = self.field[
            :, :, 0, 2
        ].max()  # Gauss background magnetic field strength in 10^-4 kg/(s^2A) = 10^-4 T
        PB0 = (B0 * 10**-4) ** 2 / (
            2 * MU0
        )  # magnetic pressure b0**2 / 2mu0 in kg/(s^2m)
        BETA0 = P0 / PB0  # Plasma Beta, ration plasma to magnetic pressure

        bd_matrix = np.zeros_like(self.ddensity)
        bd_matrix[:, :, :] = self.bdensity

        return (
            BETA0 / (2.0 * H) * T0 / T_PHOTOSPHERE * bd_matrix + self.ddensity
        )  #  *(B0 * 10**-4) ** 2.0 / (MU0 * G_SOLAR * L)


def calculate_magfield(
    field2d: Field2dData,
    a: float,
    b: float,
    alpha: float,
    z0: np.float64,
    deltaz: np.float64,
    asymptotic=True,
    tanh=True,
) -> Field3dData:

    mf3d, dbz3d = b3d_fb(field2d, a, b, alpha, z0, deltaz, asymptotic, tanh)

    data = Field3dData(
        nx=field2d.nx,
        ny=field2d.ny,
        nz=field2d.nz,
        nf=field2d.nf,
        x=field2d.x,
        y=field2d.y,
        z=field2d.z,
        bz=field2d.bz,
        field=mf3d,
        dfield=dbz3d,
        a=a,
        b=b,
        alpha=alpha,
        z0=z0,
        deltaz=deltaz,
        tanh=tanh,
    )

    return data


def btemp_linear(
    field3d: Field3dData, heights: np.ndarray, temps: np.ndarray
) -> np.ndarray:

    temp = np.zeros_like(field3d.z)

    if len(heights) != len(temps):
        raise ValueError("Number of heights and temperatures do not match")

    for iz, z in enumerate(field3d.z):

        h_index = 0

        for i in range(0, len(heights) - 1):
            if z >= heights[i] and z <= heights[i + 1]:
                h_index = i

        temp[iz] = temps[h_index] + (temps[h_index + 1] - temps[h_index]) / (
            heights[h_index + 1] - heights[h_index]
        ) * (z - heights[h_index])

    return temp


def bpressure_linear(
    field3d: Field3dData, heights: np.ndarray, temps: np.ndarray
) -> np.ndarray:

    temp = np.zeros_like(field3d.z)

    for iheight, height in enumerate(heights):
        if height == field3d.z0:
            T0 = temps[iheight]

    H = KB * T0 / (MBAR * G_SOLAR) / L

    for iz, z in enumerate(field3d.z):

        h_index = 0

        for i in range(0, len(heights) - 1):
            if heights[i] <= z <= heights[i + 1]:
                h_index = i

        pro = 1.0
        for j in range(0, h_index):
            qj = (temps[j + 1] - temps[j]) / (heights[j + 1] - heights[j])
            expj = -T0 / (H * qj)
            tempj = (
                abs(temps[j] + qj * (heights[j + 1] - heights[j])) / temps[j]
            ) ** expj
            pro = pro * tempj

        q = (temps[h_index + 1] - temps[h_index]) / (
            heights[h_index + 1] - heights[h_index]
        )
        tempz = (abs(temps[h_index] + q * (z - heights[h_index])) / temps[h_index]) ** (
            -T0 / (H * q)
        )

        temp[iz] = pro * tempz

    return temp


def bdensity_linear(
    field3d: Field3dData, heights: np.ndarray, temps: np.ndarray
) -> np.ndarray:

    temp0 = temps[0]
    dummypres = bpressure_linear(field3d, heights, temps)
    dummytemp = btemp_linear(field3d, heights, temps)

    return dummypres / dummytemp * temp0


def fpressure_linear(
    field3d: Field3dData, heights: np.ndarray, temps: np.ndarray
) -> np.ndarray:

    bp_matrix = np.zeros_like(field3d.dpressure)
    bp_matrix[:, :, :] = bpressure_linear(field3d, heights, temps)

    B0 = field3d.field[
        :, :, 0, 2
    ].max()  # Gauss background magnetic field strength in 10^-4 kg/(s^2A) = 10^-4 T
    PB0 = (B0 * 10**-4) ** 2 / (2 * MU0)  # magnetic pressure b0**2 / 2mu0 in kg/(s^2m)
    BETA0 = P0 / PB0  # Plasma Beta, ration plasma to magnetic pressure

    return (BETA0 / 2.0 * bp_matrix + field3d.dpressure) * (B0 * 10**-4) ** 2.0 / MU0


def fdensity_linear(
    field3d: Field3dData, heights: np.ndarray, temps: np.ndarray
) -> np.ndarray:

    for iheight, height in enumerate(heights):
        if height == field3d.z0:
            T0 = temps[iheight]

    H = KB * T0 / (MBAR * G_SOLAR) / L
    B0 = field3d.field[
        :, :, 0, 2
    ].max()  # Gauss background magnetic field strength in 10^-4 kg/(s^2A) = 10^-4 T
    PB0 = (B0 * 10**-4) ** 2 / (2 * MU0)  # magnetic pressure b0**2 / 2mu0 in kg/(s^2m)
    BETA0 = P0 / PB0  # Plasma Beta, ration plasma to magnetic pressure

    bd_matrix = np.zeros_like(field3d.ddensity)
    bd_matrix[:, :, :] = bdensity_linear(field3d, heights, temps)

    return (
        (BETA0 / (2.0 * H) * T0 / T_PHOTOSPHERE * bd_matrix + field3d.ddensity)
        * (B0 * 10**-4) ** 2.0
        / (MU0 * G_SOLAR * L)
    )


def j3d(field3d: Field3dData) -> np.ndarray:
    """
    Return current density, calucated from magnetic field as
    j = (alpha B + curl(0,0,f(z)Bz))/ mu0.
    """

    j = np.zeros_like(field3d.field)

    j[:, :, :, 2] = field3d.alpha * field3d.field[:, :, :, 2]

    f_matrix = np.zeros_like(field3d.dfield[:, :, :, 0])
    f_matrix[:, :, :] = f(field3d.z, field3d.z0, field3d.deltaz, field3d.a, field3d.b)

    j[:, :, :, 0] = (
        field3d.alpha * field3d.field[:, :, :, 1]
        + f_matrix * field3d.dfield[:, :, :, 0]
    )

    j[:, :, :, 1] = (
        field3d.alpha * field3d.field[:, :, :, 0]
        + f_matrix * field3d.dfield[:, :, :, 1]
    )
    return j / MU0


def lf3d(field3d: Field3dData) -> np.ndarray:
    """
    Calculate Lorentz force.
    """

    j = j3d(field3d)

    lf = np.zeros_like(field3d.field)

    lf[:, :, :, 0] = (
        j[:, :, :, 1] * field3d.field[:, :, :, 2]
        - j[:, :, :, 2] * field3d.field[:, :, :, 1]
    )
    lf[:, :, :, 1] = (
        j[:, :, :, 2] * field3d.field[:, :, :, 0]
        - j[:, :, :, 0] * field3d.field[:, :, :, 2]
    )
    lf[:, :, :, 2] = (
        j[:, :, :, 0] * field3d.field[:, :, :, 1]
        - j[:, :, :, 1] * field3d.field[:, :, :, 0]
    )

    return lf


def fftcoeff_fb(
    data_bz: np.ndarray,
    nf_max: np.int32,
) -> Tuple:

    anm = np.zeros((nf_max, nf_max))
    bnm = np.zeros((nf_max, nf_max))
    cnm = np.zeros((nf_max, nf_max))
    dnm = np.zeros((nf_max, nf_max))

    nresol_y = int(data_bz.shape[0])
    nresol_x = int(data_bz.shape[1])

    signal = np.fft.fftshift(np.fft.fft2(data_bz) / nresol_x / nresol_y)

    for ix in range(0, nresol_x, 2):
        for iy in range(1, nresol_y, 2):
            temp = signal[iy, ix]
            signal[iy, ix] = -temp

    for ix in range(1, nresol_x, 2):
        for iy in range(0, nresol_y, 2):
            temp = signal[iy, ix]
            signal[iy, ix] = -temp

    if nresol_x % 2 == 0:
        centre_x = int(nresol_x / 2)
    else:
        centre_x = int((nresol_x - 1) / 2)
    if nresol_y % 2 == 0:
        centre_y = int(nresol_y / 2)
    else:
        centre_y = int((nresol_y - 1) / 2)

    for ix in range(1, nf_max):
        for iy in range(1, nf_max):
            anm[iy, ix] = (
                -signal[centre_y + iy, centre_x + ix]
                + signal[centre_y + iy, centre_x - ix]
                + signal[centre_y - iy, centre_x + ix]
                - signal[centre_y - iy, centre_x - ix]
            ).real
            bnm[iy, ix] = (
                -signal[centre_y + iy, centre_x + ix]
                + signal[centre_y + iy, centre_x - ix]
                - signal[centre_y - iy, centre_x + ix]
                + signal[centre_y - iy, centre_x - ix]
            ).imag
            cnm[iy, ix] = (
                -signal[centre_y + iy, centre_x + ix]
                + signal[centre_y - iy, centre_x + ix]
                - signal[centre_y + iy, centre_x - ix]
                + signal[centre_y - iy, centre_x - ix]
            ).imag
            dnm[iy, ix] = (
                signal[centre_y + iy, centre_x + ix]
                + signal[centre_y + iy, centre_x - ix]
                + signal[centre_y - iy, centre_x + ix]
                + signal[centre_y - iy, centre_x - ix]
            ).real

    for iy in range(1, nf_max):
        dnm[iy, 0] = (
            signal[centre_y + iy, centre_x + 0] + signal[centre_y - iy, centre_x + 0]
        ).real
        cnm[iy, 0] = (
            -signal[centre_y + iy, centre_x + 0] + signal[centre_y - iy, centre_x + 0]
        ).imag

    for ix in range(1, nf_max):
        dnm[0, ix] = (
            signal[centre_y + 0, centre_x + ix] + signal[centre_y + 0, centre_x - ix]
        ).real
        bnm[0, ix] = (
            -signal[centre_y + 0, centre_x + ix] + signal[centre_y + 0, centre_x - ix]
        ).imag

    return anm, bnm, cnm, dnm


def b3d_fb(
    field: Field2dData,
    a: float,
    b: float,
    alpha: float,
    z0: np.float64,
    deltaz: np.float64,
    asymptotic=True,
    tanh=True,
) -> Tuple:

    nf = int(np.floor(field.nf / 2))

    l = 1.0
    lx = field.nx * field.px * l
    ly = field.ny * field.py * l

    lxn = lx / l
    lyn = ly / l

    kx = np.arange(nf) * 2.0 * np.pi / lxn
    ky = np.arange(nf) * 2.0 * np.pi / lyn
    ones = 0.0 * np.arange(nf) + 1.0

    ky_grid = np.outer(ky, ones)
    kx_grid = np.outer(ones, kx)

    k2 = np.outer(ky**2, ones) + np.outer(ones, kx**2)
    k2[0, 0] = (2.0 * np.pi / lxn) ** 2 + (2.0 * np.pi / lyn) ** 2

    # print(k2.min)

    anm, bnm, cnm, dnm = np.divide(fftcoeff_fb(field.bz, nf), k2)

    if tanh:

        # print("Do tanh")
        p = 0.5 * deltaz * np.sqrt(k2 * (1.0 - a - a * b) - alpha**2)
        q = 0.5 * deltaz * np.sqrt(k2 * (1.0 - a + a * b) - alpha**2)

        phi, dphi = get_phi_dphi(
            field.z,
            q,
            p,
            nf,
            field.nz,
            z0=z0,
            deltaz=deltaz,
            asymptotic=asymptotic,
            tanh=tanh,
        )
    else:
        # print("Do exp")
        aL = a
        kappa = deltaz

        p = 2.0 / kappa * np.sqrt(k2 - alpha**2)
        q = 2.0 / kappa * np.sqrt(k2 * aL)

        phi, dphi = get_phi_dphi(
            field.z,
            q,
            p,
            nf,
            field.nz,
            kappa=kappa,
            asymptotic=asymptotic,
            tanh=tanh,
        )

    bfield = np.zeros((field.ny, field.nx, field.nz, 3))
    dbz = np.zeros((field.ny, field.nx, field.nz, 3))

    sin_x = np.sin(np.outer(kx, field.x - lxn / 2.0))
    sin_y = np.sin(np.outer(ky, field.y - lyn / 2.0))
    cos_x = np.cos(np.outer(kx, field.x - lxn / 2.0))
    cos_y = np.cos(np.outer(ky, field.y - lyn / 2.0))

    # print("b", b.shape)

    for iz in range(0, field.nz):

        coeffs1 = np.multiply(np.multiply(k2, phi[:, :, iz]), anm)
        coeffs2 = np.multiply(np.multiply(k2, phi[:, :, iz]), bnm)
        coeffs3 = np.multiply(np.multiply(k2, phi[:, :, iz]), cnm)
        coeffs4 = np.multiply(np.multiply(k2, phi[:, :, iz]), dnm)

        bfield[:, :, iz, 2] = (
            np.matmul(sin_y.T, np.matmul(coeffs1, sin_x))
            + np.matmul(cos_y.T, np.matmul(coeffs2, sin_x))
            + np.matmul(sin_y.T, np.matmul(coeffs3, cos_x))
            + np.matmul(cos_y.T, np.matmul(coeffs4, cos_x))
        )

        coeffs1 = np.multiply(
            np.multiply(anm, dphi[:, :, iz]), ky_grid
        ) + alpha * np.multiply(np.multiply(dnm, phi[:, :, iz]), kx_grid)
        coeffs2 = -np.multiply(
            np.multiply(dnm, dphi[:, :, iz]), ky_grid
        ) - alpha * np.multiply(np.multiply(anm, phi[:, :, iz]), kx_grid)
        coeffs3 = -np.multiply(
            np.multiply(bnm, dphi[:, :, iz]), ky_grid
        ) + alpha * np.multiply(np.multiply(cnm, phi[:, :, iz]), kx_grid)
        coeffs4 = np.multiply(
            np.multiply(cnm, dphi[:, :, iz]), ky_grid
        ) - alpha * np.multiply(np.multiply(bnm, phi[:, :, iz]), kx_grid)

        bfield[:, :, iz, 0] = (
            np.matmul(cos_y.T, np.matmul(coeffs4, cos_x))
            + np.matmul(sin_y.T, np.matmul(coeffs3, sin_x))
            + np.matmul(cos_y.T, np.matmul(coeffs1, sin_x))
            + np.matmul(sin_y.T, np.matmul(coeffs2, cos_x))
        )

        coeffs1 = -np.multiply(
            np.multiply(cnm, dphi[:, :, iz]), kx_grid
        ) - alpha * np.multiply(np.multiply(bnm, phi[:, :, iz]), ky_grid)
        coeffs2 = np.multiply(
            np.multiply(bnm, dphi[:, :, iz]), kx_grid
        ) + alpha * np.multiply(np.multiply(cnm, phi[:, :, iz]), ky_grid)
        coeffs3 = np.multiply(
            np.multiply(anm, dphi[:, :, iz]), kx_grid
        ) - alpha * np.multiply(np.multiply(dnm, phi[:, :, iz]), ky_grid)
        coeffs4 = -np.multiply(
            np.multiply(dnm, dphi[:, :, iz]), kx_grid
        ) + alpha * np.multiply(np.multiply(anm, phi[:, :, iz]), ky_grid)

        bfield[:, :, iz, 1] = (
            np.matmul(cos_y.T, np.matmul(coeffs2, cos_x))
            + np.matmul(sin_y.T, np.matmul(coeffs1, sin_x))
            + np.matmul(sin_y.T, np.matmul(coeffs3, cos_x))
            + np.matmul(cos_y.T, np.matmul(coeffs4, sin_x))
        )

        coeffs1 = np.multiply(np.multiply(k2, dphi[:, :, iz]), anm)
        coeffs2 = np.multiply(np.multiply(k2, dphi[:, :, iz]), bnm)
        coeffs3 = np.multiply(np.multiply(k2, dphi[:, :, iz]), cnm)
        coeffs4 = np.multiply(np.multiply(k2, dphi[:, :, iz]), dnm)
        dbz[:, :, iz, 2] = (
            np.matmul(sin_y.T, np.matmul(coeffs1, sin_x))
            + np.matmul(cos_y.T, np.matmul(coeffs2, sin_x))
            + np.matmul(sin_y.T, np.matmul(coeffs3, cos_x))
            + np.matmul(cos_y.T, np.matmul(coeffs4, cos_x))
        )

        coeffs1 = np.multiply(np.multiply(np.multiply(k2, phi[:, :, iz]), anm), kx_grid)
        coeffs2 = np.multiply(np.multiply(np.multiply(k2, phi[:, :, iz]), bnm), kx_grid)
        coeffs3 = -np.multiply(
            np.multiply(np.multiply(k2, phi[:, :, iz]), cnm), kx_grid
        )
        coeffs4 = -np.multiply(
            np.multiply(np.multiply(k2, phi[:, :, iz]), dnm), kx_grid
        )

        dbz[:, :, iz, 0] = (
            np.matmul(sin_y.T, np.matmul(coeffs1, cos_x))
            + np.matmul(cos_y.T, np.matmul(coeffs2, cos_x))
            + np.matmul(sin_y.T, np.matmul(coeffs3, sin_x))
            + np.matmul(cos_y.T, np.matmul(coeffs4, sin_x))
        )

        coeffs1 = np.multiply(np.multiply(np.multiply(k2, phi[:, :, iz]), anm), ky_grid)
        coeffs2 = -np.multiply(
            np.multiply(np.multiply(k2, phi[:, :, iz]), bnm), ky_grid
        )
        coeffs3 = +np.multiply(
            np.multiply(np.multiply(k2, phi[:, :, iz]), cnm), ky_grid
        )
        coeffs4 = -np.multiply(
            np.multiply(np.multiply(k2, phi[:, :, iz]), dnm), ky_grid
        )
        dbz[:, :, iz, 1] = (
            np.matmul(cos_y.T, np.matmul(coeffs1, sin_x))
            + np.matmul(sin_y.T, np.matmul(coeffs2, sin_x))
            + np.matmul(cos_y.T, np.matmul(coeffs3, cos_x))
            + np.matmul(sin_y.T, np.matmul(coeffs4, cos_x))
        )

    return bfield, dbz
