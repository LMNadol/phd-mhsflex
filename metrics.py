from __future__ import annotations

import numpy as np
import math

from typing import Tuple

from scipy.stats import pearsonr

from msat.pyvis.fieldline3d import fieldline3d

from mhsflex.field3d import Field3dData

from mhsflex.field3d import fpressure_linear, fdensity_linear


def VecCorr(B: np.ndarray, b: np.ndarray) -> np.float64:
    """
    Returns Vector Correlation metric of B : B_ref and b : B_rec.
    """

    if B.shape != b.shape:
        raise ValueError("Field sizes do not match.")

    return np.sum(np.multiply(B, b)) / np.sqrt(
        np.sum(np.multiply(B, B)) * np.sum(np.multiply(b, b))
    )


def CauSchw(B: np.ndarray, b: np.ndarray) -> np.float64:
    """
    Returns Cauchy Schwarz metric of B : B_ref and b : B_rec.
    """

    if B.shape != b.shape:
        raise ValueError("Field sizes do not match.")

    div = np.multiply(
        np.sqrt(B[:, :, :, 0] ** 2 + B[:, :, :, 1] ** 2 + B[:, :, :, 2] ** 2),
        np.sqrt(b[:, :, :, 0] ** 2 + b[:, :, :, 1] ** 2 + b[:, :, :, 2] ** 2),
    )

    num = (
        B[:, :, :, 0] * b[:, :, :, 0]
        + B[:, :, :, 1] * b[:, :, :, 1]
        + B[:, :, :, 2] * b[:, :, :, 2]
    )

    return np.sum(np.divide(num, div)) / (B.shape[0] * B.shape[1] * B.shape[2])


def CauSchw_z(B: np.ndarray, b: np.ndarray) -> np.float64:
    """
    Returns Cauchy Schwarz metric of B : B_ref and b : B_rec.
    """

    if B.shape != b.shape:
        raise ValueError("Field sizes do not match.")

    div = np.multiply(
        np.sqrt(B[:, :, 0] ** 2 + B[:, :, 1] ** 2 + B[:, :, 2] ** 2),
        np.sqrt(b[:, :, 0] ** 2 + b[:, :, 1] ** 2 + b[:, :, 2] ** 2),
    )

    num = B[:, :, 0] * b[:, :, 0] + B[:, :, 1] * b[:, :, 1] + B[:, :, 2] * b[:, :, 2]

    return np.sum(np.divide(num, div)) / (B.shape[0] * B.shape[1])


def NormErr(B: np.ndarray, b: np.ndarray) -> np.float64:
    """
    Returns Normalised Vector Error metric of B : B_ref and b : B_rec.
    """

    if B.shape != b.shape:
        raise ValueError("Field sizes do not match.")

    # return np.sum(np.sqrt(np.multiply(B - b, B - b))) / np.sum(
    #     np.sqrt(np.multiply(B, B))
    # )

    diff = B - b

    temp = np.sqrt(
        diff[:, :, :, 0] ** 2 + diff[:, :, :, 1] ** 2 + diff[:, :, :, 2] ** 2
    )

    num = np.sum(temp)

    temp2 = np.sqrt(B[:, :, :, 0] ** 2 + B[:, :, :, 1] ** 2 + B[:, :, :, 2] ** 2)

    div = np.sum(temp2)

    return np.divide(num, div)


def NormErr_z(B: np.ndarray, b: np.ndarray) -> np.float64:
    """
    Returns Normalised Vector Error metric of B : B_ref and b : B_rec.
    """

    if B.shape != b.shape:
        raise ValueError("Field sizes do not match.")

    # return np.sum(np.sqrt(np.multiply(B - b, B - b))) / np.sum(
    #     np.sqrt(np.multiply(B, B))
    # )

    diff = B - b

    temp = np.sqrt(diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2 + diff[:, :, 2] ** 2)

    num = np.sum(temp)

    temp2 = np.sqrt(B[:, :, 0] ** 2 + B[:, :, 1] ** 2 + B[:, :, 2] ** 2)

    div = np.sum(temp2)

    return np.divide(num, div)


def MeanErr(B: np.ndarray, b: np.ndarray) -> np.float64:
    """
    Returns Mean Vector Error metric of B : B_ref and b : B_rec.
    """

    if B.shape != b.shape:
        raise ValueError("Field sizes do not match.")

    div = np.sqrt(B[:, :, :, 0] ** 2 + B[:, :, :, 1] ** 2 + B[:, :, :, 2] ** 2)

    temp = B - b
    num = np.sqrt(temp[:, :, :, 0] ** 2 + temp[:, :, :, 1] ** 2 + temp[:, :, :, 2] ** 2)

    return np.sum(np.divide(num, div)) / (B.shape[0] * B.shape[1] * B.shape[2])


def MeanErr_z(B: np.ndarray, b: np.ndarray) -> np.float64:
    """
    Returns Mean Vector Error metric of B : B_ref and b : B_rec.
    """

    if B.shape != b.shape:
        raise ValueError("Field sizes do not match.")

    div = np.sqrt(B[:, :, 0] ** 2 + B[:, :, 1] ** 2 + B[:, :, 2] ** 2)

    temp = B - b
    num = np.sqrt(temp[:, :, 0] ** 2 + temp[:, :, 1] ** 2 + temp[:, :, 2] ** 2)

    return np.sum(np.divide(num, div)) / (B.shape[0] * B.shape[1])


def MagEnergy(B: np.ndarray, b: np.ndarray) -> np.float64:
    """
    Returns Magnetic Energy metric of B : B_ref and b : B_rec.
    """
    if B.shape != b.shape:
        raise ValueError("Field sizes do not match.")

    return np.sum(np.multiply(b, b)) / np.sum(np.multiply(B, B))


def field_div_metric(
    dataB: Field3dData,
    datab: Field3dData,
) -> np.float64:
    """
    Returns Field Line Divergence metric of B : B_ref and b : B_rec.
    xmin, ymin, ymax as when using plot_fieldline_grid.

    Tracing field lines from footpoints on a grid on the bottom boundary in both reference model
    and reconstruction. If both field lines end again on the bottom boundary, a score is assigned:
    the distance between the two endpoints divided by the length of the field line in the reference model.
    Then the metric score can be obtained by the fraction of the initial footpoints in which these scores
    are less than 10%.
    """

    h1 = 1.0 / 100.0  # Initial step length for fieldline3D
    eps = 1.0e-8
    # Tolerance to which we require point on field line known for fieldline3D
    hmin = 0.0  # Minimum step length for fieldline3D
    hmax = 1.0  # Maximum step length for fieldline3D

    xmin, xmax, ymin, ymax, zmin, zmax = (
        datab.x[0],
        datab.x[-1],
        datab.y[0],
        datab.y[-1],
        datab.z[0],
        datab.z[-1],
    )

    if datab.field.shape != dataB.field.shape:
        raise ValueError("Fields not of same size.")

    x_big = np.arange(2.0 * datab.nx) * 2.0 * xmax / (2.0 * datab.nx - 1) - xmax
    y_big = np.arange(2.0 * datab.ny) * 2.0 * ymax / (2.0 * datab.ny - 1) - ymax
    z_arr = np.arange(datab.nz) * (zmax - zmin) / (datab.nz - 1) + zmin

    x_0 = 1.0 * 10**-8
    y_0 = 1.0 * 10**-8
    dx = xmax / (datab.nx / 5.0)
    dy = ymax / (datab.ny / 5.0)

    nlinesmaxx = math.floor(xmax / dx)
    nlinesmaxy = math.floor(ymax / dy)

    # Limit fieldline plot to original data size (rather than Seehafer size)
    boxedges = np.zeros((2, 3))

    # Y boundaries must come first, X second due to switched order explained above
    boxedges[0, 0] = ymin
    boxedges[1, 0] = ymax
    boxedges[0, 1] = xmin
    boxedges[1, 1] = xmax
    boxedges[0, 2] = zmin
    boxedges[1, 2] = zmax

    # set fieldline3D initial stepsize
    h1_ref = h1
    h1_rec = h1

    # counter for number of footpoints that have an error in the endpoints smaller than 10 percent of field line length in reference model
    count = 0

    # counter for number of field lines that are closed within box
    count_closed = 0

    # counter for all field lines
    count_all = 0

    for ilinesx in range(0, nlinesmaxx):
        for ilinesy in range(0, nlinesmaxy):

            count_all = count_all + 1

            # Footpoint
            x_start = x_0 + dx * ilinesx
            y_start = y_0 + dy * ilinesy
            ystart = [y_start, x_start, 0.0]

            # decide direction of fieldline for reference model
            if dataB.field[int(y_start), int(x_start), 0, 2] < 0.0:
                h1_ref = -h1_ref
            # decide direction of fieldline for reconstruction model
            if datab.field[int(y_start), int(x_start), 0, 2] < 0.0:
                h1_rec = -h1_rec

            # Fieldline3D expects startpt, BField, Row values, Column values so we need to give Y first, then X

            # Get fieldline coordinates for reference model
            fieldline_ref = fieldline3d(
                ystart,
                dataB.field,
                y_big,
                x_big,
                z_arr,
                h1_ref,
                hmin,
                hmax,
                eps,
                oneway=False,
                boxedge=boxedges,
                gridcoord=False,
                coordsystem="cartesian",
            )

            # Get fieldline coordinates for reconstruction model
            fieldline_rec = fieldline3d(
                ystart,
                datab.field,
                y_big,
                x_big,
                z_arr,
                h1_rec,
                hmin,
                hmax,
                eps,
                oneway=False,
                boxedge=boxedges,
                gridcoord=False,
                coordsystem="cartesian",
            )

            len_ref = len(fieldline_ref)
            len_rec = len(fieldline_rec)

            valid_fieldline = True

            # Check if field lines end on bottom boundary
            if not np.isclose(fieldline_ref[len_ref - 1, 2], 0.0):
                valid_fieldline = False
            if not np.isclose(fieldline_rec[len_rec - 1, 2], 0.0):
                valid_fieldline = False

            if not (0.0 <= fieldline_ref[len_ref - 1, 1] <= xmax):
                valid_fieldline = False
            if not (0.0 <= fieldline_rec[len_rec - 1, 1] <= xmax):
                valid_fieldline = False
            if not (0.0 <= fieldline_ref[len_ref - 1, 0] <= ymax):
                valid_fieldline = False
            if not (0.0 <= fieldline_rec[len_rec - 1, 0] <= ymax):
                valid_fieldline = False

            if valid_fieldline:

                # add to counter if field line is closed field line within box
                count_closed = count_closed + 1

                # calculate distance between the endpoints of reference field line and of reconstructed field line
                num = np.sqrt(
                    (fieldline_rec[len_rec - 1, 1] - fieldline_ref[len_ref - 1, 1])
                    ** 2.0
                    + (fieldline_rec[len_rec - 1, 0] - fieldline_ref[len_ref - 1, 0])
                    ** 2.0
                    + (fieldline_rec[len_rec - 1, 2] - fieldline_ref[len_ref - 1, 2])
                    ** 2.0
                )

                # calculate length of reference field line
                div = 0.0
                for i in range(0, len_ref - 1):
                    div = div + np.sqrt(
                        (fieldline_ref[i, 1] - fieldline_ref[i + 1, 1]) ** 2.0
                        + (fieldline_ref[i, 0] - fieldline_ref[i + 1, 0]) ** 2.0
                        + (fieldline_ref[i, 2] - fieldline_ref[i + 1, 2]) ** 2.0
                    )

                # divide distance between endpoints by length of reference field line
                # gives error between endpoints as percentage of length of reference field line

                temp = num / div

                # add to counter if error is smaller than 10 percent
                if temp <= 0.1:
                    count = count + 1

    print(count_all, count_closed, count)
    # return number of footpoints with error smaller than 10 percent as percentage of all footpoints, of all closed fieldlines
    return np.float64(count / count_closed)


def field_div_metric_fb(
    dataB: Field3dData,
    datab: Field3dData,
) -> np.float64:
    """
    Returns Field Line Divergence metric of B : B_ref and b : B_rec.
    xmin, ymin, ymax as when using plot_fieldline_grid.

    Tracing field lines from footpoints on a grid on the bottom boundary in both reference model
    and reconstruction. If both field lines end again on the bottom boundary, a score is assigned:
    the distance between the two endpoints divided by the length of the field line in the reference model.
    Then the metric score can be obtained by the fraction of the initial footpoints in which these scores
    are less than 10%.
    """

    h1 = 1.0 / 100.0  # Initial step length for fieldline3D
    eps = 1.0e-8
    # Tolerance to which we require point on field line known for fieldline3D
    hmin = 0.0  # Minimum step length for fieldline3D
    hmax = 1.0  # Maximum step length for fieldline3D

    xmin, xmax, ymin, ymax, zmin, zmax = (
        datab.x[0],
        datab.x[-1],
        datab.y[0],
        datab.y[-1],
        datab.z[0],
        datab.z[-1],
    )

    if datab.field.shape != dataB.field.shape:
        raise ValueError("Fields not of same size.")

    z_arr = np.arange(datab.nz) * (zmax - zmin) / (datab.nz - 1) + zmin

    x_0 = 1.0 * 10**-8
    y_0 = 1.0 * 10**-8
    dx = xmax / (datab.nx / 5.0)
    dy = ymax / (datab.ny / 5.0)

    nlinesmaxx = math.floor(xmax / dx)
    nlinesmaxy = math.floor(ymax / dy)

    # Limit fieldline plot to original data size (rather than Seehafer size)
    boxedges = np.zeros((2, 3))

    # Y boundaries must come first, X second due to switched order explained above
    boxedges[0, 0] = ymin
    boxedges[1, 0] = ymax
    boxedges[0, 1] = xmin
    boxedges[1, 1] = xmax
    boxedges[0, 2] = zmin
    boxedges[1, 2] = zmax

    # set fieldline3D initial stepsize
    h1_ref = h1
    h1_rec = h1

    # counter for number of footpoints that have an error in the endpoints smaller than 10 percent of field line length in reference model
    count = 0

    # counter for number of field lines that are closed within box
    count_closed = 0

    # counter for all field lines
    count_all = 0

    for ilinesx in range(0, nlinesmaxx):
        for ilinesy in range(0, nlinesmaxy):

            count_all = count_all + 1

            # Footpoint
            x_start = x_0 + dx * ilinesx
            y_start = y_0 + dy * ilinesy
            ystart = [y_start, x_start, 0.0]

            # decide direction of fieldline for reference model
            if dataB.field[int(y_start), int(x_start), 0, 2] < 0.0:
                h1_ref = -h1_ref
            # decide direction of fieldline for reconstruction model
            if datab.field[int(y_start), int(x_start), 0, 2] < 0.0:
                h1_rec = -h1_rec

            # Fieldline3D expects startpt, BField, Row values, Column values so we need to give Y first, then X

            # Get fieldline coordinates for reference model
            fieldline_ref = fieldline3d(
                ystart,
                dataB.field,
                dataB.y,
                dataB.x,
                z_arr,
                h1_ref,
                hmin,
                hmax,
                eps,
                oneway=False,
                boxedge=boxedges,
                gridcoord=False,
                coordsystem="cartesian",
            )

            # Get fieldline coordinates for reconstruction model
            fieldline_rec = fieldline3d(
                ystart,
                datab.field,
                datab.y,
                datab.x,
                z_arr,
                h1_rec,
                hmin,
                hmax,
                eps,
                oneway=False,
                boxedge=boxedges,
                gridcoord=False,
                coordsystem="cartesian",
            )

            len_ref = len(fieldline_ref)
            len_rec = len(fieldline_rec)

            valid_fieldline = True

            # Check if field lines end on bottom boundary
            if not np.isclose(fieldline_ref[len_ref - 1, 2], 0.0):
                valid_fieldline = False
            if not np.isclose(fieldline_rec[len_rec - 1, 2], 0.0):
                valid_fieldline = False

            if not (0.0 <= fieldline_ref[len_ref - 1, 1] <= xmax):
                valid_fieldline = False
            if not (0.0 <= fieldline_rec[len_rec - 1, 1] <= xmax):
                valid_fieldline = False
            if not (0.0 <= fieldline_ref[len_ref - 1, 0] <= ymax):
                valid_fieldline = False
            if not (0.0 <= fieldline_rec[len_rec - 1, 0] <= ymax):
                valid_fieldline = False

            if valid_fieldline:

                # add to counter if field line is closed field line within box
                count_closed = count_closed + 1

                # calculate distance between the endpoints of reference field line and of reconstructed field line
                num = np.sqrt(
                    (fieldline_rec[len_rec - 1, 1] - fieldline_ref[len_ref - 1, 1])
                    ** 2.0
                    + (fieldline_rec[len_rec - 1, 0] - fieldline_ref[len_ref - 1, 0])
                    ** 2.0
                    + (fieldline_rec[len_rec - 1, 2] - fieldline_ref[len_ref - 1, 2])
                    ** 2.0
                )

                # calculate length of reference field line
                div = 0.0
                for i in range(0, len_ref - 1):
                    div = div + np.sqrt(
                        (fieldline_ref[i, 1] - fieldline_ref[i + 1, 1]) ** 2.0
                        + (fieldline_ref[i, 0] - fieldline_ref[i + 1, 0]) ** 2.0
                        + (fieldline_ref[i, 2] - fieldline_ref[i + 1, 2]) ** 2.0
                    )

                # divide distance between endpoints by length of reference field line
                # gives error between endpoints as percentage of length of reference field line

                temp = num / div

                # add to counter if error is smaller than 10 percent
                if temp <= 0.1:
                    count = count + 1

    print(count_all, count_closed, count)
    # return number of footpoints with error smaller than 10 percent as percentage of all footpoints, of all closed fieldlines
    return np.float64(count / count_closed)


def pearson_corr_coeff(
    dataB: Field3dData,
    datab: Field3dData,
) -> Tuple:
    """
    B : B_ref and b : B_rec.
    Returns line-of-sigth integration (using composite trapezoidal rule) with respect to
    the z-direction for pressure and density for two given magnetic field models
    (reference model and reconstructed model) in the order:
        (1) Pressure surface data for reference field
        (2) Density surface data for reference field
        (3) Pressure surface data for reconstructed field
        (4) Density surface data for reconstructed field
    Also, prints the Pearson Correlation Coefficient (reference and actual) for the line-of-sight integration
    for both pressure and density between the reference and the recreated model.
    """

    zmin, zmax = datab.z[0], datab.z[-1]

    if datab.field.shape != dataB.field.shape:
        raise ValueError("Fields not of same size.")

    z_arr = np.arange(datab.nz) * (zmax - zmin) / (datab.nz - 1) + zmin

    pres_surface_ref = np.zeros((datab.ny, datab.nx))
    den_surface_ref = np.zeros((datab.ny, datab.nx))
    pres_surface_rec = np.zeros((datab.ny, datab.nx))
    den_surface_rec = np.zeros((datab.ny, datab.nx))

    for ix in range(datab.nx):
        for iy in range(datab.ny):
            pres_surface_ref[iy, ix] = np.trapz(dataB.fpressure[iy, ix, :], z_arr)
            den_surface_ref[iy, ix] = np.trapz(dataB.fdensity[iy, ix, :], z_arr)
            pres_surface_rec[iy, ix] = np.trapz(datab.fpressure[iy, ix, :], z_arr)
            den_surface_rec[iy, ix] = np.trapz(datab.fdensity[iy, ix, :], z_arr)

    print(
        "Pearson Correlation reference value for pressure ",
        pearsonr(pres_surface_ref.flatten(), pres_surface_ref.flatten()),
    )
    print(
        "Pearson Correlation reference value for density ",
        pearsonr(den_surface_ref.flatten(), den_surface_ref.flatten()),
    )
    print(
        "Pearson Correlation actual value for pressure ",
        pearsonr(pres_surface_rec.flatten(), pres_surface_ref.flatten()),
    )
    print(
        "Pearson Correlation actual value for density ",
        pearsonr(den_surface_rec.flatten(), den_surface_ref.flatten()),
    )

    return pres_surface_ref, den_surface_ref, pres_surface_rec, den_surface_rec


def field_div_metric_issi(
    B_ref: np.ndarray,
    datab: Field3dData,
) -> np.float64:
    """
    Returns Field Line Divergence metric of B : B_ref and b : B_rec.
    xmin, ymin, ymax as when using plot_fieldline_grid.

    Tracing field lines from footpoints on a grid on the bottom boundary in both reference model
    and reconstruction. If both field lines end again on the bottom boundary, a score is assigned:
    the distance between the two endpoints divided by the length of the field line in the reference model.
    Then the metric score can be obtained by the fraction of the initial footpoints in which these scores
    are less than 10%.
    """

    h1 = 1.0 / 100.0  # Initial step length for fieldline3D
    eps = 1.0e-8
    # Tolerance to which we require point on field line known for fieldline3D
    hmin = 0.0  # Minimum step length for fieldline3D
    hmax = 1.0  # Maximum step length for fieldline3D

    xmin, xmax, ymin, ymax, zmin, zmax = (
        datab.x[0],
        datab.x[-1],
        datab.y[0],
        datab.y[-1],
        datab.z[0],
        datab.z[-1],
    )

    if datab.field.shape != B_ref.shape:
        raise ValueError("Fields not of same size.")

    z_arr = np.arange(datab.nz) * (zmax - zmin) / (datab.nz - 1) + zmin

    x_0 = 1.0 * 10**-8
    y_0 = 1.0 * 10**-8
    dx = xmax / (datab.nx / 5.0)
    dy = ymax / (datab.ny / 5.0)

    nlinesmaxx = math.floor(xmax / dx)
    nlinesmaxy = math.floor(ymax / dy)

    # Limit fieldline plot to original data size (rather than Seehafer size)
    boxedges = np.zeros((2, 3))

    # Y boundaries must come first, X second due to switched order explained above
    boxedges[0, 0] = ymin
    boxedges[1, 0] = ymax
    boxedges[0, 1] = xmin
    boxedges[1, 1] = xmax
    boxedges[0, 2] = zmin
    boxedges[1, 2] = zmax

    # set fieldline3D initial stepsize
    h1_ref = h1
    h1_rec = h1

    # counter for number of footpoints that have an error in the endpoints smaller than 10 percent of field line length in reference model
    count = 0

    # counter for number of field lines that are closed within box
    count_closed = 0

    # counter for all field lines
    count_all = 0

    for ilinesx in range(0, nlinesmaxx):
        for ilinesy in range(0, nlinesmaxy):

            count_all = count_all + 1

            # Footpoint
            x_start = x_0 + dx * ilinesx
            y_start = y_0 + dy * ilinesy
            ystart = [y_start, x_start, 0.0]

            # decide direction of fieldline for reference model
            if B_ref[int(y_start), int(x_start), 0, 2] < 0.0:
                h1_ref = -h1_ref
            # decide direction of fieldline for reconstruction model
            if datab.field[int(y_start), int(x_start), 0, 2] < 0.0:
                h1_rec = -h1_rec

            # Fieldline3D expects startpt, BField, Row values, Column values so we need to give Y first, then X

            # Get fieldline coordinates for reference model
            fieldline_ref = fieldline3d(
                ystart,
                B_ref,
                datab.y,
                datab.x,
                z_arr,
                h1_ref,
                hmin,
                hmax,
                eps,
                oneway=False,
                boxedge=boxedges,
                gridcoord=False,
                coordsystem="cartesian",
            )

            # Get fieldline coordinates for reconstruction model
            fieldline_rec = fieldline3d(
                ystart,
                datab.field,
                datab.y,
                datab.x,
                z_arr,
                h1_rec,
                hmin,
                hmax,
                eps,
                oneway=False,
                boxedge=boxedges,
                gridcoord=False,
                coordsystem="cartesian",
            )

            len_ref = len(fieldline_ref)
            len_rec = len(fieldline_rec)

            valid_fieldline = True

            # Check if field lines end on bottom boundary
            if not np.isclose(fieldline_ref[len_ref - 1, 2], 0.0):
                valid_fieldline = False
            if not np.isclose(fieldline_rec[len_rec - 1, 2], 0.0):
                valid_fieldline = False

            if not (0.0 <= fieldline_ref[len_ref - 1, 1] <= xmax):
                valid_fieldline = False
            if not (0.0 <= fieldline_rec[len_rec - 1, 1] <= xmax):
                valid_fieldline = False
            if not (0.0 <= fieldline_ref[len_ref - 1, 0] <= ymax):
                valid_fieldline = False
            if not (0.0 <= fieldline_rec[len_rec - 1, 0] <= ymax):
                valid_fieldline = False

            if valid_fieldline:

                # add to counter if field line is closed field line within box
                count_closed = count_closed + 1

                # calculate distance between the endpoints of reference field line and of reconstructed field line
                num = np.sqrt(
                    (fieldline_rec[len_rec - 1, 1] - fieldline_ref[len_ref - 1, 1])
                    ** 2.0
                    + (fieldline_rec[len_rec - 1, 0] - fieldline_ref[len_ref - 1, 0])
                    ** 2.0
                    + (fieldline_rec[len_rec - 1, 2] - fieldline_ref[len_ref - 1, 2])
                    ** 2.0
                )

                # calculate length of reference field line
                div = 0.0
                for i in range(0, len_ref - 1):
                    div = div + np.sqrt(
                        (fieldline_ref[i, 1] - fieldline_ref[i + 1, 1]) ** 2.0
                        + (fieldline_ref[i, 0] - fieldline_ref[i + 1, 0]) ** 2.0
                        + (fieldline_ref[i, 2] - fieldline_ref[i + 1, 2]) ** 2.0
                    )

                # divide distance between endpoints by length of reference field line
                # gives error between endpoints as percentage of length of reference field line

                temp = num / div

                # add to counter if error is smaller than 10 percent
                if temp <= 0.1:
                    count = count + 1

    print(count_all, count_closed, count)

    # return number of footpoints with error smaller than 10 percent as percentage of all footpoints, of all closed fieldlines
    return np.float64(count / count_closed)


def field_div_metric_issi2(
    B_ref: np.ndarray,
    datab: Field3dData,
) -> np.float64:
    """
    Returns Field Line Divergence metric of B : B_ref and b : B_rec.
    xmin, ymin, ymax as when using plot_fieldline_grid.

    Tracing field lines from footpoints on a grid on the bottom boundary in both reference model
    and reconstruction. If both field lines end again on the bottom boundary, a score is assigned:
    the distance between the two endpoints divided by the length of the field line in the reference model.
    Then the metric score can be obtained by the fraction of the initial footpoints in which these scores
    are less than 10%.
    """

    h1 = 1.0 / 100.0  # Initial step length for fieldline3D
    eps = 1.0e-8
    # Tolerance to which we require point on field line known for fieldline3D
    hmin = 0.0  # Minimum step length for fieldline3D
    hmax = 1.0  # Maximum step length for fieldline3D

    xmin, xmax, ymin, ymax, zmin, zmax = (
        datab.x[280],
        datab.x[490],
        datab.y[60],
        datab.y[240],
        datab.z[0],
        datab.z[469],
    )

    if datab.field[60:240, 280:490, 0:469, :].shape != B_ref.shape:
        raise ValueError("Fields not of same size.")

    z_arr = np.arange(datab.nz) * (zmax - zmin) / (datab.nz - 1) + zmin

    # Limit fieldline plot to original data size (rather than Seehafer size)
    boxedges = np.zeros((2, 3))

    # Y boundaries must come first, X second due to switched order explained above
    boxedges[0, 0] = ymin
    boxedges[1, 0] = ymax
    boxedges[0, 1] = xmin
    boxedges[1, 1] = xmax
    boxedges[0, 2] = zmin
    boxedges[1, 2] = zmax

    # set fieldline3D initial stepsize
    h1_ref = h1
    h1_rec = h1

    # counter for number of footpoints that have an error in the endpoints smaller than 10 percent of field line length in reference model
    count = 0

    # counter for number of field lines that are closed within box
    count_closed = 0

    # counter for all field lines
    count_all = 0

    for ix in datab.x[330:440]:
        for iy in datab.y[145:210]:

            count_all = count_all + 1

            # Footpoint
            x_start = ix
            y_start = iy
            ystart = [y_start, x_start, 0.0]

            # decide direction of fieldline for reference model
            if B_ref[int(y_start), int(x_start), 0, 2] < 0.0:
                h1_ref = -h1_ref
            # decide direction of fieldline for reconstruction model
            if datab.field[int(y_start), int(x_start), 0, 2] < 0.0:
                h1_rec = -h1_rec

            # Fieldline3D expects startpt, BField, Row values, Column values so we need to give Y first, then X

            # Get fieldline coordinates for reference model
            fieldline_ref = fieldline3d(
                ystart,
                B_ref,
                datab.y,
                datab.x,
                z_arr,
                h1_ref,
                hmin,
                hmax,
                eps,
                oneway=False,
                boxedge=boxedges,
                gridcoord=False,
                coordsystem="cartesian",
            )

            # Get fieldline coordinates for reconstruction model
            fieldline_rec = fieldline3d(
                ystart,
                datab.field,
                datab.y,
                datab.x,
                z_arr,
                h1_rec,
                hmin,
                hmax,
                eps,
                oneway=False,
                boxedge=boxedges,
                gridcoord=False,
                coordsystem="cartesian",
            )

            len_ref = len(fieldline_ref)
            len_rec = len(fieldline_rec)

            valid_fieldline = True

            # Check if field lines end on bottom boundary
            if not np.isclose(fieldline_ref[len_ref - 1, 2], 0.0):
                valid_fieldline = False
            if not np.isclose(fieldline_rec[len_rec - 1, 2], 0.0):
                valid_fieldline = False

            if not (0.0 <= fieldline_ref[len_ref - 1, 1] <= xmax):
                valid_fieldline = False
            if not (0.0 <= fieldline_rec[len_rec - 1, 1] <= xmax):
                valid_fieldline = False
            if not (0.0 <= fieldline_ref[len_ref - 1, 0] <= ymax):
                valid_fieldline = False
            if not (0.0 <= fieldline_rec[len_rec - 1, 0] <= ymax):
                valid_fieldline = False

            if valid_fieldline:

                # add to counter if field line is closed field line within box
                count_closed = count_closed + 1

                # calculate distance between the endpoints of reference field line and of reconstructed field line
                num = np.sqrt(
                    (fieldline_rec[len_rec - 1, 1] - fieldline_ref[len_ref - 1, 1])
                    ** 2.0
                    + (fieldline_rec[len_rec - 1, 0] - fieldline_ref[len_ref - 1, 0])
                    ** 2.0
                    + (fieldline_rec[len_rec - 1, 2] - fieldline_ref[len_ref - 1, 2])
                    ** 2.0
                )

                # calculate length of reference field line
                div = 0.0
                for i in range(0, len_ref - 1):
                    div = div + np.sqrt(
                        (fieldline_ref[i, 1] - fieldline_ref[i + 1, 1]) ** 2.0
                        + (fieldline_ref[i, 0] - fieldline_ref[i + 1, 0]) ** 2.0
                        + (fieldline_ref[i, 2] - fieldline_ref[i + 1, 2]) ** 2.0
                    )

                # divide distance between endpoints by length of reference field line
                # gives error between endpoints as percentage of length of reference field line

                temp = num / div

                # add to counter if error is smaller than 10 percent
                if temp <= 0.1:
                    count = count + 1

    print(count_all, count_closed, count)

    # return number of footpoints with error smaller than 10 percent as percentage of all footpoints, of all closed fieldlines
    return np.float64(count / count_closed)


def pearson_corr_coeff_issi(
    fpres_3d_ref: np.ndarray,
    fden_3d_ref: np.ndarray,
    datab: Field3dData,
    heights: np.ndarray,
    temps: np.ndarray,
) -> None:
    """
    B : B_ref and b : B_rec.
    Returns line-of-sigth integration (using composite trapezoidal rule) with respect to
    the z-direction for pressure and density for two given magnetic field models
    (reference model and reconstructed model) in the order:
        (1) Pressure surface data for reference field
        (2) Density surface data for reference field
        (3) Pressure surface data for reconstructed field
        (4) Density surface data for reconstructed field
    Also, prints the Pearson Correlation Coefficient (reference and actual) for the line-of-sight integration
    for both pressure and density between the reference and the recreated model.
    """

    zmin, zmax = datab.z[0], datab.z[-1]
    z_arr = np.arange(datab.nz) * (zmax - zmin) / (datab.nz - 1) + zmin

    pres_surface_ref = np.zeros((datab.ny, datab.nx))
    den_surface_ref = np.zeros((datab.ny, datab.nx))
    pres_surface_rec = np.zeros((datab.ny, datab.nx))
    den_surface_rec = np.zeros((datab.ny, datab.nx))

    for ix in range(datab.nx):
        for iy in range(datab.ny):
            pres_surface_ref[iy, ix] = np.trapz(fpres_3d_ref[iy, ix, :], z_arr)
            den_surface_ref[iy, ix] = np.trapz(fden_3d_ref[iy, ix, :], z_arr)
            pres_surface_rec[iy, ix] = np.trapz(
                fpressure_linear(datab, heights, temps)[iy, ix, :], z_arr
            )
            den_surface_rec[iy, ix] = np.trapz(
                fdensity_linear(datab, heights, temps)[iy, ix, :], z_arr
            )

    print(
        "Pearson Correlation reference value for pressure ",
        pearsonr(pres_surface_ref.flatten(), pres_surface_ref.flatten()),
    )
    print(
        "Pearson Correlation reference value for density ",
        pearsonr(den_surface_ref.flatten(), den_surface_ref.flatten()),
    )
    print(
        "Pearson Correlation actual value for pressure ",
        pearsonr(pres_surface_rec.flatten(), pres_surface_ref.flatten()),
    )
    print(
        "Pearson Correlation actual value for density ",
        pearsonr(den_surface_rec.flatten(), den_surface_ref.flatten()),
    )


def pearson_corr_coeff_issi_2(
    fpres_3d_ref: np.ndarray,
    fden_3d_ref: np.ndarray,
    datab: Field3dData,
    fpres_3d_rec: np.ndarray,
    fden_3d_rec: np.ndarray,
    heights: np.ndarray,
    temps: np.ndarray,
) -> None:
    """
    B : B_ref and b : B_rec.
    Returns line-of-sigth integration (using composite trapezoidal rule) with respect to
    the z-direction for pressure and density for two given magnetic field models
    (reference model and reconstructed model) in the order:
        (1) Pressure surface data for reference field
        (2) Density surface data for reference field
        (3) Pressure surface data for reconstructed field
        (4) Density surface data for reconstructed field
    Also, prints the Pearson Correlation Coefficient (reference and actual) for the line-of-sight integration
    for both pressure and density between the reference and the recreated model.
    """

    z_arr = datab.z[0:469]

    pres_surface_ref = np.zeros((datab.ny, datab.nx))
    den_surface_ref = np.zeros((datab.ny, datab.nx))
    pres_surface_rec = np.zeros((datab.ny, datab.nx))
    den_surface_rec = np.zeros((datab.ny, datab.nx))

    for ix in range(fpres_3d_ref.shape[1]):
        for iy in range(fpres_3d_ref.shape[0]):
            pres_surface_ref[iy, ix] = np.trapz(fpres_3d_ref[iy, ix, :], z_arr)
            den_surface_ref[iy, ix] = np.trapz(fden_3d_ref[iy, ix, :], z_arr)
            pres_surface_rec[iy, ix] = np.trapz(fpres_3d_rec[iy, ix, :], z_arr)
            den_surface_rec[iy, ix] = np.trapz(fden_3d_rec[iy, ix, :], z_arr)

    print(
        "Pearson Correlation reference value for pressure ",
        pearsonr(pres_surface_ref.flatten(), pres_surface_ref.flatten()),
    )
    print(
        "Pearson Correlation reference value for density ",
        pearsonr(den_surface_ref.flatten(), den_surface_ref.flatten()),
    )
    print(
        "Pearson Correlation actual value for pressure ",
        pearsonr(pres_surface_rec.flatten(), pres_surface_ref.flatten()),
    )
    print(
        "Pearson Correlation actual value for density ",
        pearsonr(den_surface_rec.flatten(), den_surface_ref.flatten()),
    )


def compare_field3d(data_ref: Field3dData, data_rec: Field3dData) -> None:

    nx = data_ref.nx
    ny = data_ref.ny

    b_ref = data_ref.field[nx : 2 * nx, ny : 2 * ny, :, :]
    b_rec = data_rec.field[nx : 2 * nx, ny : 2 * ny, :, :]

    VC = VecCorr(b_ref, b_rec)
    CS = CauSchw(b_ref, b_rec)
    NE = NormErr(b_ref, b_rec)
    ME = MeanErr(b_ref, b_rec)
    MAGE = MagEnergy(b_ref, b_rec)

    print("MAGNETIC FIELD VECTOR METRICS")
    print(
        "-----------------------------------------------------------------------------------------------------------"
    )
    print(
        "Vector correlation metric: ",
        VC,
        "(Reference value: ",
        VecCorr(b_ref, b_ref),
        ")",
    )
    print(
        "Cauchy-Schwarz metric: ",
        CS,
        "(Reference value: ",
        CauSchw(b_ref, b_ref),
        ")",
    )
    print(
        "Normalised vector error metric: ",
        NE,
        "(Reference value: ",
        NormErr(b_ref, b_ref),
        ")",
    )
    print(
        "Mean vector error metric: ",
        ME,
        "(Reference value: ",
        MeanErr(b_ref, b_ref),
        ")",
    )
    print(
        "Magnetic energy metric: ",
        MAGE,
        "(Reference value: ",
        MagEnergy(b_ref, b_ref),
        ")",
    )
    print(
        "-----------------------------------------------------------------------------------------------------------"
    )
    print("FIELD LINE DIVERGENCE METRIC")
    print(
        "-----------------------------------------------------------------------------------------------------------"
    )

    ratioclosed = field_div_metric(data_ref, data_rec)

    print(
        "Percentage of footpoints with error smaller than 10 percent of all closed fieldlines: ",
        ratioclosed,
    )
    print(
        "-----------------------------------------------------------------------------------------------------------"
    )
    print("PLASMA PARAMETER PEARSON CORRELATION COEFFICIENT METRICS")
    print(
        "-----------------------------------------------------------------------------------------------------------"
    )

    pres_ref, den_ref, pres_rec, den_rec = pearson_corr_coeff(data_ref, data_rec)
