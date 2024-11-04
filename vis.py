from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import math

from scipy.ndimage import maximum_filter, minimum_filter, label, find_objects

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rc, colors

from mhsflex.field3d import Field3dData

from msat.pyvis.fieldline3d import fieldline3d

rc("font", **{"family": "serif", "serif": ["Times"]})
rc("text", usetex=True)

cmap = colors.LinearSegmentedColormap.from_list(
    "cmap",
    (
        (0.000, (0.000, 0.000, 0.000)),
        (0.500, (0.659, 0.659, 0.659)),
        (1.000, (1.000, 1.000, 1.000)),
    ),
)
c1 = (1.000, 0.224, 0.376)
c2 = (0.420, 0.502, 1.000)
c4 = (1.000, 0.224, 0.376)
c5 = (1.000, 0.412, 0.816)
norm = colors.SymLogNorm(50, vmin=-7.5e2, vmax=7.5e2)
c2 = (0.420, 0.502, 1.000)
c3 = "black"
c4 = (1.000, 0.224, 0.376)
c5 = (0.784, 0.231, 0.576)
c7 = (0.992, 0.251, 0.733)
c8 = (0.867, 0.871, 0.184)
c9 = (0.949, 0.922, 0.678)
c10 = (0.984, 0.455, 0.231)
c11 = (0.765, 0.835, 0.922)
c12 = (0.965, 0.694, 0.486)
c13 = (0.992, 0.584, 0.820)


def plot(
    data: Field3dData,
    view: Literal["los", "side", "angular"],
    footpoints_grid: bool = False,
    save: bool = False,
    path: str | None = None,
    zoom: bool = False,
):
    """
    Create figure of magnetic field line from Field3dData object. Specify angle of view and optional zoom
    for the side view onto the transition region, which footpoints are chosen for field lines,
    if and where the figure is supposed to be saved.
    """

    xmin, xmax, ymin, ymax, zmin, zmax = (
        data.x[0],
        data.x[-1],
        data.y[0],
        data.y[-1],
        data.z[0],
        data.z[-1],
    )

    fig = plt.figure()
    ax = fig.figure.add_subplot(111, projection="3d")
    plot_magnetogram(data, ax)

    if footpoints_grid:
        plot_fieldlines_grid(data, ax)
    else:
        sinks, sources = detect_footpoints(data)
        if not zoom:
            plot_fieldlines_footpoints(data, sinks, sources, ax)
        else:
            plot_fieldlines_footpoints_zoom(data, sinks, sources, ax)

    if view == "los":
        ax.view_init(90, -90)  # type: ignore

        ax.set_xlabel("x", labelpad=5)
        ax.set_ylabel("y", labelpad=-0.1)

        ax.set_xticks(np.arange(0, xmax + 1.0 * 10**-8, xmax / 5))
        ax.set_yticks(np.arange(0, ymax + 1.0 * 10**-8, ymax / 5))

        ax.set_zticklabels([])  # type: ignore
        ax.set_zlabel("")  # type: ignore

        [t.set_va("center") for t in ax.get_yticklabels()]  # type: ignore
        [t.set_ha("center") for t in ax.get_yticklabels()]  # type: ignore

        [t.set_va("center") for t in ax.get_xticklabels()]  # type: ignore
        [t.set_ha("center") for t in ax.get_xticklabels()]  # type: ignore

    if view == "side":
        if not zoom:
            ax.view_init(0, -90)  # type: ignore
            ax.set_xlabel("x", labelpad=30)
            ax.set_zlabel("z", labelpad=0.5)  # type: ignore

            ax.set_xticks(np.arange(0, xmax + 1.0 * 10**-8, xmax / 5))
            ax.set_zticks(np.arange(0, zmax + 1.0 * 10**-8, zmax / 4))  # type: ignore

            ax.set_yticklabels([])  # type: ignore
            ax.set_ylabel("")

            [t.set_va("top") for t in ax.get_xticklabels()]  # type: ignore
            [t.set_ha("center") for t in ax.get_xticklabels()]  # type: ignore

            [t.set_va("center") for t in ax.get_zticklabels()]  # type: ignore
            [t.set_ha("center") for t in ax.get_zticklabels()]  # type: ignore
        else:
            ax.view_init(0, -90)
            ax.set_xticks(np.arange(0, xmax + 1.0 * 10**-8, xmax / 5))
            ax.set_zticks(np.arange(0, 2 * data.z0 + 1.0 * 10**-8, zmax / 5))
            ax.set_xlabel("x", labelpad=50)
            ax.set_zlabel("z", labelpad=10)  # type: ignore
            ax.set_yticklabels([])  # type: ignore
            ax.set_ylabel("")

            [t.set_va("top") for t in ax.get_xticklabels()]  # type: ignore
            [t.set_ha("center") for t in ax.get_xticklabels()]  # type: ignore

            [t.set_va("center") for t in ax.get_zticklabels()]  # type: ignore
            [t.set_ha("center") for t in ax.get_zticklabels()]  # type: ignore

    if view == "angular":
        ax.view_init(30, 240, 0)  # type: ignore

        ax.set_xticks(np.arange(0, xmax + 1.0 * 10**-8, xmax / 5))
        ax.set_yticks(np.arange(0, ymax + 1.0 * 10**-8, ymax / 5))
        ax.set_zticks(np.arange(0, zmax + 1.0 * 10**-8, zmax / 5))  # type: ignore

        [t.set_va("bottom") for t in ax.get_yticklabels()]  # type: ignore
        [t.set_ha("right") for t in ax.get_yticklabels()]  # type: ignore

        [t.set_va("bottom") for t in ax.get_xticklabels()]  # type: ignore
        [t.set_ha("left") for t in ax.get_xticklabels()]  # type: ignore

        [t.set_va("top") for t in ax.get_zticklabels()]  # type: ignore
        [t.set_ha("center") for t in ax.get_zticklabels()]  # type: ignore

    if save:

        assert path is not None

        if not zoom:
            temp = "/fieldlines_"
        else:
            temp = "/fieldlines_zoom_"

        plotname = (
            path
            + temp
            + str(data.a)
            + "_"
            + str(data.alpha)
            + "_"
            + str(data.b)
            + "_"
            + str(data.deltaz)
            + "_"
            + view
            + ".png"
        )

        plt.savefig(plotname, dpi=600, bbox_inches="tight", pad_inches=0.1)

    plt.show()


def find_center(data: Field3dData) -> Tuple:
    """
    Find centres of poles on photospheric magentogram.
    """

    xmin, xmax, ymin, ymax, zmin, zmax = (
        data.x[0],
        data.x[-1],
        data.y[0],
        data.y[-1],
        data.z[0],
        data.z[-1],
    )

    neighborhood_size = data.nx / 1.3
    threshold = 1.0

    data_max = maximum_filter(data.bz, neighborhood_size)  # mode ='reflect'
    maxima = data.bz == data_max
    data_min = minimum_filter(data.bz, neighborhood_size)
    minima = data.bz == data_min

    diff = (data_max - data_min) > threshold
    maxima[diff == 0] = 0
    minima[diff == 0] = 0

    labeled_sources, num_objects_sources = label(maxima)
    slices_sources = find_objects(labeled_sources)
    x_sources, y_sources = [], []

    labeled_sinks, num_objects_sinks = label(minima)
    slices_sinks = find_objects(labeled_sinks)
    x_sinks, y_sinks = [], []

    for dy, dx in slices_sources:
        x_center = (dx.start + dx.stop - 1) / 2
        x_sources.append(x_center / (data.nx / xmax))
        y_center = (dy.start + dy.stop - 1) / 2
        y_sources.append(y_center / (data.ny / ymax))

    for dy, dx in slices_sinks:
        x_center = (dx.start + dx.stop - 1) / 2
        x_sinks.append(x_center / (data.nx / xmax))
        y_center = (dy.start + dy.stop - 1) / 2
        y_sinks.append(y_center / (data.ny / ymax))

    return x_sources, y_sources, x_sinks, y_sinks


def show_poles(data: Field3dData):
    """
    Show centres of poles on photospheric magentogram.
    """

    x_plot = np.outer(data.y, np.ones(data.nx))
    y_plot = np.outer(data.x, np.ones(data.ny)).T

    xmin, xmax, ymin, ymax, zmin, zmax = (
        data.x[0],
        data.x[-1],
        data.y[0],
        data.y[-1],
        data.z[0],
        data.z[-1],
    )

    x_sources, y_sources, x_sinks, y_sinks = find_center(data)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.grid(color="white", linestyle="dotted", linewidth=0.5)
    ax.contourf(y_plot, x_plot, data.bz, 1000, cmap=cmap)  # , norm=norm)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tick_params(direction="in", length=2, width=0.5)
    ax.set_box_aspect(ymax / xmax)

    for i in range(0, len(x_sinks)):

        xx = x_sinks[i]
        yy = y_sinks[i]
        ax.scatter(xx, yy, marker="x", color=c2)

    for i in range(0, len(x_sources)):

        xx = x_sources[i]
        yy = y_sources[i]
        ax.scatter(xx, yy, marker="x", color=c1)

    sinks_label = mpatches.Patch(color=c2, label="Sinks")
    sources_label = mpatches.Patch(color=c1, label="Sources")

    plt.legend(handles=[sinks_label, sources_label], frameon=False)

    plt.show()


def detect_footpoints(data: Field3dData) -> Tuple:
    """
    Detenct footpoints around centres of poles on photospheric magentogram.
    """

    sinks = data.bz.copy()
    sources = data.bz.copy()

    maxmask = sources < sources.max() * 0.4
    sources[maxmask != 0] = 0

    minmask = sinks < sinks.min() * 0.4
    sinks[minmask == 0] = 0

    return sinks, sources


def show_footpoints(data: Field3dData) -> None:
    """
    Show footpoints around centres of poles on photospheric magentogram.
    """

    sinks, sources = detect_footpoints(data)

    xmin, xmax, ymin, ymax, zmin, zmax = (
        data.x[0],
        data.x[-1],
        data.y[0],
        data.y[-1],
        data.z[0],
        data.z[-1],
    )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contourf(
        np.outer(data.x, np.ones(data.ny)).T,
        np.outer(data.y, np.ones(data.nx)),
        data.bz,
        1000,
        cmap=cmap,
        # norm=norm,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tick_params(direction="in", length=2, width=0.5)
    ax.set_box_aspect(ymax / xmax)

    for ix in range(0, data.nx, int(data.nx / 20)):
        for iy in range(0, data.ny, int(data.ny / 20)):
            if sources[iy, ix] != 0:
                ax.scatter(
                    ix / (data.nx / xmax),
                    iy / (data.ny / ymax),
                    color=c1,
                    s=0.5,
                )

            if sinks[iy, ix] != 0:
                ax.scatter(
                    ix / (data.nx / xmax),
                    iy / (data.ny / ymax),
                    color=c2,
                    s=0.5,
                )

    sinks_label = mpatches.Patch(color=c2, label="Sinks")
    sources_label = mpatches.Patch(color=c1, label="Sources")

    plt.legend(handles=[sinks_label, sources_label], frameon=False)

    plt.show()


def plot_magnetogram(data: Field3dData, ax) -> None:
    """
    Plot photospheric boundary condition as basis for field line figures.
    """

    xmin, xmax, ymin, ymax, zmin, zmax = (
        data.x[0],
        data.x[-1],
        data.y[0],
        data.y[-1],
        data.z[0],
        data.z[-1],
    )

    x_big = np.arange(2.0 * data.nx) * 2.0 * xmax / (2.0 * data.nx - 1) - xmax
    y_big = np.arange(2.0 * data.ny) * 2.0 * ymax / (2.0 * data.ny - 1) - ymax

    x_grid, y_grid = np.meshgrid(x_big, y_big)
    ax.contourf(
        x_grid[data.ny : 2 * data.ny, data.nx : 2 * data.nx],
        y_grid[data.ny : 2 * data.ny, data.nx : 2 * data.nx],
        data.bz,
        1000,
        cmap=cmap,
        offset=0.0,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")  # type: ignore
    ax.grid(False)
    ax.set_zlim(zmin, zmax)  # type: ignore
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_box_aspect((xmax, ymax, zmax))  # type : ignore # (xmax, ymax, 2 * data.z0)

    ax.xaxis._axinfo["tick"]["inward_factor"] = 0.2  # type : ignore
    ax.xaxis._axinfo["tick"]["outward_factor"] = 0  # type : ignore
    ax.yaxis._axinfo["tick"]["inward_factor"] = 0.2  # type : ignore
    ax.yaxis._axinfo["tick"]["outward_factor"] = 0  # type : ignore
    ax.zaxis._axinfo["tick"]["inward_factor"] = 0.2  # type : ignore
    ax.zaxis._axinfo["tick"]["outward_factor"] = 0  # type : ignore

    # ax.xaxis.pane.fill = False  # type : ignore
    # ax.yaxis.pane.fill = False  # type : ignore
    # ax.zaxis.pane.fill = False  # type : ignore

    [t.set_va("center") for t in ax.get_yticklabels()]  # type : ignore
    [t.set_ha("center") for t in ax.get_yticklabels()]  # type : ignore

    [t.set_va("top") for t in ax.get_xticklabels()]  # type : ignore
    [t.set_ha("center") for t in ax.get_xticklabels()]  # type : ignore

    [t.set_va("center") for t in ax.get_zticklabels()]  # type : ignore
    [t.set_ha("center") for t in ax.get_zticklabels()]  # type : ignore


def plot_fieldlines_footpoints(
    data: Field3dData, sinks: np.ndarray, sources: np.ndarray, ax
):
    """
    Plot field lines starting at detected foot points around poles.
    """

    xmin, xmax, ymin, ymax, zmin, zmax = (
        data.x[0],
        data.x[-1],
        data.y[0],
        data.y[-1],
        data.z[0],
        data.z[-1],
    )

    x_big = np.arange(2.0 * data.nx) * 2.0 * xmax / (2.0 * data.nx - 1) - xmax
    y_big = np.arange(2.0 * data.ny) * 2.0 * ymax / (2.0 * data.ny - 1) - ymax

    h1 = 1.0 / 100.0  # Initial step length for fieldline3D
    eps = 1.0e-8
    # Tolerance to which we require point on field line known for fieldline3D
    hmin = 0.0  # Minimum step length for fieldline3D
    hmax = 1.0  # Maximum step length for fieldline3D

    # Limit fieldline plot to original data size (rather than Seehafer size)
    boxedges = np.zeros((2, 3))

    # # Y boundaries must come first, X second due to switched order explained above
    boxedges[0, 0] = ymin
    boxedges[1, 0] = ymax
    boxedges[0, 1] = xmin
    boxedges[1, 1] = xmax
    boxedges[0, 2] = zmin
    boxedges[1, 2] = zmax  # 2 * data.z0  # FOR ZOOM

    for ix in range(0, data.nx, int(data.nx / 20)):
        for iy in range(0, data.ny, int(data.ny / 20)):
            if sources[iy, ix] != 0 or sinks[iy, ix] != 0:

                x_start = ix / (data.nx / xmax)
                y_start = iy / (data.ny / ymax)
                # print(x_start, y_start)
                if data.bz[int(y_start), int(x_start)] < 0.0:
                    h1 = -h1

                ystart = [y_start, x_start, 0.0]

                fieldline = fieldline3d(
                    ystart,
                    data.field,
                    y_big,
                    x_big,
                    data.z,
                    h1,
                    hmin,
                    hmax,
                    eps,
                    oneway=False,
                    boxedge=boxedges,
                    gridcoord=False,
                    coordsystem="cartesian",
                )  # , periodicity='xy')

                if np.isclose(fieldline[:, 2][-1], 0.0) and np.isclose(
                    fieldline[:, 2][0], 0.0
                ):
                    # Need to give row direction first/ Y, then column direction/ X
                    ax.plot(
                        fieldline[:, 1],
                        fieldline[:, 0],
                        fieldline[:, 2],
                        color=c7,
                        linewidth=0.5,
                        zorder=4000,
                    )

                else:
                    ax.plot(
                        fieldline[:, 1],
                        fieldline[:, 0],
                        fieldline[:, 2],
                        color=c7,
                        linewidth=0.5,
                        zorder=4000,
                    )


def plot_fieldlines_footpoints_zoom(
    data: Field3dData, sinks: np.ndarray, sources: np.ndarray, ax
):
    """
    Plot field lines starting at detected foot points around poles zoomed into transition region.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = (
        data.x[0],
        data.x[-1],
        data.y[0],
        data.y[-1],
        data.z[0],
        data.z[-1],
    )

    ax.set_zlim(zmin, 2 * data.z0)
    ax.set_zticks(np.arange(0, 2 * data.z0 + 1, 2))
    ax.set_box_aspect((xmax, ymax, 4 * data.z0))

    x_big = np.arange(2.0 * data.nx) * 2.0 * xmax / (2.0 * data.nx - 1) - xmax
    y_big = np.arange(2.0 * data.ny) * 2.0 * ymax / (2.0 * data.ny - 1) - ymax

    h1 = 1.0 / 100.0  # Initial step length for fieldline3D
    eps = 1.0e-8
    # Tolerance to which we require point on field line known for fieldline3D
    hmin = 0.0  # Minimum step length for fieldline3D
    hmax = 1.0  # Maximum step length for fieldline3D

    # Limit fieldline plot to original data size (rather than Seehafer size)
    boxedges = np.zeros((2, 3))

    # # Y boundaries must come first, X second due to switched order explained above
    boxedges[0, 0] = ymin
    boxedges[1, 0] = ymax
    boxedges[0, 1] = xmin
    boxedges[1, 1] = xmax
    boxedges[0, 2] = zmin
    boxedges[1, 2] = 2 * data.z0  # FOR ZOOM

    for ix in range(0, data.nx, int(data.nx / 28)):
        for iy in range(0, data.ny, int(data.ny / 28)):
            if sources[iy, ix] != 0:  # or sinks[iy, ix] != 0:

                x_start = ix / (data.nx / xmax)
                y_start = iy / (data.ny / ymax)

                if data.bz[int(y_start), int(x_start)] < 0.0:
                    h1 = -h1

                ystart = [y_start, x_start, 0.0]

                fieldline = fieldline3d(
                    ystart,
                    data.field,
                    y_big,
                    x_big,
                    data.z,
                    h1,
                    hmin,
                    hmax,
                    eps,
                    oneway=False,
                    boxedge=boxedges,
                    gridcoord=False,
                    coordsystem="cartesian",
                )  # , periodicity='xy')

                if np.isclose(fieldline[:, 2][-1], 0.0) and np.isclose(
                    fieldline[:, 2][0], 0.0
                ):
                    # Need to give row direction first/ Y, then column direction/ X
                    ax.plot(
                        fieldline[:, 1],
                        fieldline[:, 0],
                        fieldline[:, 2],
                        color=c2,
                        linewidth=0.5,
                        zorder=4000,
                    )
                else:
                    ax.plot(
                        fieldline[:, 1],
                        fieldline[:, 0],
                        fieldline[:, 2],
                        color=c2,
                        linewidth=0.5,
                        zorder=4000,
                    )


def plot_fieldlines_grid(data: Field3dData, ax) -> None:
    """
    Plot field lines on grid.
    """

    xmin, xmax, ymin, ymax, zmin, zmax = (
        data.x[0],
        data.x[-1],
        data.y[0],
        data.y[-1],
        data.z[0],
        data.z[-1],
    )

    x_big = np.arange(2.0 * data.nx) * 2.0 * xmax / (2.0 * data.nx - 1) - xmax
    y_big = np.arange(2.0 * data.ny) * 2.0 * ymax / (2.0 * data.ny - 1) - ymax

    x_0 = 0.0000001
    y_0 = 0.0000001
    dx = xmax / 20.0
    dy = ymax / 20.0

    nlinesmaxx = math.floor(xmax / dx)
    nlinesmaxy = math.floor(ymax / dy)

    h1 = 1.0 / 100.0  # Initial step length for fieldline3D
    eps = 1.0e-8
    # Tolerance to which we require point on field line known for fieldline3D
    hmin = 0.0  # Minimum step length for fieldline3D
    hmax = 1.0  # Maximum step length for fieldline3D

    # Limit fieldline plot to original data size (rather than Seehafer size)
    boxedges = np.zeros((2, 3))

    # # Y boundaries must come first, X second due to switched order explained above
    boxedges[0, 0] = ymin
    boxedges[1, 0] = ymax
    boxedges[0, 1] = xmin
    boxedges[1, 1] = xmax
    boxedges[0, 2] = zmin
    boxedges[1, 2] = zmax

    for ilinesx in range(0, nlinesmaxx):
        for ilinesy in range(0, nlinesmaxy):
            x_start = x_0 + dx * ilinesx
            y_start = y_0 + dy * ilinesy

            if data.bz[int(y_start), int(x_start)] < 0.0:
                h1 = -h1

            ystart = [y_start, x_start, 0.0]
            # Fieldline3D expects startpt, BField, Row values, Column values so we need to give Y first, then X
            fieldline = fieldline3d(
                ystart,
                data.field,
                y_big,
                x_big,
                data.z,
                h1,
                hmin,
                hmax,
                eps,
                oneway=False,
                boxedge=boxedges,
                gridcoord=False,
                coordsystem="cartesian",
            )  # , periodicity='xy')

            if np.isclose(fieldline[:, 2][-1], 0.0) and np.isclose(
                fieldline[:, 2][0], 0.0
            ):
                # Need to give row direction first/ Y, then column direction/ X
                ax.plot(
                    fieldline[:, 1],
                    fieldline[:, 0],
                    fieldline[:, 2],
                    color=c7,  # (0.576, 1.000, 0.271),
                    linewidth=0.5,
                    zorder=4000,
                )
            else:
                ax.plot(
                    fieldline[:, 1],
                    fieldline[:, 0],
                    fieldline[:, 2],
                    color=c7,  # (0.757, 0.329, 1.000),
                    linewidth=0.5,
                    zorder=4000,
                )


# def plot_plasma_parameters(data: Field3dData, path: str | None = None):

#     ix_max = np.unravel_index(data.bz.argmax(), data.bz.shape)[1]
#     iy_max = np.unravel_index(data.bz.argmax(), data.bz.shape)[0]

#     xmin, xmax, ymin, ymax, zmin, zmax = (
#         data.x[0],
#         data.x[-1],
#         data.y[0],
#         data.y[-1],
#         data.z[0],
#         data.z[-1],
#     )

#     z_arr = np.arange(data.nz) * (zmax - zmin) / (data.nz - 1) + zmin

#     fig, ax1 = plt.subplots()

#     p1 = ax1.plot(
#         z_arr,
#         data.dpressure[iy_max, ix_max, :],
#         linewidth=0.5,
#         linestyle="solid",
#         color=c2,
#         label=r"$\Delta p$",
#     )
#     ax1.set_ylabel(r"$\Delta p$")
#     ax2 = ax1.twinx()
#     p2 = ax2.plot(
#         z_arr,
#         data.ddensity[iy_max, ix_max, :],
#         linewidth=0.5,
#         linestyle="solid",
#         color=c1,
#         label=r"$\Delta \rho$",
#     )
#     ax2.set_ylabel(r"$\Delta \rho$")
#     # plt.xlim([0, 2 * data.z0])
#     ax1.set_xlabel("z")
#     ax1.tick_params(direction="in", length=2, width=0.5)
#     ax2.tick_params(direction="in", length=2, width=0.5)
#     lns = p1 + p2
#     labs = [l.get_label() for l in lns]
#     ax1.legend(lns, labs, loc=0, frameon=False)

#     if path is not None:
#         plotname = path + "/pp_variations.png"
#         plt.savefig(plotname, dpi=600, bbox_inches="tight", pad_inches=0.1)

#     plt.show()

#     fig, ax1 = plt.subplots()

#     p1 = ax1.plot(
#         z_arr,
#         data.bpressure,
#         linewidth=0.5,
#         linestyle="solid",
#         color=c2,
#         label=r"$p_b$",
#     )
#     ax1.set_ylabel(r"$p_b$")
#     ax2 = ax1.twinx()
#     p2 = ax2.plot(
#         z_arr,
#         data.bdensity,
#         linewidth=0.5,
#         linestyle="solid",
#         color=c1,
#         label=r"$\rho_b$",
#     )
#     ax2.set_ylabel(r"$\rho_b$")
#     # plt.xlim([0, 2 * data.z0])
#     ax1.set_xlabel("z")
#     ax1.tick_params(direction="in", length=2, width=0.5)
#     ax2.tick_params(direction="in", length=2, width=0.5)
#     lns = p1 + p2
#     labs = [l.get_label() for l in lns]
#     ax1.legend(lns, labs, loc=0, frameon=False)

#     if path is not None:
#         plotname = path + "/pp_background.png"
#         plt.savefig(plotname, dpi=600, bbox_inches="tight", pad_inches=0.1)

#     plt.show()

#     fig, ax1 = plt.subplots()

#     p1 = ax1.plot(
#         z_arr,
#         data.fpressure[iy_max, ix_max, :],
#         linewidth=0.5,
#         linestyle="solid",
#         color=c2,
#         label=r"$p$",
#     )
#     ax1.set_ylabel(r"$p$")
#     ax2 = ax1.twinx()
#     p2 = ax2.plot(
#         z_arr,
#         data.fdensity[iy_max, ix_max, :],
#         linewidth=0.5,
#         linestyle="solid",
#         color=c1,
#         label=r"$\rho$",
#     )
#     ax2.set_ylabel(r"$\rho$")
#     # plt.xlim([0, 2 * data.z0])
#     ax1.set_xlabel("z")
#     ax1.tick_params(direction="in", length=2, width=0.5)
#     ax2.tick_params(direction="in", length=2, width=0.5)
#     lns = p1 + p2
#     labs = [l.get_label() for l in lns]
#     ax1.legend(lns, labs, loc=0, frameon=False)

#     if path is not None:
#         plotname = path + "/pp.png"
#         plt.savefig(plotname, dpi=600, bbox_inches="tight", pad_inches=0.1)

#     plt.show()


# def plot_pp_comp(data1: Field3dData, data2: Field3dData, path: str):

#     if data1.field.shape != data2.field.shape:
#         raise ValueError("Field sizes do not match.")

#     xmin, xmax, ymin, ymax, zmin, zmax = (
#         data1.x[0],
#         data1.x[-1],
#         data1.y[0],
#         data1.y[-1],
#         data1.z[0],
#         data1.z[-1],
#     )

#     z_arr = np.arange(data1.nz) * (zmax - zmin) / (data1.nz - 1) + zmin

#     fig, ax1 = plt.subplots()

#     plt.plot(
#         z_arr,
#         data1.dpressure[0, 0, :],
#         linewidth=0.4,
#         linestyle="solid",
#         color=c4,
#         alpha=0.5,
#         label="Field A MHS",
#     )
#     plt.plot(
#         z_arr,
#         data2.dpressure[0, 0, :],
#         linewidth=0.4,
#         linestyle="solid",
#         color=c2,
#         alpha=0.5,
#         label="Field B MHS",
#     )
#     for ix in range(1, data1.nx, 10):
#         for iy in range(1, data1.ny, 10):
#             plt.plot(
#                 z_arr,
#                 data1.dpressure[iy, ix, :],
#                 linewidth=0.4,
#                 linestyle="solid",
#                 color=c4,
#                 alpha=0.5,
#             )
#             plt.plot(
#                 z_arr,
#                 data2.dpressure[iy, ix, :],
#                 linewidth=0.4,
#                 linestyle="solid",
#                 color=c2,
#                 alpha=0.5,
#             )
#     plt.plot(
#         z_arr,
#         np.zeros_like(z_arr),
#         linewidth=0.4,
#         linestyle="solid",
#         color="black",
#         label="LLF",
#     )
#     plt.xlim([0, 2 * data1.z0])
#     plt.ylabel(r"$\Delta p$")
#     plt.xlabel("z")
#     plt.legend(frameon=False)
#     plt.tick_params(direction="in", length=2, width=0.5)
#     plotname = path + "/pressurevar_comp.png"
#     plt.savefig(plotname, dpi=600, bbox_inches="tight", pad_inches=0.1)
#     plt.show()

#     plt.plot(
#         z_arr,
#         data1.ddensity[0, 0, :],
#         linewidth=0.4,
#         linestyle="solid",
#         color=c4,
#         alpha=0.5,
#         label="Field A MHS",
#     )
#     plt.plot(
#         z_arr,
#         data2.ddensity[0, 0, :],
#         linewidth=0.4,
#         linestyle="solid",
#         color=c2,
#         alpha=0.5,
#         label="Field B MHS",
#     )
#     for ix in range(1, data1.nx, 10):
#         for iy in range(1, data1.ny, 10):
#             plt.plot(
#                 z_arr,
#                 data1.ddensity[iy, ix, :],
#                 linewidth=0.4,
#                 linestyle="solid",
#                 color=c4,
#                 alpha=0.5,
#             )
#             plt.plot(
#                 z_arr,
#                 data2.ddensity[iy, ix, :],
#                 linewidth=0.4,
#                 linestyle="solid",
#                 color=c2,
#                 alpha=0.5,
#             )
#     plt.plot(
#         z_arr,
#         np.zeros_like(z_arr),
#         linewidth=0.4,
#         linestyle="solid",
#         color="black",
#         label="LLF",
#     )
#     plt.xlim([0, 2 * data1.z0])
#     plt.ylabel(r"$\Delta \rho$")
#     plt.xlabel("z")
#     plt.legend(frameon=False)
#     plt.tick_params(direction="in", length=2, width=0.5)
#     plotname = path + "/densityvar_comp.png"
#     plt.savefig(plotname, dpi=600, bbox_inches="tight", pad_inches=0.1)
#     plt.show()
