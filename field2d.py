from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from astropy.io.fits import open as astroopen
from astropy.io.fits import getdata
from astropy.coordinates import SkyCoord
from astropy import units as u

import sunpy.map


@dataclass
class Field2dData:
    """
    Dataclass of type Field2dData with the following attributes:
    nx, ny, nz  :   Dimensions of 3D magnetic field, usually nx and ny determined by magnetogram size,
                    while nz defined by user through height to which extrapolation is carried out.
    nf          :   Number of Fourier modes used in calculation of magnetic field vector, usually
                    nf = min(nx, ny) is taken. To do: split into nfx, nfy, sucht that all possible modes
                    in both directions can be used.
    px, py, pz  :   Pixel sizes in x-, y-, z-direction, in normal length scale (Mm).
    x, y, z     :   1D arrays of grid points on which magnetic field is given with shapes (nx,), (ny,)
                    and (nz,) respectively.
    bz          :   Bottom boundary magentogram of size (ny, nx,). Indexing of vectors done in this order,
                    such that, following intuition, x-direction corresponds to latitudinal extension and
                    y-direction to longitudinal extension of the magnetic field.
    """

    nx: np.int32
    ny: np.int32
    nz: np.int32
    nf: np.int32
    px: np.float64
    py: np.float64
    pz: np.float64
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    bz: np.ndarray

    @classmethod
    def from_fits_SolOr(cls, path):
        """
        Creates dataclass of type Field2dData from SolarOrbiter Archive data in .fits format.
        Only needs to be handed path to file and the creates Field2dData for extrapolation to 20 Mm
        (can be adjusted by hand). Most SolarOrbiter images need to be cut for extraplation of a
        certain active region. No straightforward method to automate this process through user input
        has been found yet, therefore size of the magnetogram needs to be adjusted by hand by the user.

        Steps:
        (1)     From the file given at path read in the image and header data refarding the distance to
                the sun, pixel unit and pixel size in arcsec.
        (2)     Cut image to specific size [sty:lsty, stx:lstx] around feature under investigation.
        (3)     Determine nx, ny, nf, px, py, xmax, ymax from data.
        (4)     Choose nz, pz and zmax.
        (5)     Determine x, y, z.
        (6)     Write all into Field2dData object.
        """

        with astroopen(path) as data:

            image = getdata(path, ext=False)

            hdr = data[0].header
            dist = hdr["DSUN_OBS"]
            px_unit = hdr["CUNIT1"]
            py_unit = hdr["CUNIT2"]
            px_arcsec = hdr["CDELT1"]
            py_arcsec = hdr["CDELT2"]

        stx = int(input("First pixel x axis: "))
        lstx = int(input("Last pixel x axis: "))
        sty = int(input("First pixel y axis: "))
        lsty = int(input("Last pixel y axis: "))

        image = image[sty:lsty, stx:lstx]

        nx = image.shape[1]
        ny = image.shape[0]

        nf = min(nx, ny)

        px_radians = px_arcsec / 206265.0
        py_radians = py_arcsec / 206265.0

        dist_Mm = dist * 10**-6
        px = px_radians * dist_Mm
        py = py_radians * dist_Mm

        print(px, py)

        xmin = 0.0
        ymin = 0.0
        zmin = 0.0

        xmax = nx * px
        ymax = ny * py
        zmax = 20.0

        pz = np.float64(90.0 * 10**-3)

        nz = np.int32(np.floor(zmax / pz))

        x = np.arange(nx) * (xmax - xmin) / (nx - 1) - xmin
        y = np.arange(ny) * (ymax - ymin) / (ny - 1) - ymin
        z = np.arange(nz) * (zmax - zmin) / (nz - 1) - zmin

        return Field2dData(nx, ny, nz, nf, px, py, pz, x, y, z, image)

    @classmethod
    def from_fits_SDO(cls, path):
        """
        Creates dataclass of type Field2dData from SDO HMI data in .fits format.
        Only needs to be handed path to file and the creates Field2dData for extrapolation to 20 Mm
        (can be adjusted by hand). Most HMI images need to be cut for extraplation of a certain active
        region. No straightforward method to automate this process through user input has been found yet,
        therefore size of the magnetogram needs to be adjusted by hand by the user.

        Steps:
        (1)     From the file given at path read in the image and header data refarding the distance to
                the sun, pixel unit and pixel size in arcsec.
        (2)     Cut image to specific size [sty:lsty, stx:lstx] around feature under investigation.
        (3)     Determine nx, ny, nf, px, py, xmax, ymax from data.
        (4)     Choose nz, pz and zmax.
        (5)     Determine x, y, z.
        (6)     Write all into Field2dData object.
        """

        hmi_image = sunpy.map.Map(path).rotate()

        hdr = hmi_image.fits_header

        sty = int(input("Upper boundary latitute: "))
        lsty = int(input("Upper boundary longitude:  "))
        stx = int(input("Lower boundary latitute: "))
        lstx = int(input("Lower boundary longitude: "))

        left_corner = SkyCoord(
            Tx=lsty * u.arcsec, Ty=sty * u.arcsec, frame=hmi_image.coordinate_frame
        )
        right_corner = SkyCoord(
            Tx=lstx * u.arcsec, Ty=stx * u.arcsec, frame=hmi_image.coordinate_frame
        )

        image = hmi_image.submap(left_corner, top_right=right_corner)

        dist = hdr["DSUN_OBS"]
        px_unit = hdr["CUNIT1"]
        py_unit = hdr["CUNIT2"]
        px_arcsec = hdr["CDELT1"]
        py_arcsec = hdr["CDELT2"]

        print(px_unit, py_unit, px_arcsec, py_arcsec)

        nx = image.data.shape[1]
        ny = image.data.shape[0]

        nf = min(nx, ny)

        px_radians = px_arcsec / 206265.0
        py_radians = py_arcsec / 206265.0

        print(px_radians, py_radians)

        print(dist)

        dist_Mm = dist * 10**-6
        px = px_radians * dist_Mm
        py = py_radians * dist_Mm

        print(dist_Mm)
        print(px, py)

        xmin = 0.0
        ymin = 0.0
        zmin = 0.0

        xmax = nx * px
        ymax = ny * py
        zmax = 20.0

        pz = np.float64(90.0 * 10**-3)

        nz = np.int32(np.floor(zmax / pz))

        x = np.arange(nx) * (xmax - xmin) / (nx - 1) - xmin
        y = np.arange(ny) * (ymax - ymin) / (ny - 1) - ymin
        z = np.arange(nz) * (zmax - zmin) / (nz - 1) - zmin

        return Field2dData(nx, ny, nz, nf, px, py, pz, x, y, z, image.data)


def check_fluxbalance(data: Field2dData) -> float:
    """
    Summation of flux through the bottom boundary (photospheric Bz) normalised
    by the sum of absolute values. Value between -1 and 1, corresponding to entirely
    outward and inward flux, respectively. Can (probably) consider values between
    -0.01 and 0.01 as flux-balanced, such that the application of Seehafer is not
    necessary.
    """
    return np.sum(data.bz) / np.sum(np.fabs(data.bz))


def alpha_HS04(bx: np.ndarray, by: np.ndarray, bz: np.ndarray) -> float:
    """
    "Optimal" alpha calculated according to Hagino and Sakurai (2004).
    Alpha is calculated from the vertical electric current in the photosphere
    (from horizontal photospheric field) and the photospheric vertical magnetic field.
    """
    Jz = np.gradient(by, axis=1) - np.gradient(bx, axis=0)
    return np.sum(Jz * np.sign(bz)) / np.sum(np.fabs(bz))
