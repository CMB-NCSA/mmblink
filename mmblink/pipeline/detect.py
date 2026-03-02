"""Functions for detecting sources in images and producing catalogs."""

import numpy as np
from photutils.background import Background2D, StdBackgroundRMS
from photutils import segmentation
from photutils.segmentation import SourceCatalog

__all__ = [
    "background_rms_map",
    "catalog_sources",
    "filter_ellipticity",
    "n_sigma_threshold",
    "segment_sources",
    "set_catalog_obs_info",
    "set_sky_centroid_string_column",
    "set_snr_max_column",
]


def background_rms_map(
    flux, *, box_size, mask=None, filter_size=3, sigma_clip=None
):
    """Estimate a map of the background RMS (Root Mean Square).

    Estimates a map of the background RMS across the 2D input data using the
    standard deviation. A box size must be specified for the grid used to
    estimate the background. Optionally, a mask can be used to ignore pixels in
    the image and a sigma clip can be used for outlier rejection.

    Parameters
    ----------
    flux : ndarray
        A 2D array containing the input data.
    box_size : int or tuple of int
        The size of the box used to estimate the background. If it is a scalar,
        a square box size is used. If it has two elements, they should be in
        (y, x) order.
    mask : ndarray, optional
        A 2D boolean array of the same shape as data, where `True` values
        indicate masked pixels. If `None`, no masking is applied. Default is
        `None`.
    filter_size : int or tuple of int, optional
        The size of the filter used for smoothing the background estimate. If
        filter_size is a scalar, a square of size filter_size will be used. If
        a tuple is provided, it must be in (y, x) order. Both dimensions must
        be odd. Default is 3.
    sigma_clip : astropy.stats.SigmaClip or None, optional
        If specified, values in the input data will be clipped according to the
        SigmaClip object when computing box statistics in the background.
        Otherwise, sigma clipping is not used. Default is `None`.

    Returns
    -------
    background_rms : ndarray
        A map the background RMS with the same shape as the data.

    Notes
    -----
    - The sigma_clip is applied to the background estimation if specified. It
      helps to reject outliers in the background calculation.
    - The mask is not applied to the returned maps. However, masked pixels are
      not used during the calculation of the background and RMS maps.
    """
    # Sigma clipping is already built into the Background2D object, so it is
    # not needed for the estimator.
    bkgrms_estimator = StdBackgroundRMS()
    bkg = Background2D(
        flux,
        box_size,
        mask=mask,
        filter_size=filter_size,
        sigma_clip=sigma_clip,
        bkgrms_estimator=bkgrms_estimator,
    )
    return bkg.background_rms


def n_sigma_threshold(n_sigma, background_rms):
    """The flux threshold for an n-sigma source detection.

    Creates a threshold required for an n-sigma detection in a
    background-subtracted image. If background_rms is a scalar, the threshold
    is a scalar. If background_rms is an ndarray, the threshold is an ndarray of
    the same shape.

    Parameters
    ----------
    n_sigma : float
        The number of sigmas for a detection.
    background_rms : float or ndarray
        The standard deviation of the data. This can be a single scalar
        value or an RMS map matching the data dimensions.

    Returns
    -------
    threshold : float or ndarray
        The threshold required for an n-sigma detection matching the shape of
        background_rms.
    """
    return n_sigma * background_rms


def segment_sources(flux, threshold, *, npixels, mask=None):
    """Create a segmented image of detected sources in a map.

    Detects all sources in the image above a specified threshold. Sources must
    have more than npixels connected pixels to be valid.

    Parameters
    ----------
    flux : ndarray
        Background-subtracted 2D array of image data.
    threshold : float or ndarray
        If threshold is a scalar, it used for the detection threshold across the
        entire image. If threshold is an ndarray, it should have the same shape
        as flux and the corresponding pixels will be used as the threshold for
        detection.
    npixels : int
        The minimum number of connected pixels to consider for a valid
        detection.
    mask : ndarray, optional
        A mask to apply to the data before detection with the same shape as
        flux.

    Returns
    -------
    seg_map : photutils.segmentation.SegmentationImage or `None`
        The segmentation map of detected sources or `None` if no sources were
        found.
    """
    seg_map = segmentation.detect_sources(
        flux, threshold, npixels=npixels, mask=mask
    )
    if seg_map is None:
        # No sources were found.
        return None
    seg_map = segmentation.deblend_sources(
        flux,
        seg_map,
        npixels=npixels,
        nlevels=32,
        contrast=0.001,
        progress_bar=False,
    )
    return seg_map


def catalog_sources(flux, seg_map, *, weight=None, mask=None, wcs=None):
    """Create a table of sources and derive parameters.

    Parameters
    ----------
    flux : ndarray
        Background-subtracted 2D array of image data.
    seg_map : photutils.segmentation.SegmentationImage
        The segmented map created for the same flux.
    weight : ndarray, optional
        Map of the weight of the image with the same shape as flux.
    mask : ndarray, optional
        A mask to apply to the image with the same shape as flux.
    wcs : astropy.wcs.WCS, optional
        The WCS to convert the image pixel coordinates into sky coordinates.

    Returns
    -------
    table : astropy.table.QTable
        A table of sources properties with each source in a row.
    """
    catalog = SourceCatalog(flux, seg_map, error=weight, mask=mask, wcs=wcs)
    # The columns to use that are computed by SourceCatalog.
    columns = [
        "label",
        "xcentroid",
        "ycentroid",
        "sky_centroid",
        "bbox_xmin",
        "bbox_xmax",
        "bbox_ymin",
        "bbox_ymax",
        "area",
        "semimajor_sigma",
        "semiminor_sigma",
        "orientation",
        "eccentricity",
        "min_value",
        "max_value",
        "local_background",
        "segment_flux",
        "segment_fluxerr",
        "kron_flux",
        "kron_fluxerr",
        "elongation",
        "ellipticity",
    ]
    return catalog.to_table(columns)


def set_snr_max_column(catalog, sigma):
    """Sets the estimated maximum signal to noise ratio column in the catalog.

    Estimates the signal-to-noise-ratio (SNR) using the maximum pixel flux as
    signal. If an estimated noise map is given, it samples the noise at the
    source centroid. The column `"snr_max"` is added to the table in-place.

    Parameters
    ----------
    table : astropy.table.Table
        The table to add the column. Must have a `"max_value"` column. If sigma
        is a 2D array, must have `"xcentroid"` and `"ycentroid"` columns.
    sigma : float or ndarray
        An estimate of the noise. If an ndarray is given, the noise is estimated
        at the centroid of the source.

    Returns
    -------
    table : astropy.table.Table
        The table with the `"snr_max"` column added.
    """
    if np.isscalar(sigma):
        noise = sigma
    else:
        # Use the centroid in the 2D array to estimate noise.
        x = np.round(catalog["xcentroid"].data).astype(int)
        y = np.round(catalog["ycentroid"].data).astype(int)
        noise = sigma[y, x]
    catalog["snr_max"] = catalog["max_value"].data / noise
    return catalog


def set_sky_centroid_string_column(catalog):
    """Add a human-readable string of the centroid sky coordinate to the table.

    Determines the source centroids from the `"sky_centroid"` column. The string
    is formatted in hmsdms. The column "`sky_centroid_dms`" is added to the
    table in-place. This column will not be updated when the `"sky_centroid"`
    column is updated.

    Parameters
    ----------
    catalog : astropy.table.Table
        The table to add the column. Must have a `"sky_centroid"` column.

    Returns
    -------
    new_catalog : astropy.table.Table
        The table with the "sky_centroid_dms" column added.

    """
    catalog["sky_centroid_dms"] = catalog["sky_centroid"].to_string(
        "hmsdms", precision=0
    )
    return catalog


def set_catalog_obs_info(catalog, *, band, obs_id, field):
    """Add information about the observation to catalog in-place.

    Adds the band, obsID, and field to the catalogs's metadata, and adds the
    `"obs"`, `"obs_max"`, and `"band"` columns to the catalog. The `"obs_max"`
    column is formatted as obsID_band.

    Parameters
    ----------
    catalog : astropy.table.Table
        The table to modify.
    band : str
        The name of the band for this catalog
    obs_id : str
        The observation ID for this catalog.
    field : str
        The field for this catalog.

    Returns
    -------
    new_catalog : astropy.table.Table
        The modified catalog with the `"band"`, `"obsID"` and `"field"` metadata
        added and the `"obs"`, `"obs_max"` and `"band"` columns added.
    """
    catalog.meta["band"] = band
    catalog.meta["obsID"] = obs_id
    catalog.meta["field"] = field

    catalog.add_column(obs_id, name="obs", index=0)
    catalog.add_column(f"{obs_id}_{band}", name="obs_max", index=0)
    catalog.add_column(band, name="band", index=0)

    return catalog


def filter_ellipticity(catalog, cut):
    """Remove objects with an ellipticity greater than or equal to a cut.

    Filters a catalog to only keep rows where the ellipticity is below the
    given cut. It also removes all rows where the ellipticity is NaN.

    Parameters
    ----------
    catalog : astropy.table.Table
        The catalog of objects to filter based on ellipticity. Must have a
        column named "ellipticity".
    cut : float
        Only rows below this value are kept.

    Returns
    -------
    filtered : astropy.table.Table
        A new table with filtered rows removed.

    """
    return catalog[
        ~((catalog["ellipticity"] >= cut) | np.isnan(catalog["ellipticity"]))
    ]
