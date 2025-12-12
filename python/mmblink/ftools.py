import mmblink.dtools as du
from spt3g import core, maps, sources
import numpy as np
import logging
import time
import math
from astropy.nddata import Cutout2D
import astropy.io.fits
import re
import os
import copy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from astropy.time import Time


# Mapping of metadata to FITS keywords
_keywords_map = {'ObservationStart': ('DATE-BEG', 'Observation start date'),
                 'ObservationStop': ('DATE-END', 'Observation end date'),
                 'ObservationID': ('OBSID', 'Observation ID'),
                 'Id': ('BAND', 'Band name, Observation Frequency'),
                 'SourceName': ('FIELD', 'Name of object'),
                 }
# Logger
logger = logging.getLogger(__name__)


def g3_to_fits(g3file, output_dir=None, trim=True, compress=False, quantize_level=16.0, overwrite=True):
    """
    Transform a G3 file containing a FlatSkyMap into a FITS file.

    Parameters:
    - g3file (str): Input G3 file path.
    - output_dir (str): The output directory for the fits files. 
      If None, then it's the same as the input directory.
    - trim (bool): If True, trims the FITS image to a smaller region.
    - compress (bool): If True, compresses the FITS file.
    - quantize_level (float): Quantization level for the transformation.
    - overwrite (bool): If True, overwrites an existing FITS file.
    """
    # Set the output name
    basename = g3file.split(".")[0]
    # Get a g3 handle
    g3 = core.G3File(g3file)
    # Extract metadata that will augment the header
    metadata = {}
    metadata['PARENT'] = (os.path.basename(g3file), 'Name of parent file')
    for frame in g3:
        logger.debug(f"Reading frame: {frame.type}")
        if frame.type == core.G3FrameType.Observation or frame.type == core.G3FrameType.Map:
            logger.debug(f"Extracting metadata from frame: {frame.type}")
            metadata = extract_metadata_frame(frame, metadata)

        if frame.type == core.G3FrameType.Map:
            t0 = time.time()
            logger.info(f"Transforming to FITS: {frame.type} -- Id: {frame['Id']}")
            logger.debug("Removing weights")
            maps.RemoveWeights(frame, zero_nans=True)
            maps.MakeMapsUnpolarized(frame)
            logger.debug("Removing units --> mJy")
            remove_units(frame, units=core.G3Units.mJy)
            hdr = maps.fitsio.create_wcs_header(frame['T'])
            hdr.update(metadata)
            hdr['UNITS'] = ('mJy', 'Data units')
            band = frame['Id']
            # We need to add band to the filename, so we have different outputs
            if band in basename:
                fname = f"{basename}.fits"
            else:
                fname = f"{basename}_{band}.fits"

            if output_dir is None:
                fitsfile = fname
            else:
                os.makedirs(output_dir, exist_ok=True)
                fitsfile = os.path.join(output_dir,fname.split("/")[-1])

            # If fitsfile already exists and overwrite is False we do nothing
            if overwrite is False and os.path.exists(fitsfile):
                logger.info(f"Skipping file: {fitsfile} -- already exists")
                continue

            if trim:
                field = metadata['FIELD'][0]
                logger.info(f"Will write trimmed FITS file for field: {field}")
                save_skymap_fits_trim(frame, fitsfile, field,
                                      hdr=hdr,
                                      compress=compress,
                                      overwrite=overwrite)
            else:
                # Get the T and weight frames
                T = frame['T']
                W = frame.get('Wpol', frame.get('Wunpol', None))
                maps.fitsio.save_skymap_fits(fitsfile, T,
                                             overwrite=overwrite,
                                             compress=compress,
                                             hdr=hdr,
                                             W=W)
            logger.info(f"Wrote file: {fitsfile}")
            logger.info(f"Time: {du.elapsed_time(t0)}")
    del g3


def extract_metadata_frame(frame, metadata=None):
    """
    Extract selected metadata from a frame in a G3 file.

    Parameters:
    - frame: A frame object from the G3 file.
    - metadata (dict): A dictionary to store the extracted metadata (optional).

    Returns:
    - metadata (dict): Updated metadata dictionary with extracted values.
    """
    # Loop over all items and select only the ones in the Mapping
    if not metadata:
        metadata = {}
    for k in iter(frame):
        if k in _keywords_map.keys():
            keyword = _keywords_map[k][0]
            # We need to treat BAND diferently to avoid inconsistensies
            # in how Id is defined (i.e Coadd_90GHz, 90GHz, vs combined_90GHz)
            if keyword == 'BAND':
                try:
                    value = re.findall("90GHz|150GHz|220GHz", frame[k])[0]
                except IndexError:
                    continue
            # Need to re-cast G3Time objects
            elif isinstance(frame[k], core.G3Time):
                value = astropy.time.Time(frame[k].isoformat(), format='isot', scale='utc').isot
            else:
                value = frame[k]
            metadata[keyword] = (value, _keywords_map[k][1])

    return metadata


def get_field_bbox(field, wcs, gridsize=100):
    """
    Get the image extent and central position in pixels for a given WCS.

    Parameters:
    - field (str): The name of the field.
    - wcs: WCS (World Coordinate System) object.
    - gridsize (int): Number of grid points along each axis.

    Returns:
    - tuple: (xc, yc, xsize, ysize) where xc, yc are the center coordinates
      and xsize, ysize are the image sizes in pixels.
    """
    deg = core.G3Units.deg
    (ra, dec) = sources.get_field_extent(field,
                                         ra_pad=1.5*deg,
                                         dec_pad=3*deg,
                                         sky_pad=True)
    # we convert back from G3units to degrees
    ra = (ra[0]/deg, ra[1]/deg)
    dec = (dec[0]/deg, dec[1]/deg)
    # Get the new ras corners in and see if we cross RA=0
    crossRA0, ra = crossRAzero(ra)
    # Create a grid of ra, dec to estimate the projected extent for the frame WCS
    ras = np.linspace(ra[0], ra[1], gridsize)
    decs = np.linspace(dec[0], dec[1], gridsize)
    ra_grid, dec_grid = np.meshgrid(ras, decs)
    # Transform ra, dec grid to image positions using astropy
    (x_grid, y_grid) = wcs.wcs_world2pix(ra_grid, dec_grid, 0)
    # Get min, max values for x,y grid
    xmin = math.floor(x_grid.min())
    xmax = math.ceil(x_grid.max())
    ymin = math.floor(y_grid.min())
    ymax = math.ceil(y_grid.max())
    # Get the size in pixels rounded to the nearest hundred
    xsize = round((xmax - xmin), -2)
    ysize = round((ymax - ymin), -2)
    xc = round((xmax+xmin)/2.)
    yc = round((ymax+ymin)/2.)
    logger.debug(f"Found center: ({xc}, {yc})")
    logger.debug(f"Found size: ({xsize}, {ysize})")
    return xc, yc, xsize, ysize


def crossRAzero(ras):
    """
    Check if the RA coordinates cross RA=0 and adjust accordingly.

    Parameters:
    - ras (array): An array of RA coordinates.

    Returns:
    - tuple: A tuple (CROSSRA0, ras) where CROSSRA0 is a boolean indicating if RA
      crosses zero, and ras is the adjusted RA array.
    """    # Make sure that they are numpy objetcs
    ras = np.array(ras)
    racmin = ras.min()
    racmax = ras.max()
    if (racmax - racmin) > 180.:
        # Currently we switch order. Perhaps better to +/-360.0?
        # Note we want the total extent which is not necessarily the maximum and minimum in this case
        ras2 = ras
        wsm = np.where(ras > 180.0)
        ras2[wsm] = ras[wsm] - 360.
        CROSSRA0 = True
        ras = ras2
    else:
        CROSSRA0 = False
    return CROSSRA0, ras


def save_skymap_fits_trim(frame, fitsfile, field, hdr=None, compress=False,
                          overwrite=True):
    """
    Save a trimmed version of the sky map to a FITS file.

    Parameters:
    - frame: A frame object containing the map data.
    - fitsfile (str): The path to the output FITS file.
    - field (str): The field name to be used for trimming.
    - hdr (astropy.io.fits.Header): Header to be included in the FITS file (optional).
    - compress (bool): If True, compresses the FITS file.
    - overwrite (bool): If True, overwrites the existing FITS file.
    """
    if frame.type != core.G3FrameType.Map:
        raise TypeError(f"Input map: {frame.type} must be a FlatSkyMap or HealpixSkyMap")

    ctype = None
    if compress is True:
        ctype = 'GZIP_2'
    elif isinstance(compress, str):
        ctype = compress

    # Get the T and weight frames
    T = frame['T']
    W = frame.get('Wpol', frame.get('Wunpol', None))

    data_sci = np.asarray(T)
    if W is not None:
        data_wgt = np.asarray(W.TT)
    logger.debug("Read data and weight")

    # Get the box size and center position to trim
    xc, yc, xsize, ysize = get_field_bbox(field, T.wcs)
    # Trim sci and wgt image using astropy cutour2D
    cutout_sci = Cutout2D(data_sci, (xc, yc), (ysize, xsize), wcs=T.wcs)
    cutout_wgt = Cutout2D(data_wgt, (xc, yc), (ysize, xsize), wcs=T.wcs)
    if hdr is None:
        hdr = maps.fitsio.create_wcs_header(T)
    hdr.update(cutout_sci.wcs.to_header())
    hdr_sci = copy.deepcopy(hdr)
    hdr_wgt = copy.deepcopy(hdr)
    hdr_wgt['ISWEIGHT'] = True

    hdul = astropy.io.fits.HDUList()
    if compress:
        logger.debug(f"Will compress using: {ctype} compression")
        hdu_sci = astropy.io.fits.CompImageHDU(
            data=cutout_sci.data,
            name='SCI',
            header=hdr_sci,
            compression_type=ctype)
        if W:
            hdu_wgt = astropy.io.fits.CompImageHDU(
                data=cutout_wgt.data,
                name='WGT',
                header=hdr_wgt,
                compression_type=ctype)
    else:
        hdu_sci = astropy.io.fits.ImageHDU(data=cutout_sci.data, header=hdr)
        if W:
            hdu_wgt = astropy.io.fits.ImageHDU(data=cutout_wgt.data, header=hdr)
    hdul.append(hdu_sci)
    hdul.append(hdu_wgt)
    hdul.writeto(fitsfile, overwrite=overwrite)
    del data_sci
    del data_wgt
    del hdr_sci
    del hdr_wgt
    del hdu_sci
    del hdu_wgt


def remove_units(frame, units):
    """
    Remove units from the frame, scaling the data accordingly.

    Parameters:
    - frame: A G3 frame object containing the data.
    - units: The unit to scale the data to.

    Returns:
    - frame: The modified G3 frame with units removed.
    """
    if frame.type != core.G3FrameType.Map:
        return frame

    if frame['T'].weighted:
        t_scale = units
    else:
        t_scale = 1./units
    w_scale = units * units
    for k in ['T', 'Q', 'U']:
        if k in frame:
            frame[k] = frame.pop(k) * t_scale
    for k in ['Wunpol', 'Wpol']:
        if k in frame:
            frame[k] = frame.pop(k) * w_scale

    return frame


def load_fits_stamp(fits_file):
    """
    Load 2-D images from SCI-n extensions in a multi-extension FITS file.

    Parameters:
        fits_file (str): Path to the FITS file.

    Returns:
        dict: A dictionary with OBSID as keys and 2-D numpy arrays as values.
        str: The ID value from the PRIMARY HDU header.
    """
    images = {}
    headers = {}
    with astropy.io.fits.open(fits_file) as hdul:
        # Get NFILES and ID from the primary header
        primary_header = hdul[0].header
        nfiles = primary_header['NFILES']
        id = primary_header['ID']
        band = primary_header['BAND']

        # Loop through the extensions to get SCI-n images
        for i in range(1, nfiles + 1):
            sci_ext_name = f"SCI_{i}"
            if sci_ext_name in hdul:
                obsid = hdul[sci_ext_name].header['OBSID']
                images[obsid] = hdul[sci_ext_name].data
                headers[obsid] = hdul[sci_ext_name].header
                headers[obsid]['SNRMAX'] = primary_header['SNRMAX']

    return images, headers, id, band


def load_fits_table(fits_table_file, target_id=None):
    """
    Load a FITS table and return the row matching the given 'id'.

    Parameters:
    - fits_table_file (str): Path to the FITS table file.
    - target_id (str): The 'id' to match in the table.

    Returns:
    - dict: A dictionary containing the matching row with columns 'id', 'dates_ave', 'flux_SCI', and 'flux_WGT'.
    """
    with astropy.io.fits.open(fits_table_file) as hdul:

        table_data = hdul[1].data
        band = hdul[1].header['BAND']
        ids = table_data['id']

        rows = []
        if target_id:
            target_ids = [target_id]
        else:
            target_ids = ids

        # Find the index where 'id' matches target_id
        for target_id in target_ids:
            match_index = np.where(ids == target_id)[0]
            if len(match_index) == 0:
                raise ValueError(f"ID {target_id} not found in {fits_table_file}")

            # Extract row data and return data
            row = table_data[match_index[0]]
            rows.append(row)
        return rows, band


def plot_stamps_lc(images_dict, headers_dict, lightcurve_dict,
                   obsmin=None, obsmax=None, format="png", outdir="", show=False):

    # Use ScalarFormatter with useMathText=True for non-scientific notation
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(False)

    bands = list(images_dict.keys())
    n_bands = len(bands)
    bands = du.sort_bands(bands)

    # Make sure that all dictionaries have same number of observations
    k = 0
    for band in bands:
        if k == 0:
            selected_IDs = images_dict[band].keys()
            n_images = len(selected_IDs)
            continue
        elif n_images != len(images_dict[band].keys()):
            raise ValueError(f"Mismatch of observations for band: {band}")
        k += 1
    # Select index for obsid range
    selected_IDs = np.asarray(list(selected_IDs))
    i1 = 0
    i2 = len(selected_IDs)

    # In case we have obsmin and obsmax
    if obsmin is not None:
        i1 = np.where(selected_IDs >= obsmin)[0][0]
        n_images = len(selected_IDs[i1:i2])
    if obsmax is not None:
        i2 = np.where(selected_IDs <= obsmax)[0][-1] + 1  # need to add +1 to slice
        n_images = len(selected_IDs[i1:i2])

    # Horizontal (4%) and vertical (15%) space for margins
    hmargin = 0.04
    vmargin = 0.15
    height_ratios = [1]*n_bands
    height_ratios.append(0.2)   # space between thumbnails and lightcurve
    height_ratios.append(n_bands)
    hscale = (1-2*hmargin)/(1-2*vmargin)
    fig = plt.figure(figsize=(n_images, (n_bands*2)*hscale))
    gs = fig.add_gridspec(n_bands+2, n_images, height_ratios=height_ratios,
                          left=hmargin, right=1-hmargin,
                          top=1-vmargin, bottom=vmargin,
                          hspace=0.05*hscale, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')

    # Loop over all of the files
    j = 0
    for band in bands:
        i = 0
        axs[j, 0].set_ylabel(f"{band}", size="medium")
        for ID in selected_IDs[i1:i2]:
            image_data = images_dict[band][ID]
            header = headers_dict[band][ID]
            date_beg = header["DATE-BEG"]
            obsid = header["OBSID"]
            snr = header['SNRMAX']
            t = Time(date_beg, format='isot')
            obstime = f"{t.mjd:.2f}"

            if i == 0:
                t0 = float(obstime)
            days = float(obstime) - t0
            days = f"{days:.2f}"

            # axs[j, i].axis('off')
            axs[j, i].imshow(image_data, origin='lower', cmap='viridis', vmin=-35, vmax=+35)
            axs[j, i].set_xticks([])
            axs[j, i].set_yticks([])
            axs[j, i].set_xticklabels([])
            axs[j, i].set_yticklabels([])
            # axs[-3, i].set_xlabel(str(days), size="small")
            axs[-3, i].set_xlabel(obstime, size="small")
            if j == 0:
                axs[0, i].set_title(obsid, size="small")
            i += 1
        j += 1

    # Remove axis for last row that will be dedicated to the LC
    for i in range(n_images):
        axs[j, i].set_axis_off()
        axs[j+1, i].set_axis_off()

    ax0 = fig.add_subplot(gs[n_bands, :])
    ax0.set_axis_off()
    ax1 = fig.add_subplot(gs[-1, :])
    fig.subplots_adjust(bottom=0.1)

    fcolor = {}
    fcolor['90GHz'] = 'red'
    fcolor['150GHz'] = 'blue'
    fcolor['220GHz'] = 'yellow'

    # Sorting gymastics for bands when 90Ghz is present
    bands = list(lightcurve_dict.keys())
    bands = du.sort_bands(bands)

    # Loop over all band in lightcurve
    for band in bands:
        id = lightcurve_dict[band]['id']
        dates_ave = lightcurve_dict[band]['dates_ave'][i1:i2]
        flux_SCI = lightcurve_dict[band]['flux_SCI'][i1:i2]
        flux_WGT = lightcurve_dict[band]['flux_WGT'][i1:i2]
        # obsids = lightcurve_dict[band]['obsids'][i1:i2]

        # Convert the first MJD date to a calendar date using Astropy
        # start_date = Time(dates_ave[0], format='mjd').iso
        # Shift dates_ave to start from the first date
        # dates_ave = [date - dates_ave[0] for date in dates_ave]

        # Plot with error bar
        ax1.errorbar(dates_ave, flux_SCI, yerr=1/np.sqrt(flux_WGT),
                     fmt='o', mfc=fcolor[band], mec='black',
                     elinewidth=1, ecolor=fcolor[band], capsize=1,
                     label=band)
        ax1.legend(loc='upper left', frameon=True)
        ax1.grid(color='black', linestyle='-.', linewidth=0.2)
        ax1.set_xlabel("Time[MJD]")
        ax1.set_ylabel('Flux [mJy]')

    # Set the formatter for both axes
    ax1.xaxis.set_major_formatter(formatter)
    ax1.yaxis.set_major_formatter(formatter)

    fig.suptitle(f"{id} | SN={snr:.1f}")
    if show:
        plt.show()
    du.create_dir(outdir)
    file = os.path.join(outdir, f"{id}.{format}")
    fig.savefig(file)
    print(f"Plot saved to file: {file}")
