from concurrent.futures import ProcessPoolExecutor
import copy
import importlib.metadata
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
import time
import types
import warnings

from astropy import units as u
from astropy.coordinates import FK5, search_around_sky, SkyCoord
from astropy.io import ascii
from astropy.stats import SigmaClip
from astropy.table import QTable, Table, vstack
from astropy.wcs import WCS
import fitsio
import magic
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np
import numpy.ma as ma
import photutils.background
from photutils.segmentation import SourceCatalog, SourceFinder
from photutils.utils.exceptions import NoDetectionsWarning
from scipy.stats import norm
from spt3g import core, maps, sources

from . import cutterlib

# to ignore astropy NoDetectionsWarning
warnings.filterwarnings("ignore", category=NoDetectionsWarning)


# Logger
LOGGER = logging.getLogger(__name__)
LOGGER.propagate = False

PPRINT_KEYS = ['label', 'xcentroid', 'ycentroid', 'sky_centroid', 'sky_centroid_dms',
               'max_value', 'snr_max', 'ellipticity', 'area']


# Set matplotlib logger at warning level to disengable from default logger
plt.set_loglevel(level='warning')

# Update objid, file naming convention
cutterlib.FITS_OUTNAME = "{outdir}/{objID}_{obsid}_{filter}.{ext}"
cutterlib.OBJ_ID = "{prefix}_J{ra}{dec}"
cutterlib.BASEDIR_OUTNAME = "{outdir}/{objID}"
cutterlib.FITS_LC_OUTNAME = "{outdir}/lightcurve_{filter}.{ext}"


class g3detect:
    """
    A class to run and manage transient detections on SPT (South Pole Telescope)
    files/frames.

    This class provides functionality to initialize the worker with necessary
    configurations, set up logging, prepare resources, and verify input files
    before starting the transient detection process.

    Attributes:
    config (types.SimpleNamespace): Configuration object containing input keys.
    logger (logging.Logger): Logger instance used for logging throughout the
    class methods.
    """

    def __init__(self, **keys):
        """
        Initializes the detect_3gworker class with the provided configuration keys.
        This method sets up the configuration, logging, prepares necessary resources,
        and checks the input files to ensure that everything is in place for the
        transient detection.

        Parameters:
        - **keys (dict): A variable number of keyword arguments representing
          configuration settings.
        """

        # Load the configurarion
        self.config = types.SimpleNamespace(**keys)

        # Start Logging
        self.logger = LOGGER
        self.setup_logging()

        os.makedirs(self.config.outdir, mode=0o755, exist_ok=True)

        # Check input files vs file list
        self.check_input_files()

    def setup_logging(self):
        """
        Sets up the logging configuration using `create_logger` and logs key info.
        Configures logging level, format, and other related settings based on the
        configuration object. Logs the start of logging and the version of the
        `mmblink` package.

        Raises:
        - ValueError: If the logger configuration is invalid or incomplete.
        """

        # Create the logger
        if self.logger is None:
            self.logger = logging.getLogger(__name__)
        create_logger(logger=self.logger, level=self.config.loglevel,
                      log_format=self.config.log_format,
                      log_format_date=self.config.log_format_date)
        self.logger.info(f"Logging Started at level:{self.config.loglevel}")

        try:
            # Get the version from the package metadata.
            version = importlib.metadata.version("mmblink")
        except importlib.metadata.PackageNotFoundError:
            # Get the hardcoded version with the risk of a circular import.
            from mmblink import __version__
            version = __version__

        self.logger.info(f"Running mmblink version: {version}")

    def check_input_files(self):
        """
        Check if the provided input is a single file containing a list of
        filenames or a list of individual files.

        - If the input is a single text file, it reads the filenames from the
          file and updates self.config.files with the list of files.
        - If the input is already a list of files, it logs the number of files detected.

        Attributes:
            self.config.files (list): A list of file paths or a single text file
            containing file paths.

        Returns:
        None
        """

        t = magic.Magic(mime=True)
        if len(self.config.files) == 1 and t.from_file(self.config.files[0]) == 'text/plain':
            self.logger.info(f"{self.config.files[0]} is a list of files")
            # Now read them in
            with open(self.config.files[0], 'r') as f:
                lines = []
                for line in f.read().splitlines():
                    if line[0] == "#":
                        continue
                    lines.append(line)
                # lines = f.read().splitlines()
            self.logger.info(f"Read: {len(lines)} input files")
            self.files = lines
        else:
            self.files = self.config.files.copy()
            self.logger.info(f"Detected list of [{len(self.files)}] files")

    def load_g3frames(self, filename, k):
        """
        Load data and metadata from a G3 file and extract relevant observation information.

        This method reads G3 frames from a file, extracts observation metadata
        (ObservationID and SourceName), and collects the relevant map frames based
        on the specified frequency bands. If necessary, missing metadata is set
        using previously extracted values.

        Parameters:
            filename (str): Path to the G3 file to be loaded.
             k (int): Index of the current file being processed, used for logging.

        Attributes:
            # self.config.bands (list): List of frequency bands to process (e.g., ['90GHz', '150GHz']).
            self.config.field (str): Optional field name to verify or substitute for SourceName.

        Returns:
            frames (list): A list of G3 frames containing map data for the specified bands.

        Notes:
           - Logs warnings if metadata (ObservationID or SourceName) cannot be extracted.
           - Skips frames not matching the configured frequency bands.
           - Sets ObservationID and SourceName if they are missing from the frame.
           """
        t0 = time.time()
        self.logger.info(f"Opening file: {filename}")
        self.logger.info(f"Doing: {k}/{len(self.files)} files")

        frames = []
        metadata_extracted = False
        for frame in core.G3File(filename):
            # Extract ObservationID and field (SourceName)
            if frame.type == core.G3FrameType.Observation:
                obsID = frame['ObservationID']
                try:
                    SourceName = frame['SourceName']
                    if self.config.field is not None and SourceName != self.config.field:
                        self.logger.warning(f"Extracted SourceName: {SourceName} doesn't match configuration")
                except KeyError:
                    if self.config.field is not None:
                        SourceName = self.config.field
                    self.logger.warning("Could not extract SourceName from Observation frame")
                metadata_extracted = True

            # check if obsID/SourceName are actualy in the frame
            elif frame.type == core.G3FrameType.Map and metadata_extracted is False:
                try:
                    obsID = frame['ObservationID']
                except KeyError:
                    self.logger.warning("Could not extract obsID from frame")
                try:
                    SourceName = frame['SourceName']
                except KeyError:
                    SourceName = ''
                    self.logger.warning("Could not extract SourceName from frame")

            # only read in the map
            if frame.type != core.G3FrameType.Map:
                continue
            if 'ObservationID' not in frame:
                self.logger.info(f"Setting ObservationID to: {obsID}")
                frame['ObservationID'] = obsID
            if 'SourceName' not in frame:
                self.logger.info(f"Setting SourceName to: {SourceName}")
                frame['SourceName'] = SourceName
            frames.append(frame)
        self.logger.info(f"Total metadata time: {elapsed_time(t0)} for: {filename}")
        return frames

    def load_g3frame_map(self, frame):
        """
        Load a G3 frame map into memory.

        Parameters:
            frame (core.G3Frame): The G3 frame containing map data.

        Returns:
            str: The observation key for the loaded frame.

        Notes:
            - Extracts metadata (ObservationID, SourceName, Band).
            - Creates a FITS header with WCS info and adds OBSID, FIELD, and BAND.
            - Removes weights and converts flux and weight maps to mJy.
            - Creates a flux mask based on the frame mask.
            - Adds observation ID to the list of loaded IDs.
        """
        # Get the metadata
        t0 = time.time()
        obsID = frame['ObservationID']
        band = frame["Id"]
        key = f"{obsID}_{band}"
        field = frame['SourceName']

        # Create a fits header for the frame map
        hdr = frame['T'].wcs.to_header()
        # Add OBSID, BAND and FIELD to the header (for thumbnails, etc)
        hdr['OBSID'] = (obsID, 'Observation ID')
        hdr['FIELD'] = (field, 'Name of Observing Field')
        hdr['BAND'] = (band, 'Observing Frequency')
        self.header[key] = hdr

        self.logger.info(f"Reading frame[Id]: {frame['Id']}")
        self.logger.debug(f"Reading frame: {frame}")
        self.logger.debug(f"ObservationID: {obsID}")
        self.logger.debug(f"Removing weights: {frame['Id']}")
        t1 = time.time()
        maps.RemoveWeights(frame, zero_nans=True)
        self.logger.info(f"Remove Weights time: {elapsed_time(t1)}")
        self.flux[key] = np.asarray(frame['T'])/core.G3Units.mJy
        self.flux_wgt[key] = np.asarray(frame['Wunpol'].TT)*core.G3Units.mJy*core.G3Units.mJy
        self.logger.debug(f"Min/Max Flux: {self.flux[key].min()} {self.flux[key].max()}")
        self.logger.debug(f"Min/Max Wgt: {self.flux_wgt[key].min()} {self.flux_wgt[key].max()}")
        # Now we exctract the mask
        try:
            # Zero is no data, and Ones is data
            g3_mask = frame["T"].to_mask()
            g3_mask_map = g3_mask.to_map()
            flux_mask = np.asarray(g3_mask_map)
            # Mask is True for pixels with no data.
            self.flux_mask[key] = flux_mask == 0
        except Exception as e:
            self.logger.warning(e.message)
            self.flux_mask[key] = None
        self.logger.info(f"Map from frame loaded for {obsID} {band}: {elapsed_time(t0)}")
        # Adding obsID to list of loaded list
        if obsID not in self.obsIDs:
            self.obsIDs.append(obsID)
        return key

    def add_obs_column_to_cat(self):
        """
        Add obs and obs_max columns to catalogs
        This needs to be done outside the MP call,
        otherwise the dictionaries are not updated as expected
        """
        for band in self.config.detect_bands:
            for obsid in self.cat[band].keys():
                self.logger.info(f"Adding obs/obs_max/band column for {band}:{obsid}")
                obsID = self.cat[band][obsid].meta['obsID']
                band = self.cat[band][obsid].meta['band']
                key = f"{obsID}_{band}"
                self.cat[band][obsid].add_column(np.array([obsID]*len(self.cat[band][obsid])), name='obs', index=0)
                self.cat[band][obsid].add_column(np.array([key]*len(self.cat[band][obsid])), name='obs_max', index=0)
                self.cat[band][obsid].add_column(np.array([band]*len(self.cat[band][obsid])), name='band', index=0)

    def run_detection_file(self, filename, k):
        """
        Run the task(s) for one g3file.
        The outputs are stored in self.cat and self.segm
        """
        t0 = time.time()
        # We need to setup logging again for MP
        if self.NP > 1:
            self.setup_logging()
        self.logger.info(f"Opening file: {filename}")
        self.logger.info(f"Doing: {k}/{len(self.files)} files")
        # Check if g3 or FITS file
        filetype = g3_or_fits(filename)
        self.logger.info(f"This file: {filename} is a {filetype} file")

        # We need to loop as each frame can contain more than one map (mult-band case)
        if filetype == "G3":
            # Load framea and extract keys
            frames = self.load_g3frames(filename, k)
            keys = [self.load_g3frame_map(frame) for frame in frames]
        elif filetype == "FITS":
            keys = self.load_fits_map(filename)

        # Loop over keys -- just one key for FITS files.
        for key in keys:
            # Here we store the files used (per band) to get cutouts later
            band = self.header[key]['BAND']
            # Call to detect_with_photutils per key. We will populate self.cat
            # and self.segm dictionary if sources are found.
            # We will only run detect_with_photutils_key if band in detection bands
            # Make sure we match the bands requested
            if band in self.config.detect_bands:
                self.logger.info(f"Running detection for {key}")
                self.detect_with_photutils_key(key)
            else:
                self.logger.info(f"Will not run detection for {key} -- not in detection bands")

        self.logger.info(f"Completed: {k}/{len(self.files)} files")
        self.logger.info(f"Total time: {elapsed_time(t0)} for: {filename}")

    def run_detection_files(self):
        """
        Run detection on all g3 files, using multiprocessing if NP > 1.
        """
        t0 = time.time()
        if self.NP > 1:
            self.logger.info("Running detection jobs with multiprocessing")
            self.run_detection_async()
            # self.run_detection_mp()
        else:
            self.logger.info("Running detection jobs serialy")
            self.run_detection_serial()

        # We add the obs and obs_max columns
        self.add_obs_column_to_cat()

        self.logger.info(f"Total time: {elapsed_time(t0)} for [run_detection_files]")

    def load_detection_files(self):
        self.logger.info("Loading detect_cat files for detections.")
        results = []
        for file in self.config.detect_cat:
            try:
                result = QTable.read(file, format="ascii.ecsv")
            except Exception as e:
                message = f"Occurred at file {file}."
                e.add_note(message)
                raise
            results.append(result)
        self.detect_catalogs = results
        return results

    def detect_all_sources(self):
        """Run detection on the files, loading each one modularly when needed.

        Returns
        -------
        results : list
            List of results in the same order as the files. Each element is the
            result of running detect_sources_in_file on that file.

        Raises
        ------
        Exception
            Re-raises any exception raised by a file, adding a note of the
            failing file.
        """
        if self.config.np > 1:
            self.logger.warning(
                "Multiprocessing is currently disabled. "
                "Running detection jobs serially."
            )
            return self.detect_all_sources_serial()
            # self.logger.info("Running detection jobs with multiprocessing")
            # return self.detect_all_sources_async()
        else:
            self.logger.info("Running detection jobs serially")
            return self.detect_all_sources_serial()

    def detect_all_sources_serial(self):
        """Find detections across all files in order with modular loading.

        If any file raises an exception during the detection process:
            - All remaining files are canceled.
            - The process pool is shut down.
            - The exception is re-raised with a note of the failing file.

        Returns
        -------
        results : list
            List of results in the same order as the files. Each element is the
            result of running detect_sources_in_file on that file.

        Raises
        ------
        Exception
            Re-raises any exception raised by a file, adding a note of the
            failing file.
        """
        results = []
        start_time = time.time()
        for i, file in enumerate(self.files):
            t0 = time.time()
            try:
                self.logger.info(f"Opening file: {file}")
                self.logger.info(f"Doing: {i}/{len(self.files)} files")
                result = detect_sources_in_file(file, self.config)
            except Exception as e:
                message = f"Occurred at file {file}."
                e.add_note(message)
                raise
            results.append(result)
            self.logger.info(f"Completed: {i}/{len(self.files)} files")
            self.logger.info(f"Total time: {elapsed_time(t0)} for: {file}")
        self.detect_catalogs = [
            catalog for catalog in results if catalog is not None
        ]
        self.logger.info(
            f"Total time: {elapsed_time(start_time)} for detecting all sources"
        )
        return results

    def detect_all_sources_async(self):
        """Find detections across all files in parallel with modular loading.

        If any file raises an exception during the detection process:
            - All remaining files are canceled.
            - The process pool is shut down.
            - The exception is re-raised with a note of the failing file.

        Returns
        -------
        results : list
            List of results in the same order as the files. Each element is the
            result of running detect_sources_in_file on that file.

        Raises
        ------
        Exception
            Re-raises any exception raised by a file, adding a note of the
            failing file.
        """
        # When workers is None, as many workers as processors are used.
        workers = None if self.config.np == 0 else self.config.np
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(detect_sources_in_file, file, self.config)
                for file in self.files
            ]
            results = []
            for index, future in enumerate(futures):
                try:
                    result = future.result()
                except Exception as e:
                    message = f"Occurred at file {self.files[index]}."
                    e.add_note(message)
                    # Futures that are already running cannot be canceled, so it
                    # may take some time for the program to fully complete even
                    # after an exception is raised.
                    for future in futures:
                        future.cancel()
                    raise
                results.append(result)
        self.detect_catalogs = [
            catalog for catalog in results if catalog is not None
        ]
        return results

    def match_dual_bands(self):
        """Match sources in the same observation detected in two bands.

        The list of detection catalogs stored in detect_catalogs is used for
        matching.

        Returns
        -------
        matched : dict or None
            Dictionary mapping obsID to a matched catalog of sources or None if
            matching could not occur.
        """

        if len(self.config.detect_bands) != 2:
            self.logger.info(f"Not enough bands: {self.config.detect_bands} to run dual match")
            return None

        catalogs_by_obsid = {}
        for catalog in self.detect_catalogs:
            catalog_list = catalogs_by_obsid.get(catalog.meta["obsID"])
            if catalog_list is None:
                catalogs_by_obsid[catalog.meta["obsID"]] = [catalog]
            else:
                catalog_list.append(catalog)

        matched_cat = {}
        for obsid, catalogs in catalogs_by_obsid.items():
            bands = [catalog.meta["band"] for catalog in catalogs]
            if set(bands) == set(self.config.detect_bands):
                cat1 = catalogs[0]
                cat2 = catalogs[1]
                band1 = cat1.meta["band"]
                band2 = cat2.meta["band"]
                self.logger.info(f"Attempting dual band match for obsID: {obsid} -- {band1} vs {band2}")
                matched = find_dual_detections(cat1, cat2)
                if matched is None:
                    continue
                matched_cat[obsid] = matched
            else:
                self.logger.debug(f"No dual match for {obsid}, bands: {bands}.")
        self.logger.info("---------------------------- Done dual band -------------------------")
        return matched_cat

    def collect_dual(self):
        # Function to collect and match sources in dual detection

        matched_dual = self.match_dual_bands()
        if len(matched_dual) == 0:
            self.logger.warning("No sources could be matched -- stopping here")
            return

        self.logger.info("Running unique centroids for dual matching per obsID")
        self.stacked_centroids = find_unique_centroids(
            matched_dual, separation=self.config.max_sep, plot=False
        )
        # Remove non repeat sources
        self.stacked_centroids = remove_non_repeat_sources(
            self.stacked_centroids, ncoords=self.config.nr
        )
        # Write catalogs with centroids
        self.write_centroids(self.stacked_centroids)

    def collect_single(self):
        # Function to collect and match sources in single_detection

        # Make a per band call to find_unique_centroids() in order to get
        # the unique centroids in each of the detection bands
        # Store the centroids in dict keyed to band.
        catalogs_by_band = {}
        for catalog in self.detect_catalogs:
            catalogs = catalogs_by_band.get(catalog.meta["band"])
            if catalogs is None:
                catalogs_by_band[catalog.meta["band"]] = {
                    catalog.meta["obsID"]: catalog
                }
            else:
                catalogs[catalog.meta["obsID"]] = catalog
        centroids = {}
        for band, catalog in catalogs_by_band.items():
            self.logger.info(f"Getting unique centroids for band: {band}")
            # Proceed only if we have any catalogs
            if len(catalog) > 0:
                centroids[band] = find_unique_centroids(
                    catalog, separation=self.config.max_sep, plot=False
                )
            else:
                self.logger.warning(f"Skipping band: {band} --  no catalogs")
        # And now get the unique/stacked centroids
        if len(centroids) == 0:
            self.logger.warning("Skipping stacked_centroids --  no catalogs")
            self.logger.warning("Will NOT write centroids")
            self.stacked_centroids = None
            return

        self.stacked_centroids = find_unique_centroids(
            centroids, separation=self.config.max_sep, plot=False
        )
        # Remove non repeat sources
        self.stacked_centroids = remove_non_repeat_sources(
            self.stacked_centroids, ncoords=self.config.nr
        )
        # Write catalogs with centroids and per band
        self.write_centroids(self.stacked_centroids)
        for band, catalog in centroids.items():
            self.write_centroids(catalog, band=band)

    def make_stamps_and_lighcurves(self):
        # Generate cutouts and repack stamps and lightcurve results
        if self.stacked_centroids is None:
            self.logger.warning("Will not make stamps or light curves: NO centroids")
            return
        self.run_cutouts(self.stacked_centroids)
        self.repack_lc()
        self.repack_stamps()

    def run_cutouts(self, cat):
        """
        Run cutouts for a catalog and a list of files to generate source cutouts
        and lightcurves.

        This function processes a catalog of sources, extracting their sky coordinates (RA and Dec),
        and uses the cutterlib to generate cutouts for each source. It supports lightcurve extraction,
        uniform coverage, and rejection of invalid cutouts. The resulting cutouts and other relevant
        information are stored in dictionaries and used to populate the configuration for further analysis.

        Parameters:
            cat (astropy.table.Table): The source catalog containing the sky
                coordinates (RA and Dec) for the sources.

        Returns:
            None: The function updates the instance's `cutout_names`, `lightcurve`,
                  and other relevant configuration attributes in-place.

        Example:
            >>> run_cutouts(catalog)

        Notes:
            - The cutout generation uses a predefined file naming convention
              and settings from `self.config`.
            - This function handles multiple files and bands, and it tracks
              the progress across files.
            - The cutouts are stored in dictionaries `cutout_dict`,
              `rejected_dict`, and `lightcurve_dict`.
        """

        cutout_dict = {}
        rejected_dict = {}
        lightcurve_dict = {}

        # Extract ra and dec from cat:
        ra = cat['sky_centroid'].ra.data
        dec = cat['sky_centroid'].dec.data
        snr_max = cat['snr_max'].data

        xsize = self.config.stamp_size  # arcmin
        ysize = self.config.stamp_size  # arcmin
        prefix = self.config.prefix
        outdir = self.config.outdir
        objID = None  # set to None so it's done automatically
        get_lightcurve = True
        get_uniform_coverage = False
        no_fits = False
        stage = False
        stage_path = '/tmp'
        stage_prefix = os.path.join(stage_path, 'spt3g_cutter-stage-')

        k = 1
        n_files = len(self.files)
        for file in self.files:
            counter = f"{k}/{n_files} files"
            ar = (file, ra, dec, cutout_dict, rejected_dict, lightcurve_dict)
            kw = {'xsize': xsize, 'ysize': ysize, 'units': 'arcmin', 'objID': objID,
                    'prefix': prefix, 'outdir': outdir, 'counter': counter,
                    'get_lightcurve': get_lightcurve,
                    'get_uniform_coverage': get_uniform_coverage,
                    'nofits': no_fits,
                    'stage': stage,
                    'stage_prefix': stage_prefix,
                    'obsid_names': True}
            names, pos, lc = cutterlib.fitscutter(*ar, **kw)
            cutout_dict.update(names)
            rejected_dict.update(pos)
            lightcurve_dict.update(lc)
            k += 1

        self.cutout_names = cutout_dict
        self.lightcurve = lightcurve_dict
        self.id_names = cutterlib.get_id_names(ra, dec, prefix)
        self.obs_dict = cutterlib.get_obs_dictionary(lightcurve_dict)
        # Pass the centroids to the class
        self.ra_centroid = ra
        self.dec_centroid = dec
        self.snr_max = snr_max

    def repack_lc(self):
        """
        Repack and write the lightcurve dictionary as a FITS table.

        This function repacks the lightcurve data stored in `self.lightcurve`
        for each band with observations and writes it to a FITS table. The
        output file is named according  to the predefined file naming convention
        and the appropriate lightcurve data is processed  using the
        `cutterlib.repack_lightcurve_band_filetype` function.

        Parameters:
            None

        Returns:
            None: The function writes the lightcurve data as FITS files.

        Example:
            >>> repack_lc()

        Notes:
            - The function updates the file naming convention using
              `cutterlib.FITS_LC_OUTNAME`.
            - It processes lightcurve data for each band and uses the
             `repack_lightcurve_band_filetype` function from `cutterlib`
             to write the FITS files.
        """
        for band in self.obs_dict.keys():
            file_type = 'None'
            cutterlib.repack_lightcurve_band_filetype(
                self.lightcurve,
                self.id_names,
                self.obs_dict,
                band,
                file_type,
                self.config,
            )

    def repack_stamps(self):
        """
        Repack individual FITS stamp files into a single multiple-HDU FITS file.

        This method iterates over a dictionary of cutout names and their corresponding
        bands. For each combination of cutout and band, it creates a multiple-HDU FITS
        file by concatenating the individual FITS files corresponding to that cutout and
        band. After the files are concatenated, the individual files are removed.

        The following operations are performed for each stamp and band:
        - A sorted list of input FITS filenames is generated.
        - The files are concatenated into one multiple-HDU FITS file.
        - The individual files are removed after concatenation.

        Parameters: None
        Returns: None

        Raises:
        - FileNotFoundError: If any of the input FITS files specified in `filenames` do not exist.
        - OSError: If there are any issues during file operations (e.g., permission issues during removal).
        """

        self.logger.info("Repacking stamps into single file per source")
        for k, stamp_name in enumerate(self.cutout_names):
            position = (self.ra_centroid[k], self.dec_centroid[k])
            snr_max = self.snr_max[k]
            for band in self.cutout_names[stamp_name].keys():
                fitsfile = f"{self.config.outdir}/{stamp_name}_{band}.fits"
                self.logger.debug(f"Combining into: {fitsfile}")
                filenames = self.cutout_names[stamp_name][band]
                # Sort the filenames -- will be sorted by obsID in the filename
                filenames.sort()
                concatenate_fits(filenames, fitsfile, stamp_name, band, position, snr_max)
                remove_files(filenames)

    def write_centroids(self, catalog, band=None):

        # Make a copy of the catalog, so that changes are not propagated
        CAT_KEYS = ['index', 'id', 'band', 'label', 'obs_max',
                    'xcentroid', 'ycentroid',
                    'sky_centroid', 'sky_centroid_dms',
                    'max_value', 'snr_max', 'ellipticity', 'area', 'ncoords']

        # Add id -- only if not already there
        if 'id' not in catalog.columns:
            # Extract ra and dec from cat:
            ra = catalog['sky_centroid'].ra.data
            dec = catalog['sky_centroid'].dec.data
            ids = cutterlib.get_id_names(ra, dec, self.config.prefix)
            catalog.add_column(ids, name='id', index=0)
        if band:
            catname = os.path.join(self.config.outdir, f"centroids_{band}.cat")
            msg = f"Wrote {band} catalog to: {catname}"
        else:
            catname = os.path.join(self.config.outdir, "centroids.cat")
            msg = f"Wrote combined catalog to: {catname}"
        ascii.write(catalog[CAT_KEYS],
                    catname, overwrite=True, format='fixed_width')
        self.logger.info(msg)


def get_fits_map(filename):
    """Read data and metadata from a FITS file of a map.

    Reads the header from the SCI HDU. For FITS files without EXTNAME, the SCI
    and WGT HDUs are assumed to be the first and second headers containing data
    respectively.

    Parameters
    ----------
    filename : str
        Path to the FITS file.

    Returns
    -------
    header : fitsio.header.FITSHDR
    hdus : dict
        A dictionary containing map data. The keys are:
        - "flux" : ndarray
        - "wgt" : ndarray
        - "mask" : ndarray
    """
    header_map, hdunum = cutterlib.get_headers_hdus(filename)
    sci_ind = hdunum['SCI']
    wgt_ind = hdunum['WGT']

    with fitsio.FITS(filename, "r") as fits:
        header = header_map['SCI']
        flux = fits[sci_ind].read()
        wgt = fits[wgt_ind].read()
        # Mask is True for pixels with no data.
        mask = wgt == 0
        return header, {"flux": flux, "wgt": wgt, "mask": mask}


def detect_sources_in_file(filename, config):
    """Catalog the source detections within a FITS file of a map.

    If the file's band is not in `config.detect_bands`, the file is skipped
    and (None, None) is returned.

    Parameters
    ----------
    filename : str
        Path to the file for detection.
    config : types.SimpleNamespace
        Configuration for processing files in the same format as configuration
        for `g3detect`.

    Returns
    -------
    catalog : astropy.table.Table or `None`
        A catalog of sources properties with each source in a row or `None` if
        no sources were found.
    """
    filetype = g3_or_fits(filename)
    if filetype  != "FITS":
        raise ValueError(
            f"Invalid file extension at {filename}. "
            "Only FITS files are currently supported."
        )
    LOGGER.info(f"This file: {filename} is {filetype} file")
    header, hdus = get_fits_map(filename)
    obsid = header["OBSID"]
    band = header["BAND"]
    if band in config.detect_bands:
        LOGGER.info((f"Running detection for {obsid}_{band}"))
        flux = hdus["flux"]
        wgt = hdus["wgt"]
        mask = hdus["mask"]
        wcs = WCS(header)
        plot_name = os.path.join(config.outdir, f"{obsid}_{band}_cat")
        plot_title = header["BAND"]
        field = header["FIELD"]
        segm_p, cat_p = detect_with_photutils(
            flux,
            wgt=wgt,
            mask=mask,
            nsigma_thresh=config.nsigma_thresh,
            npixels=config.npixels, wcs=wcs,
            rms2D=config.rms2D,
            rms2Dimage=config.rms2D_image,
            box=config.rms2D_box,
            plot=config.plot,
            plot_name=plot_name, plot_title=plot_title
        )

        if config.detect_negative:
            segm_n, cat_n = detect_with_photutils(
                -flux,
                wgt=wgt,
                mask=mask,
                nsigma_thresh=config.nsigma_thresh,
                npixels=config.npixels, wcs=wcs,
                rms2D=config.rms2D,
                box=config.rms2D_box,
            )
            cat_n['snr_max'] *= -1
            cats = [c for c in [cat_p, cat_n] if c is not None]
            cat = vstack(cats) if len(cats) > 0 else None
        else:
            cat = cat_p

        if cat is not None:
            # Remove objects that match the sources catalog for that field.
            if config.no_remove_source == False:
                cat = remove_objects_near_sources(
                    cat, field, config.point_source_file
                )

            # Filter sources with ellipticity greater than or equal to the cut.
            remove_mask = (
                (cat["ellipticity"] >= config.ell_cut) |
                np.isnan(cat["ellipticity"])
            )
            LOGGER.info(
                f"Removing {np.count_nonzero(remove_mask)} source(s) "
                f"with ellipticity >= {config.ell_cut}"
            )
            LOGGER.debug("Will remove:")
            if LOGGER.getEffectiveLevel() == logging.DEBUG:
                print(cat[PPRINT_KEYS][remove_mask])
            cat = cat[~remove_mask]
        if cat is None or len(cat) == 0:
            # Either no sources were detected or they were all filtered.
            LOGGER.info(f"Will not include catalog for {obsid}_{band}")
            return None
        else:
            cat.meta["band"] = band
            cat.meta["obsID"] = obsid
            cat.meta["field"] = field
            LOGGER.info(f"Adding obs/obs_max/band column for {band}:{obsid}")
            cat.add_column(obsid, name="obs", index=0)
            cat.add_column(f"{obsid}_{band}", name="obs_max", index=0)
            cat.add_column(band, name="band", index=0)
            if config.write_obscat:
                catname = os.path.join(config.outdir, f"{obsid}_{band}_full.cat")
                cat.write(catname, overwrite=True, format="ascii.ecsv")
                LOGGER.info(f"Wrote catalog to: {catname}")
            return cat
    else:
        LOGGER.info(
            f"Will not run detection for {obsid}_{band} -- "
            "not in detection bands"
        )
        return None


def check_index_ncoords_columns(catalog):
    # Make sure that index is present as a column
    if 'index' not in catalog.colnames:
        tblidx = np.arange(len(catalog)) + 1
        catalog.add_column(tblidx, name='index', index=0)
    if 'ncoords' not in catalog.columns:
        ncoords = np.ones(len(catalog), dtype='int')
        catalog.add_column(ncoords, name='ncoords')

    return catalog


def remove_non_repeat_sources(catalog, ncoords=1):
    # Remove entries from catalog with ncoords <= nr
    if ncoords > 1:
        inds = np.where(catalog['ncoords'] >= ncoords)[0]
        catsize = len(catalog)
        nk = len(inds)  # N keep
        nr = catsize - nk  # N remove
        if nr > 0:
            LOGGER.info(f"Removing {nr} sources with ncoords < {ncoords}")
            cutcat = catalog[inds]
        else:
            cutcat = catalog
    else:
        LOGGER.warning(f"Will not remove non-repeats ncoords <= 1: ncoords: {ncoords}")
        cutcat = catalog
    return cutcat


def concatenate_fits(input_files, output_file, id, band, position, snr_max):
    """
    Concatenates a list of FITS files into a single multi-HDU FITS file.
    Each input FITS file has two extensions: SCI (Science) and WGT (Weight).

    Parameters:
    input_files (list of str): List of input FITS file paths.
    output_file (str): Path to the output multi-HDU FITS file.
    id (str): Name of stamp to include in the PRIMARY header.
    band (str): Band information to include in the PRIMARY header.
    """
    # Create a new FITS file for output
    with fitsio.FITS(output_file, 'rw', clobber=True) as fits_out:
        # Create PRIMARY header
        primary_header = {
            'ID': id,
            'BAND': band,
            'SNRMAX': snr_max,
            'NFILES': len(input_files),
            'RA': position[0],
            'DEC': position[1],
            'COMMENT': 'Concatenated FITS file'
        }
        fits_out.write(None, header=primary_header)
        for idx, infile in enumerate(input_files):
            with fitsio.FITS(infile, 'r') as fits_in:

                # Auto-discover HDU indices for SCI and WGT extensions
                sci_hdu = next(i for i, hdu in enumerate(fits_in) if hdu.get_extname() == "SCI")
                wgt_hdu = next(i for i, hdu in enumerate(fits_in) if hdu.get_extname() == "WGT")

                # Read the SCI extension
                sci_data = fits_in[sci_hdu].read()
                sci_header = fits_in[sci_hdu].read_header()
                fits_out.write(sci_data, header=sci_header, extname=f"SCI_{idx+1}")

                # Read the WGT extension
                wgt_data = fits_in[wgt_hdu].read()
                wgt_header = fits_in[wgt_hdu].read_header()
                fits_out.write(wgt_data, header=wgt_header, extname=f"WGT_{idx+1}")

        LOGGER.debug(f"Successfully created {output_file} with {len(input_files) * 2} extensions.")


def remove_files(filelist, remove_parents=True):
    """
    Removes all (FITS) files from the given list of file paths and optionally deletes
    their parent directories if they become empty.

    Parameters:
    - filelist (list): A list of file paths to FITS files.
    - remove_parents (bool): If True, removes parent directories if they become empty.

    Returns:
    - None
    """
    parent_dirs = set()  # Track parent directories to check later

    for file_path in filelist:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                LOGGER.debug(f"Successfully removed: {file_path}")

                # Collect parent directory path
                parent_dir = os.path.dirname(file_path)
                if parent_dir:
                    parent_dirs.add(parent_dir)
            else:
                LOGGER.warning(f"File does not exist: {file_path}")
        except Exception as e:
            LOGGER.error(f"Error removing {file_path}: {e}")

    # Optionally remove parent directories if they are empty
    if remove_parents:
        for parent_dir in parent_dirs:
            try:
                if os.path.isdir(parent_dir) and not os.listdir(parent_dir):  # Check if empty
                    os.rmdir(parent_dir)
                    LOGGER.debug(f"Removed empty directory: {parent_dir}")
            except Exception as e:
                LOGGER.error(f"Error removing directory {parent_dir}: {e}")

    LOGGER.debug("Files removed" + (", including empty directories" if remove_parents else ""))


def configure_logger(logger, logfile=None, level=logging.NOTSET, log_format=None, log_format_date=None):
    """
    Configure an existing logger with specified settings. Sets the format,
    logging level, and handlers for the given logger. If a logfile is provided,
    logs are written to both the console and the file with rotation. If no log
    format or date format is provided, default values are used.

    Parameters:
    - logger (logging.Logger): The logger to configure.
    - logfile (str, optional): Path to the log file. If `None`, logs to the console.
    - level (int): Logging level (e.g., `logging.INFO`, `logging.DEBUG`).
    - log_format (str, optional): Log message format (default is detailed format with function name).
    - log_format_date (str, optional): Date format for logs (default is `'%Y-%m-%d %H:%M:%S'`).
    """
    # Define formats
    if log_format:
        FORMAT = log_format
    else:
        FORMAT = '[%(asctime)s.%(msecs)03d][%(levelname)s][%(name)s][%(funcName)s] %(message)s'
    if log_format_date:
        FORMAT_DATE = log_format_date
    else:
        FORMAT_DATE = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(FORMAT, FORMAT_DATE)

    # Need to set the root logging level as setting the level for each of the
    # handlers won't be recognized unless the root level is set at the desired
    # appropriate logging level. For example, if we set the root logger to
    # INFO, and all handlers to DEBUG, we won't receive DEBUG messages on
    # handlers.
    logger.setLevel(level)

    handlers = []
    # Set the logfile handle if required
    if logfile:
        fh = RotatingFileHandler(logfile, maxBytes=2000000, backupCount=10)
        fh.setFormatter(formatter)
        fh.setLevel(level)
        handlers.append(fh)
        logger.addHandler(fh)

    # Set the screen handle
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.setLevel(level)
    handlers.append(sh)
    logger.addHandler(sh)
    return


def create_logger(logger=None, logfile=None, level=logging.NOTSET, log_format=None, log_format_date=None):
    """
    Configures and returns a logger with specified settings.
    Sets up logging based on provided level, format, and output file. Can be
    used for both `setup_logging` and other components.

    Parameters:
    - logger (logging.Logger, optional): The logger to configure. If `None`, a new logger
      is created.
    - logfile (str, optional): Path to the log file. If `None`, logs to the console.
    - level (int): Logging level (e.g., `logging.INFO`, `logging.DEBUG`).
    - log_format (str, optional): Format for log messages (e.g., `'%(asctime)s - %(message)s'`).
    - log_format_date (str, optional): Date format for logs (e.g., `'%Y-%m-%d %H:%M:%S'`).

    Returns:
    logging.Logger: The configured logger instance.

    Raises:
    - ValueError: If the log level or format is invalid.
    """

    if logger is None:
        logger = logging.getLogger(__name__)
    configure_logger(logger, logfile=logfile, level=level,
                     log_format=log_format, log_format_date=log_format_date)
    logging.basicConfig(handlers=logger.handlers, level=level)
    logger.propagate = False
    logger.info(f"Logging Started at level:{level}")
    return logger


def elapsed_time(t1, verb=False):
    """
    Returns the time elapsed between t1 and the current time.

    Optionally, prints the formatted elapsed time.

    Parameters:
    - t1 (float): The initial time (in seconds).
    - verb (bool, optional): If `True`, prints the formatted elapsed time. Default is `False`.

    Returns:
    - str: The elapsed time as a string in the format "Xm XX.XXs", where X is the minutes and seconds.

    Example:
    >>> elapsed_time(1627387200)
    '5m 12.34s'
    """
    t2 = time.time()
    stime = "%dm %2.2fs" % (int((t2-t1)/60.), (t2-t1) - 60*int((t2-t1)/60.))
    if verb:
        print("Elapsed time: {}".format(stime))
    return stime


def create_dir(dirname):
    """
    Safely attempts to create a directory.

    If the directory does not exist, it is created with permissions `0o755`. If there is an
    error during directory creation, an exception is raised.

    Parameters:
    - dirname (str): The path to the directory to create.

    Returns:
     None
    """
    LOGGER.info(f"Creating directory: {dirname}")
    os.makedirs(dirname, mode=0o755, exist_ok=True)


def find_dual_detections(t1, t2, separation=20, plot=False):
    """
    Identifies matching sources between two catalogs (dual band match) based on
    their sky coordinates.
    The function compares two catalogs of detected sources and identifies
    matching sources within a given separation threshold. It then returns a
    catalog with updated centroid information, including both the average
    positions and maximum flux values for matched sources.

    This function performs the following steps:
    1. Ensures that both catalogs have the same observation ID.
    2. Finds matching objects based on sky coordinates, using a separation threshold.
    3. Computes the average positions (both in sky and pixel coordinates) for matched sources.
    4. Updates the centroid catalog with new position information and additional source properties
       (e.g., max flux, eccentricity).
    5. Logs the update process with debugging information.

    Parameters:
    - t1 (Table): The first catalog containing detected sources.
    - t2 (Table): The second catalog containing detected sources.
    - separation (float, optional): The maximum separation (in arcseconds) to consider a match between sources.
      Default is 20 arcseconds.
    - plot (bool, optional): If `True`, generate a plot for visualizing the matched sources (this feature is not
      implemented in the function).

    Returns:
    - Table: A catalog with updated centroid information, which includes the following columns:
      - `sky_centroid`: Sky coordinates of the matched sources (in FK5 frame).
      - `xcentroid`: Pixel coordinates of the matched sources.
      - `ycentroid`: Pixel coordinates of the matched sources.
      - `ncoords`: Number of coordinates used for averaging (always 2 for dual-band matches).
      - `obs_max`: Observation ID corresponding to the maximum flux for each source.
      - `max_value`: Maximum flux value for each matched source.
      - `sky_centroid_dms`: Sky coordinates in HMS/DMS format.
      - Additional columns for source properties, such as `eccentricity`, `elongation`, `ellipticity`, and `area`.

    Raises:
    - ValueError: If the observation IDs in the two catalogs do not match.

    Example:
    >>> find_dual_detections(catalog1, catalog2, separation=15)
    """
    logger = LOGGER
    max_sep = separation*u.arcsec
    stacked_centroids = None
    # Ensure both catalogs have the same obsID
    if t1.meta['obsID'] != t2.meta['obsID']:
        raise ValueError("values for obsID are not the same")

    obsID = t1.meta['obsID']
    band1 = t1.meta['band']
    band2 = t2.meta['band']
    cat1 = t1['sky_centroid']
    cat2 = t2['sky_centroid']
    labelID = f"{obsID}_{band1}_{band2}"
    logger.debug(f"Dual band match for: {labelID}")
    logger.debug(f"N in cat1: {len(cat1)} cat2: {len(cat2)}")

    # Find matching objects
    idxcat1, idxcat2, d2d, _ = cat2.search_around_sky(cat1, max_sep)

    # Proceed only if we have matches, otherwise return None
    if len(idxcat1) == 0:
        logger.info(f"*** No matches for: {labelID} ***")
        return None
    else:
        logger.info(f"*** Found {len(idxcat1)} matches for {labelID} ***")

    # Concatenate matched positions
    xx_sky = np.array([t1[idxcat1]['sky_centroid'].ra.data, t2[idxcat2]['sky_centroid'].ra.data])
    yy_sky = np.array([t1[idxcat1]['sky_centroid'].dec.data, t2[idxcat2]['sky_centroid'].dec.data])
    xx_pix = np.array([t1[idxcat1]['xcentroid'].data, t2[idxcat2]['xcentroid'].data])
    yy_pix = np.array([t1[idxcat1]['ycentroid'].data, t2[idxcat2]['ycentroid'].data])

    # Get the average positions
    xc_sky = np.mean(xx_sky, axis=0)
    yc_sky = np.mean(yy_sky, axis=0)
    xc_pix = np.mean(xx_pix, axis=0)
    yc_pix = np.mean(yy_pix, axis=0)
    ncoords = [2]*len(xc_sky)
    tblidx = np.arange(len(xc_sky)) + 1

    # Get the ids with max value
    value_max = np.array([t1[idxcat1]['max_value'], t2[idxcat2]['max_value']])
    obs_value = np.array([t1[idxcat1]['obs'], t2[idxcat2]['obs']])
    max_value_max = value_max.max(axis=0)
    idmax = value_max.argmax(axis=0)
    obs_max = obs_value.T[0][idmax]

    # We based our table on t1 and update with (some) averages with positions
    stacked_centroids = t1[idxcat1]
    # Before Update
    logger.debug("Before Update")
    t = stacked_centroids['label', 'xcentroid', 'ycentroid', 'sky_centroid_dms',
                          'obs', 'obs_max', 'max_value',
                          'eccentricity', 'elongation', 'ellipticity', 'area']
    logger.debug(f"\n{t}\n")
    # Update centroids with averages
    # Create a Skycoord object
    coords = SkyCoord(xc_sky, yc_sky, frame=FK5, unit='deg')
    if 'index' not in stacked_centroids.colnames:
        stacked_centroids.add_column(tblidx, name='index', index=0)
    else:
        stacked_centroids['index'] = tblidx
    stacked_centroids['sky_centroid'] = coords
    stacked_centroids['xcentroid'] = xc_pix
    stacked_centroids['ycentroid'] = yc_pix
    stacked_centroids['ncoords'] = ncoords
    stacked_centroids['obs_max'] = obs_max
    stacked_centroids['max_value'] = max_value_max
    stacked_centroids['max_value'].info.format = '.2f'
    stacked_centroids['xcentroid'].info.format = '.2f'
    stacked_centroids['ycentroid'].info.format = '.2f'
    stacked_centroids['sky_centroid_dms'] = stacked_centroids['sky_centroid'].to_string('hmsdms', precision=0)
    stacked_centroids.add_index('index')
    stacked_centroids.meta['band'] = f"{band1}_{band2}"

    logger.debug("After Update[find_dual_detections]")
    logger.debug("#### stacked_centroids ####")
    t = stacked_centroids['label', 'xcentroid', 'ycentroid', 'sky_centroid_dms',
                          'obs', 'obs_max', 'max_value', 'ncoords',
                          'eccentricity', 'elongation', 'ellipticity', 'area']
    logger.debug(f"\n{t}")
    logger.debug("#### ---- ###")

    return stacked_centroids


def find_unique_centroids(table_centroids, separation=20, plot=False):
    """
    Finds unique centroids between multiple catalogs by matching objects within a given
    separation threshold and stacking the resulting centroid information.

    Parameters:
    - table_centroids (dict): Dictionary where the keys are catalog labels and the values
                               are the catalogs containing the centroid data.
    - separation (float, optional): Maximum separation (arcseconds) to consider a match.
                                    Default is 20.
    - plot (bool, optional): If `True`, generate a plot (not implemented here). Default is `False`.

    Returns:
    - Table: A catalog with updated centroid information and matched sources, stacked across
             all catalogs.
    Notes:
    - The function operates on multiple catalogs provided in the `table_centroids` dictionary.
      It stacks matched centroids, averages their positions, and updates the catalog with the
      resulting information. If objects are unmatched, they are appended to the stacked catalog.
    """
    logger = LOGGER
    max_sep = separation*u.arcsec
    stacked_centroids = None
    labelIDs = list(table_centroids.keys())
    if len(labelIDs) < 2:
        logger.warning("Will not run find_unique_centroids() -- < 2 catalogs to match!")
        logger.warning(f"labelIDs: {labelIDs}")
        stacked_centroids = copy.deepcopy(table_centroids[labelIDs[0]])
        stacked_centroids = check_index_ncoords_columns(stacked_centroids)
        # Return the 1st and only element in the dictionary -- as the merged centroids
        return stacked_centroids

    for k in range(len(labelIDs)-1):
        # Select current and next table IDs
        label1 = labelIDs[k]
        label2 = labelIDs[k+1]
        logger.info(f"Doing: {k+1}/{len(labelIDs)-1}")

        # Extract the catalogs (i.e. SkyCoord objects) for search_around_sky
        # and make shorcuts of tables
        # For k > 0 we used the stacked/combined catalog
        if k == 0:
            t1 = copy.deepcopy(table_centroids[label1])
            cat1 = t1['sky_centroid']
        else:
            t1 = copy.deepcopy(stacked_centroids)
            cat1 = t1['sky_centroid']
        t2 = copy.deepcopy(table_centroids[label2])
        cat2 = t2['sky_centroid']

        # Remove 'obs' from tables if present
        if 'obs' in t1.columns:
            t1.remove_column('obs')
        if 'obs' in t2.columns:
            t2.remove_column('obs')

        # Find matching objects to avoid duplicates
        idxcat1, idxcat2, d2d, _ = cat2.search_around_sky(cat1, max_sep)
        # Define idxnew, the objects not matched in table2/cat2 that need to be appended
        n2 = len(cat2)
        idxall = np.arange(n2)
        idxnew2 = np.delete(idxall, idxcat2)

        # Only for the first iteration we append agaist t1, after that we use the output
        if k == 0:
            xx_sky = stack_cols_lists(t1['sky_centroid'].ra.data, t2['sky_centroid'].ra.data, idxcat1, idxcat2,)
            yy_sky = stack_cols_lists(t1['sky_centroid'].dec.data, t2['sky_centroid'].dec.data, idxcat1, idxcat2)
            xx_pix = stack_cols_lists(t1['xcentroid'].data, t2['xcentroid'].data, idxcat1, idxcat2)
            yy_pix = stack_cols_lists(t1['ycentroid'].data, t2['ycentroid'].data, idxcat1, idxcat2)
            value_max = stack_cols_lists(t1['max_value'].data, t2['max_value'].data, idxcat1, idxcat2, pad=True)
            obs_max = stack_cols_lists(t1['obs_max'].data, t2['obs_max'].data, idxcat1, idxcat2, pad=True)
            snr_max = stack_cols_lists(t1['snr_max'].data, t2['snr_max'].data, idxcat1, idxcat2, pad=True)
            # If ncoords already exists in the columns we will stack them
            if 'ncoords' in t1.columns and 'ncoords' in t2.columns:
                ncoords = stack_cols_lists(t1['ncoords'].data, t2['ncoords'].data, idxcat1, idxcat2)
        else:
            xx_sky = stack_cols_lists(xx_sky, t2['sky_centroid'].ra.data, idxcat1, idxcat2)
            yy_sky = stack_cols_lists(yy_sky, t2['sky_centroid'].dec.data, idxcat1, idxcat2)
            xx_pix = stack_cols_lists(xx_pix, t2['xcentroid'].data, idxcat1, idxcat2)
            yy_pix = stack_cols_lists(yy_pix, t2['ycentroid'].data, idxcat1, idxcat2)
            value_max = stack_cols_lists(value_max, t2['max_value'].data, idxcat1, idxcat2, pad=True)
            obs_max = stack_cols_lists(obs_max, t2['obs_max'].data, idxcat1, idxcat2, pad=True)
            snr_max = stack_cols_lists(snr_max, t2['snr_max'].data, idxcat1, idxcat2, pad=True)
            if 'ncoords' in t2.columns:
                ncoords = stack_cols_lists(ncoords, t2['ncoords'].data, idxcat1, idxcat2)

        # Here we update the max_values and obs_max label
        # We make them np.array so we can operate on them
        value_max = np.array(value_max)
        snr_max = np.array(snr_max)
        obs_max = np.array(obs_max)
        idmax = value_max.argmax(axis=1)
        # We store them back in the same arrays/lists
        value_max = value_max.max(axis=1)
        obs_max = [obs_max[i][idmax[i]] for i in range(len(idmax))]
        snr_max = [snr_max[i][idmax[i]] for i in range(len(idmax))]

        # If we have unmatched objects in cat2 (i.e. idxnew has elements), we append these
        if len(idxnew2) > 0:
            # inherit metadata from t1
            new_stack = vstack([t1, t2[idxnew2]])
            stacked_centroids = new_stack
            logger.info(f"{label1}-{label2} Stacked")
        else:
            stacked_centroids = t1
            logger.info(f"{label1}-{label2} No new positions to add")

        # Get the average positions so far
        xc_pix = mean_list_of_list(xx_pix)
        yc_pix = mean_list_of_list(yy_pix)
        xc_sky = mean_list_of_list(xx_sky)
        yc_sky = mean_list_of_list(yy_sky)
        # Update the number of coordinates points we have so far
        if 'ncoords' in t1.columns and 'ncoords' in t2.columns:
            ncoords = [sum(x) for x in ncoords]
        else:
            ncoords = [len(x) for x in xx_pix]
        tblidx = np.arange(len(xc_sky)) + 1

        # Update centroids with averages
        # Create a Skycoord object
        coords = SkyCoord(xc_sky, yc_sky, frame=FK5, unit='deg')
        if 'index' not in stacked_centroids.colnames:
            stacked_centroids.add_column(tblidx, name='index', index=0)
        else:
            stacked_centroids['index'] = tblidx

        stacked_centroids['sky_centroid'] = coords
        stacked_centroids['xcentroid'] = xc_pix
        stacked_centroids['ycentroid'] = yc_pix
        stacked_centroids['ncoords'] = ncoords
        stacked_centroids['obs_max'] = obs_max
        stacked_centroids['snr_max'] = snr_max
        stacked_centroids['max_value'] = value_max
        stacked_centroids['max_value'].info.format = '.2f'
        stacked_centroids['xcentroid'].info.format = '.2f'
        stacked_centroids['ycentroid'].info.format = '.2f'
        stacked_centroids['snr_max'].info.format = '.2f'
        stacked_centroids.add_index('index')
        logger.debug(f"centroids Done for {label1}")

    return stacked_centroids


def find_repeating_sources(cat, separation=20, plot=False, outdir=None):
    """
    Matches sources in consecutive catalogs that appear in at least two consecutive catalogs,
    with a specified separation threshold, and calculates the mean centroid of matched sources.

    Parameters:
    - cat (dict): Dictionary containing catalogs in astropy Table format, with each key
                  corresponding to a catalog of sources.
    - separation (float, optional): Maximum separation (arcseconds) to consider a match.
                                    Default is 20.
    - plot (bool, optional): If `True`, generates plots for the matched sources and saves them
                              in the specified output directory. Default is `False`.
    - outdir (str, optional): Directory to save the plot if `plot=True`. Default is `None`.

    Returns:
    - dict: A dictionary where the keys are concatenated labels of the matched catalogs
            (e.g., 'scan1_scan2') and the values are astropy Tables with matched centroid positions
            and other related information.

    Notes:
    - The function compares consecutive catalogs and finds sources with positions that match
      within the specified separation threshold. It calculates the mean centroid for each matched
      pair of catalogs, storing the results in a dictionary. If `plot` is enabled, plots are generated
      showing the matched sources and their positions.
    """

    max_sep = separation*u.arcsec
    table_centroids = {}  # Table with centroids
    scans = list(cat.keys())
    logger = LOGGER
    logger.info("++++++++ Starting Match Loop for repeating sources ++++++++++")
    logger.info(f"scans: {scans}")
    for k in range(len(scans)-1):
        logger.info(scans[k])
        scan1 = scans[k]
        scan2 = scans[k+1]
        cat1 = cat[scan1]['sky_centroid']
        cat2 = cat[scan2]['sky_centroid']
        n1 = len(cat[scan1])
        n2 = len(cat[scan2])
        labelID = f"{scan1}_{scan2}"
        logger.info("==============================")
        logger.info(f"Doing {scan1} vs {scan2}")
        logger.info(f"N in cat1: {n1} cat2: {n2}")

        # Match method using search_around_sky
        idxcat1, idxcat2, d2d, _ = cat2.search_around_sky(cat1, max_sep)
        logger.info(f"N matches: {len(idxcat1)} -- {len(idxcat2)}")

        if len(idxcat1) == 0:
            logger.info(f"*** No matches for: {labelID} ***")
            continue

        # Get the mean centroid from the matched catalogs
        xx_sky = np.array([cat[scan1][idxcat1]['sky_centroid'].ra, cat[scan2][idxcat2]['sky_centroid'].ra])
        yy_sky = np.array([cat[scan1][idxcat1]['sky_centroid'].dec, cat[scan2][idxcat2]['sky_centroid'].dec])
        xx_pix = np.array([cat[scan1][idxcat1]['xcentroid'], cat[scan2][idxcat2]['xcentroid']])
        yy_pix = np.array([cat[scan1][idxcat1]['ycentroid'], cat[scan2][idxcat2]['ycentroid']])
        xc_sky = np.mean(xx_sky, axis=0)
        yc_sky = np.mean(yy_sky, axis=0)
        xc_pix = np.mean(xx_pix, axis=0)
        yc_pix = np.mean(yy_pix, axis=0)
        ncoords = [2]*len(xc_sky)
        label_col = [labelID]*len(xc_sky)
        tblidx = np.arange(len(xc_sky)) + 1

        # Get the ids with max value
        max_value = np.array([cat[scan1][idxcat1]['max_value'], cat[scan2][idxcat2]['max_value']])
        obs_value = np.array([cat[scan1][idxcat1]['obs'], cat[scan2][idxcat2]['obs']])
        max_value_max = max_value.max(axis=0)
        idmax = max_value.argmax(axis=0)
        obs_max = obs_value.T[0][idmax]

        # Create a Skycoord object
        coords = SkyCoord(xc_sky, yc_sky, frame=FK5, unit='deg')
        table_centroids[labelID] = Table([tblidx, label_col, coords, xc_pix, yc_pix, obs_max, max_value_max, ncoords],
                                         names=('index', 'labelID', 'sky_centroid', 'xcentroid', 'ycentroid',
                                                'obs_max', 'max_value', 'ncoords'))
        table_centroids[labelID]['xcentroid'].info.format = '.2f'
        table_centroids[labelID]['ycentroid'].info.format = '.2f'
        table_centroids[labelID]['max_value'].info.format = '.2f'
        logger.info(f"centroids Done for: {labelID}")
        print(table_centroids[labelID])
        if plot:
            plot_name = os.path.join(outdir, f"{labelID}_match")
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

            # Set limits for centroids
            xmin = min(cat[scan1]['xcentroid'].min(), cat[scan2]['xcentroid'].min())
            ymin = min(cat[scan1]['ycentroid'].min(), cat[scan2]['ycentroid'].min())
            xmax = max(cat[scan1]['xcentroid'].max(), cat[scan2]['xcentroid'].max())
            ymax = max(cat[scan1]['ycentroid'].max(), cat[scan2]['ycentroid'].max())
            dx = 0.2*(xmax-xmin)
            dy = 0.2*(ymax-ymin)
            xmin = xmin-dx
            xmax = xmax+dx
            ymin = ymin-dy
            ymax = ymax+dy

            x = cat[scan1]['xcentroid']
            y = cat[scan1]['ycentroid']
            ax1.scatter(x, y, marker='o', c='red')
            ax1.set_xlabel('x[pixels]')
            ax1.set_ylabel('y[pixels]')
            ax1.set_xlim(xmin, xmax)
            ax1.set_ylim(ymin, ymax)
            ax1.set_title(f"{scan1}")

            x = cat[scan2]['xcentroid']
            y = cat[scan2]['ycentroid']
            ax2.scatter(x, y, marker='o', c='blue')
            ax2.set_xlabel('x[pixels]')
            ax2.set_ylabel('y[pixels]')
            ax2.set_xlim(xmin, xmax)
            ax2.set_ylim(ymin, ymax)
            ax2.set_title(f"{scan2}")

            x = table_centroids[labelID]['xcentroid']
            y = table_centroids[labelID]['ycentroid']
            ax3.scatter(x, y, marker='o', c='k')
            ax3.set_xlabel('x[pixels]')
            ax3.set_ylabel('y[pixels]')
            ax3.set_xlim(xmin, xmax)
            ax3.set_ylim(ymin, ymax)
            ax3.set_title(f"{scan1} - {scan2}")
            plt.savefig(f"{plot_name}.pdf")
            plt.close()

    logger.info("Done Matching Loop for repeating sources")
    logger.info("+++++++++++++++++++++++++++++")
    return table_centroids


def stack_cols_lists(c1, c2, ix1, ix2, pad=False, noNew=False):

    """
    Custom function to stack two one-dimensional columns containing lists.
    The function combines elements from two columns (`c1` and `c2`) based on
    matching indices (`ix1` and `ix2`) and allows for additional customization
    such as padding or skipping new elements.

    Parameters:
    - c1 (list or 1D numpy array): The first list (or list of lists) to be processed.
    - c2 (list or 1D numpy array): The second list (or list of lists) to be processed.
    - ix1 (list of int): List of indices of `c1` that match with `c2`.
    - ix2 (list of int): List of indices of `c2` that match with `c1`.

    Options:
    - pad (bool, optional): If `True`, pads lists with a single element to have two elements.
      Default is `False`.
    - noNew (bool, optional): If `True`, prevents the addition of new elements from `c2` to `c1`.
      Default is `False`.

    Returns:
    - list: A list where elements of `c1` and `c2` are stacked together based on the matching indices.
            Each element is guaranteed to be a list, even if it originally wasn't.
    Raises:
    - Exception: If the lengths of `ix1` and `ix2` do not match, an exception is raised.
    - Exception: If `c1` or `c2` cannot be cast to a list.

    Notes:
    - The function first augments the lists in `c1` with the matching elements from `c2` based on `ix1` and `ix2`.
    - If `noNew` is `True`, the function stops after augmenting the lists and does not append any new items from `c2`.
    - If `pad` is `True`, lists with only one element will be padded to contain two identical elements.
    """

    # Get elements that are not in c1 but not c2
    ixall1 = np.arange(len(c1))
    ixnew1 = np.delete(ixall1, ix1)
    # Get elements that are not in c2 but not c1
    ixall2 = np.arange(len(c2))
    ixnew2 = np.delete(ixall2, ix2)
    nidx = len(ix1)

    if len(ix1) != len(ix2):
        raise Exception("ix1 and ix2 have different dimensions")

    # Make sure that c1 and c2 are lists, if not we re-cast them as lists
    if not isinstance(c1, list):
        try:
            c1 = list(c1)
        except Exception as err:
            raise Exception(f"Cannot cast c1 as list {err=}, {type(err)=}")
    if not isinstance(c2, list):
        try:
            c2 = list(c2)
        except Exception as err:
            raise Exception(f"Cannot cast c2 as list {err=}, {type(err)=}")

    # We will store the new list here as a list called "newlist"
    # and make sure that the newlist elements are also lists:
    newlist = [c if isinstance(c, list) else [c] for c in c1]

    # Step 1, we stack the elements of c1, c2, by indexing (ix1, ix2) and
    # we want to augment c1 lists with new matches (ix2) from c2
    for k in range(nidx):
        i = ix1[k]
        j = ix2[k]
        newlist[i].append(c2[j])

    # If we only want the ones matches (not adding new objects, we stop here)
    if noNew:
        return newlist

    # Step 2, we want to append new elememts in c2 to existing c1
    LOGGER.debug(f"ixnew1: {ixnew1}")
    if pad:
        for k in ixnew1:
            c = newlist[k][0]
            newlist[k] = [c, c]

    LOGGER.debug(f"ixnew2: {ixnew2}")
    for k in ixnew2:
        c = c2[k]
        if pad:
            newlist.append([c, c])
            LOGGER.debug(f"padding ixnew2: {len(newlist)} {newlist[-1]}")
        else:
            newlist.append([c])
    return newlist


def mean_list_of_list(list, np_array=True):
    """
    Compute the mean of each sublist in a list of lists, and optionally return
    the result as a NumPy array.

    Parameters:
    - list (list of list-like objects): A list of lists (or arrays) for which the
      mean of each sublist will be computed.
    - np_array (bool, optional): If `True`, the result is returned as a NumPy array.
      If `False`, a list of means is returned. Default is `True`.

    Returns:
    - list or numpy.ndarray: The mean of each sublist. If `np_array` is `True`,
      a NumPy array is returned; otherwise, a list is returned.

    Example:
    >>> mean_list_of_list([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    array([2.0, 5.0, 8.0])

    Notes:
    - Each sublist in the input list is converted into a NumPy array before
      computing the mean.
    - The function will return a single mean value for each sublist.
    """
    u = [np.array(x).mean() for x in list]
    if np_array:
        u = np.array(u)
    return u


def max_list_of_list(list, np_array=True):
    """
    Compute the maximum value of each sublist in a list of lists,
    and optionally return the result as a NumPy array.

    Parameters:
    - list (list of list-like objects): A list of lists (or arrays) for which
      the maximum value of each sublist will be computed.
    - np_array (bool, optional): If `True`, the result is returned as a NumPy array.
      If `False`, a list of maximum values is returned. Default is `True`.

    Returns:
    - list or numpy.ndarray: The maximum value of each sublist. If `np_array` is `True`,
      a NumPy array is returned; otherwise, a list is returned.

    Example:
    >>> max_list_of_list([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    array([3, 6, 9])

    Notes:
    - Each sublist in the input list is converted into a NumPy array before computing
      the maximum value.
    - The function will return the maximum value for each sublist.
    """

    max_val = [np.array(x).max() for x in list]
    if np_array:
        max_val = np.array(max_val)
    return max_val


def compute_rms2D(data, mask=None, box=200, filter_size=(3, 3), sigmaclip=None):

    """
    Compute a 2D map of the RMS (Root Mean Square) using photutils.Background2D and
    photutils.StdBackgroundRMS.

    This function estimates the background RMS across the 2D input data, optionally applying
    a mask, a box size for background estimation, and a sigma clip for outlier rejection.

    Parameters:
    - data (numpy.ndarray): A 2D array containing the input data (e.g., image or map).
    - mask (numpy.ndarray, optional): A 2D boolean array of the same shape as `data`, where
      `True` values indicate masked pixels. If `None`, no masking is applied. Default is `None`.
    - box (int, optional): The size of the box used to estimate the background. The background
      is calculated within this box size. Default is 200.
    - filter_size (tuple of ints, optional): The size of the filter used for smoothing the
      background estimate. Default is (3, 3).
    - sigmaclip (float, optional): If specified, values in the input data that deviate by more
      than `sigmaclip` standard deviations from the mean will be clipped. Default is `None`.

    Returns:
    - photutils.Background2D: A Background2D object that contains the estimated background
      and the RMS of the input data.

    Example:
    >>> rms_map = compute_rms2D(data, mask=mask, box=150, filter_size=(5, 5), sigmaclip=3)

    Notes:
    - The `sigma_clip` is applied to the background estimation if specified. It helps to reject
      outliers in the background calculation.
    - If a mask is provided, the data is masked by replacing the masked areas with `NaN`,
      which prevents them from affecting the background estimation.
    """

    # Create a SigmaClip object for sigma clipping if necessary.
    sigma_clip = SigmaClip(sigma=sigmaclip) if sigmaclip is not None else None

    # Sigma clipping is already built into the Background2D object, so it is
    # not needed for the estimator.
    bkgrms_estimator = photutils.background.StdBackgroundRMS()

    bkg = photutils.background.Background2D(
        data,
        box,
        mask=mask,
        filter_size=filter_size,
        sigma_clip=sigma_clip,
        bkgrms_estimator=bkgrms_estimator,
    )
    return bkg

def detect_with_photutils(data, wgt=None, mask=None, nsigma_thresh=3.5, npixels=20,
                          rms2D=False, rms2Dimage=False, box=(200, 200),
                          filter_size=(3, 3), sigmaclip=None, wcs=None,
                          plot=False, plot_title=None, plot_name=None
):
    """
    Use photutils SourceFinder and SourceCatalog to create a catalog of sources.

    This function uses the SourceFinder from photutils to perform source detection
    and segmentation on the input data. It generates a catalog of sources, with
    various properties, and can also produce plots of the detection and background
    data. If requested, it can compute a 2D RMS background map and apply sigma
    clipping for noise reduction.

    Parameters:
        data (array-like): The 2D array of image data for source detection.
        wgt (array-like, optional): The weight map associated with the image data.
        mask (array-like, optional): A mask to apply to the data before detection.
        nsigma_thresh (float, optional): The number of sigmas to use for thresholding
            the detection (default is 3.5).
        npixels (int, optional): The minimum number of connected pixels to consider
            for a valid detection (default is 20).
        rms2D (bool, optional): If True, compute a 2D RMS background map (default is False).
        rms2Dimage (bool, optional): If True, save the computed 2D RMS image to a FITS file (default is False).
        box (tuple, optional): The size of the box for RMS computation (default is (200, 200)).
        filter_size (tuple, optional): The filter size for smoothing the RMS map (default is (3, 3)).
        sigmaclip (float, optional): A value to apply sigma clipping (default is None).
        wcs (astropy.wcs.WCS, optional): The WCS object for the image to convert the pixel coordinates
            into sky coordinates (default is None).
        plot (bool, optional): If True, plot the detection, segmentation, and distribution results (default is False).
        plot_title (str, optional): Title for the plot (default is None).
        plot_name (str, optional): Name for saving the plot (default is None).

    Returns:
        tuple: A tuple containing:
            - segm (array-like): The segmentation map of detected sources.
            - tbl (astropy.table.Table): A table containing the source catalog.

    Raises:
        ValueError: If no sources are detected in the input data.

    Example:
        >>> detect_with_photutils(data, mask=my_mask, plot=True, plot_title='Source Detection')
    """
    t0 = time.time()
    if mask is not None:
        # Make the data array a masked array (better plots)
        data = ma.masked_array(data, mask)
        mean, sigma = norm.fit(data[~mask])
    else:
        mean, sigma = norm.fit(data)

    # Define the threshold, array in the case of rms2D
    if rms2D:
        t0 = time.time()
        bkg = compute_rms2D(data, mask=mask, box=box, filter_size=filter_size, sigmaclip=sigmaclip)
        sigma2D = bkg.background_rms if mask is None else np.where(mask, np.nan, bkg.background_rms)
        threshold = nsigma_thresh * sigma2D
        LOGGER.debug(f"2D RMS computed in {elapsed_time(t0)}")
        # Dump 2D rms image into a fits file
        if rms2Dimage:
            hdr = wcs.to_header()
            hdr = astropy2fitsio_header(hdr)
            fitsname = f"{plot_name}_bkg.fits"
            fits = fitsio.FITS(fitsname, 'rw', clobber=True)
            fits.write(sigma2D, header=hdr)
            fits.close()
            LOGGER.info(f"2D RMS FITS image: {fitsname}")
        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
            plot_distribution(ax1, data if mask is None else data[~mask], mean, sigma, nsigma=nsigma_thresh)
            plot_rms2D(bkg.background, ax2, mask=mask)
            plt.savefig(f"{plot_name}_bkg.pdf")
            LOGGER.info(f"Created: {plot_name}_bkg.pdf")
    else:
        threshold = nsigma_thresh * sigma

    # Perform segmentation and deblending
    finder = SourceFinder(npixels=npixels, nlevels=32, contrast=0.001, progress_bar=False)
    segm = finder(data, threshold, mask=mask)
    # We stop if we don't find source
    if segm is None:
        LOGGER.info("No sources found in astropy/segm, returning (None, None)")
        return None, None
    cat = SourceCatalog(data, segm, error=wgt, mask=mask, wcs=wcs, progress_bar=True)
    # Make sure these are added.
    cat.default_columns.append('elongation')
    cat.default_columns.append('ellipticity')

    LOGGER.info(f"detect_with_photutils runtime: {elapsed_time(t0)}")
    LOGGER.info(f"Found: {len(cat)} objects")

    # Nicer formatting
    tbl = cat.to_table()
    tbl['xcentroid'].info.format = '.2f'  # optional format
    tbl['ycentroid'].info.format = '.2f'
    tbl['max_value'].info.format = '.2f'
    tbl['elongation'].info.format = '.2f'
    tbl['ellipticity'].info.format = '.2f'
    tbl['eccentricity'].info.format = '.2f'
    tbl['sky_centroid_dms'] = tbl['sky_centroid'].to_string('hmsdms', precision=0)
    snr_max = compute_snr(tbl, threshold/nsigma_thresh, key='max_value')
    tbl.add_column(snr_max, name='snr_max')
    tbl['snr_max'].info.format = '.2f'
    print(tbl['label', 'xcentroid', 'ycentroid', 'sky_centroid', 'sky_centroid_dms',
              'max_value', 'snr_max', 'eccentricity', 'elongation', 'ellipticity', 'area'])
    if plot:
        t1 = time.time()
        if rms2D:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        plot_detection(ax1, data, cat, sigma, nsigma_plot=5, plot_title=plot_title)
        plot_segmentation(ax2, segm, cat, mask=mask)
        plot_distribution(ax3, data if mask is None else data[~mask], mean, sigma, nsigma=nsigma_thresh)
        if rms2D:
            # Make the background image a masked arrays
            plot_rms2D(bkg.background, ax4, mask=mask)
        if plot_name:
            plt.savefig(f"{plot_name}.pdf")
            LOGGER.info(f"Saved: {plot_name}.pdf")
        else:
            plt.show()
        plt.close()
        LOGGER.info(f"detect_with_photutils PLOT runtime: {elapsed_time(t1)}")

    LOGGER.info(f"detect_with_photutils TOTAL runtime: {elapsed_time(t0)}")
    return segm, tbl


def g3_or_fits(filename):
    """
    Check the file extension to determine whether the file is a FITS or G3 file.

    This function inspects the extension of the provided filename to identify
    whether the file is a FITS file (including compressed variants) or a G3 file
    (including compressed variants). If the file type cannot be determined,
    a warning is logged and a ValueError is raised.

    Args:
        filename (str): The name of the file to check.

    Returns:
        str: The type of the file, either 'FITS' or 'G3'.

    Raises:
        ValueError: If the file extension is neither FITS nor G3.

    Example:
        >>> g3_or_fits('data/file.fits')
        'FITS'
    """
    path = Path(filename)
    ext = "".join(path.suffixes).lower()

    if ext in [".fits", ".fits.gz", ".fits.fz"]:
        return "FITS"
    elif ext in [".g3", ".g3.gz"]:
        return "G3"
    else:
        msg = f"Could not find filetype for file {filename}"
        LOGGER.warning(msg)
        raise ValueError(msg)


def plot_rms2D(bkg, ax, mask=None, nsigma_plot=3.5):
    """
    Plot the 2D noise map (RMS) with optional masking.

    This function visualizes a 2D background noise map (RMS) and optionally
    applies a mask to hide certain regions of the data. The noise map is displayed
    using a grayscale color map, and a color bar is included to indicate the values.

    Args:
        bkg (numpy.ndarray): The 2D background noise data (RMS) to be displayed.
        ax (matplotlib.axes.Axes): The axes on which to plot the data.
        mask (numpy.ndarray, optional): A boolean mask array to hide certain regions in the plot. Default is None.
        nsigma_plot (float, optional): The number of standard deviations for color scaling. Default is 3.5.

    Example:
        >>> plot_rms2D(bkg, ax)
    """
    # Plot a masked array if mask is passed
    if mask is not None:
        bkg = ma.masked_array(bkg, mask)
    im = ax.imshow(bkg, origin='lower', cmap='Greys')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_title('2D Noise Map')


def plot_detection(ax1, data, cat, sigma, nsigma_plot=5, plot_title=None):
    """
    Plot the 2D data array with overlaid catalog apertures.

    This function visualizes a 2D data array (such as an image) and overlays
    catalog apertures, which are used to highlight detected sources.
    The plot also adjusts the color limits based on the standard deviation
    of the data to improve visualization.

    Args:
        ax1 (matplotlib.axes.Axes): The axes on which to plot the data array.
        data (numpy.ndarray): The 2D data array to be displayed (e.g., an image).
        cat (astropy.table.Table): The catalog containing object information, used to overlay apertures.
        sigma (float): The standard deviation of the data, used to determine the color limits.
        nsigma_plot (int, optional): The number of sigma to scale the color limits. Default is 5.
        plot_title (str, optional): Title to be displayed on the plot. Default is None.

    Example:
        >>> plot_detection(ax1, data, cat, sigma)
    """
    vlim = nsigma_plot*sigma
    im1 = ax1.imshow(data, origin='lower', cmap='viridis', vmin=-vlim, vmax=+vlim)
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1)
    if plot_title:
        ax1.set_title(plot_title)
    cat.plot_kron_apertures(ax=ax1, color='white', lw=0.5)


def plot_segmentation(ax2, segm, cat, mask=None):
    """
    Plot a segmentation image with overlaid catalog apertures.

    This function visualizes the segmentation image, where each segment
    represents a detected object, and overlays catalog apertures (if available).
    It also allows for the optional masking of segments using a provided mask.

    Args:
        ax2 (matplotlib.axes.Axes): The axes on which to plot the segmentation image.
        segm (numpy.ndarray): The segmentation image (2D array).
        cat (astropy.table.Table): The catalog containing object information, used to overlay apertures.
        mask (numpy.ndarray, optional): A boolean mask to hide certain segments in the plot. Default is None.

    Example:
        >>> plot_segmentation(ax2, segm, cat)
    """
    # Plot a masked array if mask is passed
    if mask is not None:
        segm_plot = ma.masked_array(segm, mask)
    else:
        segm_plot = segm
    im = ax2.imshow(segm_plot, origin='lower', cmap=segm.cmap,
                    interpolation='nearest')
    divider = make_axes_locatable(ax2)
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax2)
    ax2.set_title('Segmentation Image')
    cat.plot_kron_apertures(ax=ax2, color='white', lw=0.5)


def plot_distribution(ax, data, mean, sigma, nsigma=3):
    """
    Plot the 1D distribution of data and fit a normal distribution curve.

    This function visualizes the distribution of the given data and fits a
    Gaussian (normal) distribution to it. The plot includes the data histogram,
    the fitted curve, and dashed lines representing the specified number of
    standard deviations (nsigma) from the mean.

    Args:
        ax (matplotlib.axes.Axes): The axes on which to plot the histogram and fitted curve.
        data (array-like): The data to be plotted. It will be flattened if necessary.
        mean (float): The mean value of the distribution.
        sigma (float): The standard deviation of the distribution.
        nsigma (int, optional): The number of standard deviations to plot the dashed lines.
            Default is 3.

    Example:
        >>> plot_distribution(ax, data, mean, sigma)
    """
    # Flatten the data if needed
    if data.ndim != 1:
        data = data.flatten()

    xmin = data.min()
    xmax = data.max()
    # Cut xmin and ymax at 10sigma for plotting
    if xmin < -10*sigma:
        xmin = -10*sigma
    if xmax > 10*sigma:
        xmax = 10*sigma

    legend = "$\\mu$: %.6f\n$\\sigma$: %.6f" % (mean, sigma)
    # Plot data and fit
    nbins = int(data.shape[0]/5000.)
    hist = ax.hist(data, range=(xmin, xmax), bins=nbins, density=True, alpha=0.6)
    ymin, ymax = hist[0].min(), hist[0].max()
    xmin, xmax = hist[1].min(), hist[1].max()

    x = np.linspace(xmin, xmax, 100)
    y = norm.pdf(x, mean, sigma)
    ax.plot(x, y)
    xx = [nsigma*sigma, nsigma*sigma]
    yy = [ymin, ymax]
    ax.plot(xx, yy, 'k--', linewidth=1)
    xx = [-nsigma*sigma, -nsigma*sigma]
    ax.plot(xx, yy, 'k--', linewidth=1)
    ax.set_ylim(ymin, ymax)
    ax.legend([legend], frameon=False)
    text = f"${nsigma}\\sigma$"
    ax.text(0.05*(xmax-xmin) + nsigma*sigma, (ymax-ymin)/20., text, horizontalalignment='center')
    ax.text(-0.05*(xmax-xmin) - nsigma*sigma, (ymax-ymin)/20., text, horizontalalignment='center')

    ratio = 0.95
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    ax.set_xlabel("Flux")
    ax.set_title("1D Noise Distribution and Fit")


def get_sources_catalog(field, point_source_file=None):
    """
    Retrieve the sources catalog for a given field.

    This function reads the point source catalog for the specified field and
    returns an Astropy `SkyCoord` object containing the celestial coordinates
    of the sources.

    Args:
        field (str): The name of the field for which the sources catalog is to be retrieved.

    Returns:
        astropy.coordinates.SkyCoord: A SkyCoord object containing the RA and Dec coordinates
            of the sources in the catalog.

    Example:
        >>> sources_catalog = get_sources_catalog('field_name')
    """
    # Get the catalog with masked sources
    if point_source_file is None:
        point_source_file = sources.get_field_source_list(field, analysis="lightcurve")
    _, psra, psdec, _ = sources.read_point_source_mask_file(point_source_file)
    LOGGER.debug(f"Loading source mask positions from file: {point_source_file}")
    # Create a SkyCoord object with the sources catalog to make the matching
    # psra and psdec are in G3 units and need to be converted back to degrees to be use in astropy
    psra = psra/core.G3Units.deg
    psdec = psdec/core.G3Units.deg
    pscat = SkyCoord(ra=psra*u.degree, dec=psdec*u.degree)
    return pscat


def remove_objects_near_sources(cat, field, point_source_file=None, max_dist=5*u.arcmin):
    """
    Remove objects from an Astropy catalog that are near sources in a sources catalog.

    This function compares the objects in the provided catalog (`cat`) to a
    sources catalog for a given field. Objects within a specified angular distance
    (`max_dist`) from any source in the sources catalog will be removed.

    Args:
        cat (astropy.table.Table): The Astropy catalog of objects to be filtered.
        field (str): The field name to retrieve the sources catalog for matching.
        max_dist (astropy.units.Quantity, optional): The maximum separation distance
            for matching sources, default is 5 arcminutes.

    Returns:
        astropy.table.Table: The input catalog with objects removed that were near
            sources in the sources catalog.

    Example:
        >>> filtered_cat = remove_objects_near_sources(catalog, 'field_name')
    """
    # Get a astropy SkyCoord object catalog to match
    try:
        pscat = get_sources_catalog(field, point_source_file)
    except KeyError:
        LOGGER.warning(f"Cannot get sources catalog for field: {field}")
        return cat

    # Extract the SkyCoord object
    ind_cat, ind_ps, dist, _ = search_around_sky(cat['sky_centroid'], pscat, max_dist)
    remove_mask = np.zeros(len(cat), dtype=bool)
    remove_mask[ind_cat] = True
    if len(ind_cat) > 0:
        LOGGER.info(f"Found {len(ind_cat)} matches, will remove them from catalog")
        LOGGER.debug("Will remove: ")
        if LOGGER.getEffectiveLevel() == logging.DEBUG:
            print(cat[remove_mask][PPRINT_KEYS])
    else:
        LOGGER.info("No matches found in sources catalog")
    return cat[~remove_mask]


def compute_snr(catalog, sigma, key='max_value'):

    """
    Compute signal-to-noise ratio for catalog

    Parameters:
    - catalog: input catalog. We use 'max_value' from the catalog
      as 'signal'
    - sigma: The estimage of the noise (sigma). It can be a scalar
      (i.e.: same for all pixels), or a 2d array.
    """

    # single scalar value for all objects
    if np.isscalar(sigma):
        noise = sigma
    # we have 2d numpy array. We use x,y positions
    else:
        x = np.round(catalog['xcentroid'].data).astype(int)
        y = np.round(catalog['ycentroid'].data).astype(int)
        noise = sigma[y, x]
    return catalog[key].data/noise


def astropy2fitsio_header(header):
    """
    Convert an Astropy FITS header object into a FITSIO FITSHDR object.

    This function translates an Astropy header object into a FITSHDR object that
    can be used with the FITSIO library. It extracts the header key-value pairs
    and their associated comments to create the corresponding FITSIO header.

    Args:
        header (astropy.io.fits.Header): The Astropy FITS header object to be converted.

    Returns:
        fitsio.FITSHDR: A FITSHDR object that corresponds to the input Astropy header.

    Example:
        >>> astropy_header = fits.getheader('image.fits')
        >>> fitsio_header = astropy2fitsio_header(astropy_header)
    """
    # Make the header a FITSHDR object
    hlist = []
    for key in header:
        hlist.append({'name': key, 'value': header[key], 'comment': header.comments[key]})
    h = fitsio.FITSHDR(hlist)
    return h


def replace_element(my_list, old_element, new_element):
    """
    Replaces an element in a list if it exists.

    Args:
        my_list: The list to modify.
        old_element: The element to replace.
        new_element: The new element to insert.
    """
    if old_element in my_list:
        index = my_list.index(old_element)
        my_list[index] = new_element
    return my_list


def sort_bands(bands):
    """
    Sorting gymastics for bands when 90Ghz is present
    """
    if '90GHz' in bands:
        bands = replace_element(bands, '90GHz', '090GHz')
        bands.sort(reverse=True)
        bands = replace_element(bands, '090GHz', '90GHz')
    return bands
