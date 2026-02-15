#!/usr/bin/env python
import mmblink.dtools as du
import argparse
import time


def cmdline():
    """
    Parses command-line arguments for the mm-transient detection script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="mm-wave transient detection")
    parser.add_argument("files", nargs='+',
                        help="Filename(s) to ingest")
    parser.add_argument("--outdir", type=str, action='store', default=None,
                        required=True, help="Location for output files")
    parser.add_argument("--clobber", action='store_true', default=False,
                        help="Clobber output files")

    parser.add_argument("--field", type=str, action='store', default=None,
                        help="Field name (i.e. SourceName) for automatically determining point source file to use.")
    parser.add_argument("--no_remove_source", action='store_true', default=False,
                        help="Do not perform removing objects near sources.")
    parser.add_argument("--detect_bands",  nargs='+', action='store', default=['90GHz', '150GHz'],
                        help="List of bands where will run source detection \
                              [i.e.: 90GHz, 150GHz, 200GHz], (default=90GHz 150GHz")
    parser.add_argument("--dual_detection", action='store_true', default=False,
                        help="Detection must be in both bands [90GHz, 150GHz]")
    parser.add_argument("--write_obscat", action='store_true', default=False,
                        help="Write out the catalogs for each obsID/band")
    parser.add_argument("--stamp_size", type=float, action='store', default=10,
                        help="Size (in arcmin) of the stamps to be cut [default:10 arcmin]")
    parser.add_argument("--prefix", type=str, action='store', default="SPT3G",
                        help="Prefix for IAU ID for objects (default: SPT3G)")
    # Logging options (loglevel/log_format/log_format_date)
    default_log_format = '[%(asctime)s.%(msecs)03d][%(levelname)s][%(name)s][%(funcName)s] %(message)s'
    default_log_format_date = '%Y-%m-%d %H:%M:%S'
    parser.add_argument("--loglevel", action="store", default='INFO', type=str.upper,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging Level [DEBUG/INFO/WARNING/ERROR/CRITICAL]")
    parser.add_argument("--log_format", action="store", type=str, default=default_log_format,
                        help="Format for logging")
    parser.add_argument("--log_format_date", action="store", type=str, default=default_log_format_date,
                        help="Format for date section of logging")
    # Detection options
    parser.add_argument("--rms2D", action='store_true', default=False,
                        help="Perform 2D map of the rms using photutils Background2D StdBackgroundRMS")
    parser.add_argument("--rms2D_box", action='store', type=int, default=60,
                        help="Size of box using photutils Background2D StdBackgroundRMS")
    parser.add_argument("--rms2D_image", action='store_true', default=False,
                        help="Create a FITS image of the Background2D")
    parser.add_argument("--npixels", action='store', type=int, default=20,
                        help="Minimum number of pixels in detection")
    parser.add_argument("--nsigma_thresh", action='store', type=float, default=5.0,
                        help="Number of sigmas use to compute the detection threshold [default: 5]")
    parser.add_argument("--max_sep", action='store', type=float, default=20.0,
                        help="Maximum angular separation to match sources in arcsec [default: 20 arcsec]")
    parser.add_argument("--ell_cut", action='store', type=float, default=1.0,
                        help="Ellipticity cut [default=0.3]")
    parser.add_argument("--plot", action='store_true', default=False,
                        help="Make plots with detection diagnostics?")
    parser.add_argument("--nr", action="store", default=1, type=int,
                        help="N-repeat, number of times source repeats [default=1]")
    parser.add_argument("--point_source_file", action='store', default=None,
                        help="User provided point source file to use for masking [default=None]")
    # Use multiprocessing
    parser.add_argument("--np", action="store", default=1, type=int,
                        help="Run using multi-process, 0=automatic, 1=single-process [default]")
    parser.add_argument("--ntheads", action="store", default=1, type=int,
                        help="The number of threads used by numexpr 0=automatic, 1=single [default]")

    args = parser.parse_args()
    # Sort the bands
    bands = du.sort_bands(args.detect_bands)
    args.detect_bands = bands
    return args


if __name__ == "__main__":
    """
    Main execution script for running transient detection on SPT-3G data.
    - Parses command-line arguments.
    - Runs the detection pipeline.
    - Matches detected sources across bands.
    - Generates output files including detection catalogs and cutouts.
    """
    # Start timing the script execution
    t0 = time.time()
    # Parse command-line arguments
    args = cmdline()
    # Initialize detection worker with parsed arguments
    g3d = du.g3detect(**args.__dict__)
    # Run the detection on all input files that match detection bands
    g3d.run_detection_files()

    # Source matching
    if args.dual_detection:
        # Dual matching, source should be detected in both bands per OBSID
        g3d.collect_dual()
    else:
        # Extract the unique detections per band we used
        g3d.collect_single()

    g3d.make_stamps_and_lighcurves()
    print(f"Grand total time: {du.elapsed_time(t0)}")
    exit()

    # Example 1, find all positions
    stacked_centroids = du.find_unique_centroids(g3d.cat, separation=args.max_sep, plot=args.plot)
    # print("stacked_centroids:")
    # print(stacked_centroids)
    print(f"Total time: {du.elapsed_time(t0)}")
    exit()

    # Example 2: Find repeating sources across observations
    table_centroids = du.find_repeating_sources(g3d.cat, separation=args.max_sep, plot=args.plot, outdir=args.outdir)
    stacked_centroids = du.find_unique_centroids(table_centroids, separation=args.max_sep, plot=args.plot)
    print("stacked_centroids:")
    print(stacked_centroids)
    print(f"Total time: {du.elapsed_time(t0)}")
