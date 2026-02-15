#!/usr/bin/env python

import mmblink.ftools as ft
import argparse


def cmdline():
    """
    Parse command-line arguments to load and plot thumbnails and lightcurves

    Returns:
    - args (Namespace): The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Load and plot thumbnails/stamps and lightcurves")
    parser.add_argument("stamp_files", nargs='+',
                        help="Stamps filenames to plot")
    parser.add_argument("--lc_files", nargs='+',
                        required=True, help="lightcurve filenames to plot")
    parser.add_argument("--outdir", type=str, action='store', default=None,
                        required=True, help="Location for output files")
    parser.add_argument("--plot_format", type=str, action='store', default="png",
                        help="File format for plot: png, pdf, jpeg, tiff, svg, eps (default=png)")
    parser.add_argument("--show_plot", action='store_true', default=False,
                        help="Show plot (default=False)")
    parser.add_argument("--obsmin", type=int, default=None,
                        help="Minimum observation ID to include in plot (default=None)")
    parser.add_argument("--obsmax", type=int, default=None,
                        help="Max observation ID to include in plot (default=None)")
    # We want to add range of obsid to plot
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # Get the command-line option
    args = cmdline()

    lightcurve = {}
    images = {}
    headers = {}

    # Read in the stamps and load them into a dictionary
    for k, stamp in enumerate(args.stamp_files):
        ima, hdr, id, band = ft.load_fits_stamp(stamp)
        if k == 0:
            old_id = id
        # Make sure ids are the same
        if id != old_id:
            raise ValueError(f"Object ID are not the same {old_id}!={id}")
        images[band] = ima
        headers[band] = hdr

    # Read in the lightcurve FITS table and load the corresponding lc for `id`
    # into a dictionary
    for lc_file in args.lc_files:
        lc, band = ft.load_fits_table(lc_file, target_id=id)
        lightcurve[band] = lc[0]

    # Plot
    ft.plot_stamps_lc(images, headers, lightcurve,
                      obsmin=args.obsmin,
                      obsmax=args.obsmax,
                      outdir=args.outdir,
                      format=args.plot_format,
                      show=args.show_plot)
