#!/usr/bin/env python

import mmblink.ftools as ft
import argparse
import os


def cmdline():
    """
    Parse command-line arguments to load and plot thumbnails and lightcurves

    Returns:
    - args (Namespace): The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Load and plot thumbnails/stamps and lightcurves")
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

    LC = {}
    ids = {}
    # Read in the lightcurve as LC and store the ids
    outdir = None
    for lc_file in args.lc_files:
        lc, band = ft.load_fits_table(lc_file, target_id=None)
        LC[band] = lc
        ids[band] = [row['id'] for row in lc]
        # Extract the path for the location of stamps
        if outdir and outdir != os.path.dirname(lc_file):
            raise ValueError(f"Inconsistent: {outdir}")
        outdir = os.path.dirname(lc_file)

    # Get the bands we want to plot
    bands = list(ids.keys())
    nbands = len(bands)

    # Check that ids are the same in case we have more than one key
    id0 = ids[bands[0]]
    n0 = len(id0)
    if nbands > 1:
        for k in range(nbands-1):
            k += 1
            idn = ids[bands[k]]
            nk = len(idn)
            if idn == id0:
                print(f"arrays are the same for bands: {bands[0]} -- {bands[k]}")
            else:
                print(f"# WARNING: LC arrays are not same for bands: {bands[0]} [{n0}]-- {bands[k]} [{nk}]")

    # Loop over all ids and plot one at a time
    k = 0
    for id in id0:
        images = {}
        headers = {}
        lightcurve = {}
        for band in bands:
            stamp = os.path.join(outdir, f"{id}_{band}.fits")
            images[band], headers[band], id_stamp, band_stamp = ft.load_fits_stamp(stamp)
            # Make sure ids are the same
            if id != id_stamp:
                raise ValueError(f"Object ID are not the same {id}!={id_stamp}")
            try:
                lightcurve[band] = LC[band][k]
            except IndexError:
                print(f"# WARNING: No {band} lightcurve for {id}")

        # Plot ID
        ft.plot_stamps_lc(images, headers, lightcurve,
                          obsmin=args.obsmin,
                          obsmax=args.obsmax,
                          outdir=args.outdir,
                          format=args.plot_format,
                          show=args.show_plot)
        k += 1
