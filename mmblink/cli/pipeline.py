import argparse
import os

from astropy.wcs import WCS
import numpy as np

import mmblink
from mmblink import pipeline


def basic_pipeline(filename, out_dir):
    header, map_data = mmblink.io.load_fits_map(filename)
    flux = map_data["SCI"]
    weight = map_data["WGT"]
    mask = weight == 0  # True means data is masked.

    background_rms = pipeline.background_rms_map(flux, mask=mask, box_size=60)
    threshold = pipeline.n_sigma_threshold(5.0, background_rms)
    segmented = pipeline.segment_sources(flux, threshold, npixels=20, mask=mask)
    if segmented is None:
        return
    catalog = pipeline.catalog_sources(
        flux, segmented, weight=weight, wcs=WCS(header)
    )
    catalog = pipeline.set_sky_centroid_string_column(catalog)
    catalog = pipeline.set_snr_max_column(catalog, background_rms)
    catalog = pipeline.set_catalog_obs_info(
        catalog,
        band=header["BAND"],
        obs_id=header["OBSID"],
        field=header["FIELD"],
    )
    catalog = pipeline.filter_ellipticity(catalog, 0.3)

    if len(catalog) > 0:
        key = f"{header['OBSID']}_{header['BAND']}"
        catalog.write(
            os.path.join(out_dir, f"{key}_catalog.csv"),
            format="csv",
            overwrite=True,
        )


def main():
    parser = argparse.ArgumentParser(description="mm-wave transient detection")
    parser.add_argument("files", nargs="+", help="Filename(s) to process")
    parser.add_argument(
        "out_dir", action="store", help="Location for output files"
    )
    args = parser.parse_args()
    tasks = [(file, args.out_dir) for file in args.files]
    mmblink.manage.run_pipeline_parallel(basic_pipeline, tasks, num_workers=2)


if __name__ == "__main__":
    main()
