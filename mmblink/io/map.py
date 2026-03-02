"""Functions for loading maps from files."""

import fitsio

__all__ = ["load_fits_map"]


def load_fits_map(filename):
    """Read data and metadata from a FITS file of a map.

    Reads the header from the SCI HDU. Maps the SCI and WGT extensions to a
    dictionary. For FITS files without EXTNAME, the SCI and WGT HDUs are
    assumed to be the first and second headers containing data respectively.

    Parameters
    ----------
        filename : str
            Path to the FITS file.

    Returns
    -------
    header : fitsio.header.FITSHDR
    hdus : dict
        A dictionary containing map data and metadata. The keys are:
        - "SCI" : ndarray
        - "WGT" : ndarray
    """
    header_map = {}
    hdu_indices = {}
    is_compressed = False

    with fitsio.FITS(filename, "r") as fits:
        for i, hdu in enumerate(fits):
            header = hdu.read_header()
            if header.get("ZIMAGE"):
                is_compressed = True
            if not header.get("EXTNAME"):
                # Do not include headers without EXTNAME.
                continue
            extname = header["EXTNAME"].strip()
            header_map[extname] = header
            hdu_indices[extname] = i
        if len(header_map) < 1:
            # For FITS files without EXTNAME, assume that the first HDU with
            # data is SCI and the second HDU with data is WGT.
            sci_idx = 1 if is_compressed else 0
            wgt_idx = sci_idx + 1

            hdu_indices["SCI"] = sci_idx
            header_map["SCI"] = fits[sci_idx].read_header()

            # Attempt to grab WGT if it exists at the next index.
            if len(fits) > wgt_idx:
                hdu_indices["WGT"] = wgt_idx
                header_map["WGT"] = fits[wgt_idx].read_header()
            else:
                raise ValueError(f"No WGT header found for {filename}")
        return (
            header_map["SCI"],
            {
                "SCI": fits[hdu_indices["SCI"]].read(),
                "WGT": fits[hdu_indices["WGT"]].read(),
            },
        )
