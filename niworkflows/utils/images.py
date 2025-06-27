from gzip import GzipFile
import logging
import nibabel as nb


logger = logging.getLogger(__name__)


def set_consumables(header, dataobj):

    header.set_slope_inter(dataobj.slope, dataobj.inter)
    header.set_data_offset(dataobj.offset)


def unsafe_write_nifti_header_and_data(fname, header, data):
    """Write header and data without any consistency checks or data munging

    This is almost always a bad idea, and you should not use this function
    without a battery of tests for your specific use case.

    If you're not using this for NIfTI files specifically, you're playing
    with Fortran-ordered fire.
    """
    with open(fname, "wb") as fobj:
        # Avoid setting fname or mtime, for deterministic outputs
        if str(fname).endswith(".gz"):
            fobj = GzipFile("", "wb", 9, fobj, 0.0)
        header.write_to(fobj)
        # This function serializes one block at a time to reduce memory usage a bit
        # It assumes Fortran-ordered data.
        nb.volumeutils.array_to_file(data, fobj, offset=header.get_data_offset())
        if str(fname).endswith(".gz"):
            fobj.close()
