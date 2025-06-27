import logging
import os


logger = logging.getLogger(__name__)


def _copy_any(src, dst):

    import os
    import gzip
    from shutil import copyfileobj
    from pydra.tasks.mriqc.nipype_ports.utils.filemanip import copyfile

    src_isgz = src.endswith(".gz")
    dst_isgz = dst.endswith(".gz")
    if not src_isgz and not dst_isgz:
        copyfile(src, dst, copy=True, use_hardlink=True)
        return False  # Make sure we do not reuse the hardlink later
    # Unlink target (should not exist)
    if os.path.exists(dst):
        os.unlink(dst)
    src_open = gzip.open if src_isgz else open
    with src_open(src, "rb") as f_in:
        with open(dst, "wb") as f_out:
            if dst_isgz:
                # Remove FNAME header from gzip (nipreps/fmriprep#1480)
                gz_out = gzip.GzipFile("", "wb", 9, f_out, 0.0)
                copyfileobj(f_in, gz_out)
                gz_out.close()
            else:
                copyfileobj(f_in, f_out)
    return True


def unlink(pathlike, missing_ok=False):
    """Backport of Path.unlink from Python 3.8+ with missing_ok keyword"""
    # PY37 hack; drop when python_requires >= 3.8
    try:
        os.unlink(pathlike)
    except FileNotFoundError:
        if not missing_ok:
            raise
