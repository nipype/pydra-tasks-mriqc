import logging
from pathlib import Path


logger = logging.getLogger(__name__)


def derive_bids_fname(
    orig_path: str | Path,
    entity: str | None = None,
    newsuffix: str | None = None,
    newpath: str | Path | None = None,
    newext: str | None = None,
    position: int = -1,
    absolute: bool = True,
) -> Path | str:
    """
    Derive a new file name from a BIDS-formatted path.

    Parameters
    ----------
    orig_path : :obj:`str` or :obj:`os.pathlike`
        A filename (may or may not include path).
    entity : :obj:`str`, optional
        A new BIDS-like key-value pair.
    newsuffix : :obj:`str`, optional
        Replace the BIDS suffix.
    newpath : :obj:`str` or :obj:`os.pathlike`, optional
        Path to replace the path of the input orig_path.
    newext : :obj:`str`, optional
        Replace the extension of the file.
    position : :obj:`int`, optional
        Position to insert the entity in the filename.
    absolute : :obj:`bool`, optional
        If True (default), returns the absolute path of the modified filename.

    Returns
    -------
    Absolute path of the modified filename

    Examples
    --------
    >>> derive_bids_fname(
    ...     'sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz',
    ...     entity='desc-preproc',
    ...     absolute=False,
    ... )
    PosixPath('sub-001/ses-01/anat/sub-001_ses-01_desc-preproc_T1w.nii.gz')

    >>> derive_bids_fname(
    ...     'sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz',
    ...     entity='desc-brain',
    ...     newsuffix='mask',
    ...     newext=".nii",
    ...     absolute=False,
    ... )  # doctest: +ELLIPSIS
    PosixPath('sub-001/ses-01/anat/sub-001_ses-01_desc-brain_mask.nii')

    >>> derive_bids_fname(
    ...     'sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz',
    ...     entity='desc-brain',
    ...     newsuffix='mask',
    ...     newext=".nii",
    ...     newpath="/output/node",
    ...     absolute=True,
    ... )  # doctest: +ELLIPSIS
    PosixPath('/output/node/sub-001_ses-01_desc-brain_mask.nii')

    >>> derive_bids_fname(
    ...     'sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz',
    ...     entity='desc-brain',
    ...     newsuffix='mask',
    ...     newext=".nii",
    ...     newpath=".",
    ...     absolute=False,
    ... )  # doctest: +ELLIPSIS
    PosixPath('sub-001_ses-01_desc-brain_mask.nii')

    """
    orig_path = Path(orig_path)
    newpath = orig_path.parent if newpath is None else Path(newpath)
    ext = "".join(orig_path.suffixes)
    newext = newext if newext is not None else ext
    orig_stem = orig_path.name.replace(ext, "")
    suffix = orig_stem.rsplit("_", maxsplit=1)[-1].strip("_")
    newsuffix = newsuffix.strip("_") if newsuffix is not None else suffix
    orig_stem = orig_stem.replace(suffix, "").strip("_")
    bidts = [bit for bit in orig_stem.split("_") if bit]
    if entity:
        if position == -1:
            bidts.append(entity)
        else:
            bidts.insert(position, entity.strip("_"))
    retval = newpath / f'{"_".join(bidts)}_{newsuffix}.{newext.strip(".")}'
    return retval.absolute() if absolute else retval
