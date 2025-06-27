from bids import BIDSLayout
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


def relative_to_root(path):
    """
    Calculate the BIDS root folder given one file path's.

    Examples
    --------
    >>> str(relative_to_root(
    ...     "/sub-03/sourcedata/sub-01/anat/sub-01_T1.nii.gz"
    ... ))
    'sub-01/anat/sub-01_T1.nii.gz'

    >>> str(relative_to_root(
    ...     "/sub-03/anat/sourcedata/sub-01/ses-preop/anat/sub-01_ses-preop_T1.nii.gz"
    ... ))
    'sub-01/ses-preop/anat/sub-01_ses-preop_T1.nii.gz'

    >>> str(relative_to_root(
    ...     "sub-01/anat/sub-01_T1.nii.gz"
    ... ))
    'sub-01/anat/sub-01_T1.nii.gz'

    >>> str(relative_to_root("anat/sub-01_T1.nii.gz"))
    'anat/sub-01_T1.nii.gz'

    """
    path = Path(path)
    if path.name.startswith("sub-"):
        parents = [path.name]
        for p in path.parents:
            parents.insert(0, p.name)
            if p.name.startswith("sub-"):
                return Path(*parents)
        return path
    raise ValueError(
        f"Could not determine the BIDS root of <{path}>. "
        "Only files under a subject directory are currently supported."
    )


def _init_layout(in_file=None, bids_dir=None, validate=True, database_path=None):

    if isinstance(bids_dir, BIDSLayout):
        return bids_dir
    if bids_dir is None:
        in_file = Path(in_file)
        for parent in in_file.parents:
            if parent.name.startswith("sub-"):
                bids_dir = parent.parent.resolve()
                break
        if bids_dir is None:
            raise RuntimeError("Could not infer BIDS root")
    layout = BIDSLayout(
        str(bids_dir),
        validate=validate,
        database_path=database_path,
    )
    return layout
