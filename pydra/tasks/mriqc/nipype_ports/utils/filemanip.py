import logging


logger = logging.getLogger(__name__)


_cifs_table = _generate_cifs_table()

fmlogger = logging.getLogger("nipype.utils")

related_filetype_sets = [(".hdr", ".img", ".mat"), (".nii", ".mat"), (".BRIK", ".HEAD")]
