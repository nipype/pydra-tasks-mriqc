import logging
from pydra.compose import workflow


logger = logging.getLogger(__name__)


BUILDING_WORKFLOW = "Building {modality} MRIQC workflow {detail}."

QC_UPLOAD_COMPLETE = "QC metrics successfully uploaded."

QC_UPLOAD_START = "MRIQC Web API: submitting to <{url}>"

SUSPICIOUS_DATA_TYPE = "Input image {in_file} has a suspicious data type: '{dtype}'"

VOXEL_SIZE_OK = "Voxel size is large enough."

VOXEL_SIZE_SMALL = (
    "One or more voxel dimensions (%f, %f, %f) are smaller than the "
    "requested voxel size (%f) - diff=(%f, %f, %f)"
)
