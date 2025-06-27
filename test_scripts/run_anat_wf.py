from fileformats.medimage import NiftiGzX, T1Weighted
import logging
import tempfile
from pathlib import Path
from pydra.tasks.mriqc.workflows.anatomical.base import anat_qc_workflow

log_file = Path("/Users/tclose/Data/pydra-mriqc-test.log")
log_file.unlink(missing_ok=True)

pydra_logger = logging.getLogger("pydra")
pydra_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(str(log_file))
pydra_logger.addHandler(file_handler)
pydra_logger.addHandler(logging.StreamHandler())

in_file = NiftiGzX[T1Weighted].sample()

tmp_dir = Path(tempfile.mkdtemp())

in_file = in_file.copy(tmp_dir, new_stem="sub-01_T1w")

workflow = anat_qc_workflow(in_file=in_file, modality="T1w")
workflow.cache_dir = "/Users/tclose/Data/pydra-mriqc-test-cache"
result = workflow(plugin="serial")
print(result.out)
