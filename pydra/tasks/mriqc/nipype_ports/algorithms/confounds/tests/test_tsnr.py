from fileformats.generic import File
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.nipype_ports.algorithms.confounds.tsnr import TSNR
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_tsnr_1():
    task = TSNR()
    task.inputs.in_file = [File.sample(seed=0)]
    task.inputs.tsnr_file = "tsnr.nii.gz"
    task.inputs.mean_file = "mean.nii.gz"
    task.inputs.stddev_file = "stdev.nii.gz"
    task.inputs.detrended_file = "detrend.nii.gz"
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
