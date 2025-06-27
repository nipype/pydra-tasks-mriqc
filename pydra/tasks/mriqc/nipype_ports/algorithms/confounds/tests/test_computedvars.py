from fileformats.generic import File
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.nipype_ports.algorithms.confounds.compute_dvars import (
    ComputeDVARS,
)
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_computedvars_1():
    task = ComputeDVARS()
    task.inputs.in_file = File.sample(seed=0)
    task.inputs.in_mask = File.sample(seed=1)
    task.inputs.remove_zerovariance = True
    task.inputs.variance_tol = 1e-07
    task.inputs.save_std = True
    task.inputs.save_nstd = False
    task.inputs.save_vxstd = False
    task.inputs.save_all = False
    task.inputs.save_plot = False
    task.inputs.figdpi = 100
    task.inputs.figsize = [11.7, 2.3]
    task.inputs.figformat = "png"
    task.inputs.intensity_normalization = 1000.0
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
