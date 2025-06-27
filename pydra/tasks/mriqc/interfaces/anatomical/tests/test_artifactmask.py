from fileformats.generic import File
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.interfaces.anatomical.artifact_mask import ArtifactMask
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_artifactmask_1():
    task = ArtifactMask()
    task.inputs.in_file = File.sample(seed=0)
    task.inputs.head_mask = File.sample(seed=1)
    task.inputs.glabella_xyz = [0.0, 90.0, -14.0]
    task.inputs.inion_xyz = [0.0, -120.0, -14.0]
    task.inputs.ind2std_xfm = File.sample(seed=4)
    task.inputs.zscore = 10.0
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
