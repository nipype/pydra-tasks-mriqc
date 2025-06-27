from fileformats.generic import File
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.interfaces.anatomical.structural_qc import StructuralQC
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_structuralqc_1():
    task = StructuralQC()
    task.inputs.in_file = File.sample(seed=0)
    task.inputs.in_noinu = File.sample(seed=1)
    task.inputs.in_segm = File.sample(seed=2)
    task.inputs.in_bias = File.sample(seed=3)
    task.inputs.head_msk = File.sample(seed=4)
    task.inputs.air_msk = File.sample(seed=5)
    task.inputs.rot_msk = File.sample(seed=6)
    task.inputs.artifact_msk = File.sample(seed=7)
    task.inputs.in_pvms = [File.sample(seed=8)]
    task.inputs.in_tpms = [File.sample(seed=9)]
    task.inputs.mni_tpms = [File.sample(seed=10)]
    task.inputs.human = True
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
