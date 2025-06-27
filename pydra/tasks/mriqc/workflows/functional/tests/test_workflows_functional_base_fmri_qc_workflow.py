import logging
from pydra.compose import workflow
from pydra.tasks.mriqc.workflows.functional.base import fmri_qc_workflow
import pytest


logger = logging.getLogger(__name__)


def test_fmri_qc_workflow_build():
    workflow = fmri_qc_workflow()
    assert isinstance(workflow, Workflow)


@pytest.mark.skip(
    reason="Appropriate inputs for this workflow haven't been specified yet"
)
def test_fmri_qc_workflow_run():
    workflow = fmri_qc_workflow()
    result = workflow(worker="debug")
    print(result.out)
