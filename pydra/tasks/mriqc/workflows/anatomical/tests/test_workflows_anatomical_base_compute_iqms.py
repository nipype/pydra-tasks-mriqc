import logging
from pydra.compose import workflow
from pydra.tasks.mriqc.workflows.anatomical.base import compute_iqms
import pytest


logger = logging.getLogger(__name__)


def test_compute_iqms_build():
    workflow = compute_iqms()
    assert isinstance(workflow, Workflow)


@pytest.mark.skip(
    reason="Appropriate inputs for this workflow haven't been specified yet"
)
def test_compute_iqms_run():
    workflow = compute_iqms()
    result = workflow(worker="debug")
    print(result.out)
