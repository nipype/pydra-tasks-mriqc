import logging
from pydra.compose import workflow
from pydra.tasks.mriqc.workflows.functional.base import hmc
import pytest


logger = logging.getLogger(__name__)


def test_hmc_build():
    workflow = hmc()
    assert isinstance(workflow, Workflow)


@pytest.mark.skip(
    reason="Appropriate inputs for this workflow haven't been specified yet"
)
def test_hmc_run():
    workflow = hmc()
    result = workflow(worker="debug")
    print(result.out)
