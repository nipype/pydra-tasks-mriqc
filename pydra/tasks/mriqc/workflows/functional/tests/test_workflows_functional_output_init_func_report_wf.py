import logging
from pydra.compose import workflow
from pydra.tasks.mriqc.workflows.functional.output import init_func_report_wf
import pytest


logger = logging.getLogger(__name__)


def test_init_func_report_wf_build():
    workflow = init_func_report_wf()
    assert isinstance(workflow, Workflow)


@pytest.mark.skip(
    reason="Appropriate inputs for this workflow haven't been specified yet"
)
def test_init_func_report_wf_run():
    workflow = init_func_report_wf()
    result = workflow(worker="debug")
    print(result.out)
