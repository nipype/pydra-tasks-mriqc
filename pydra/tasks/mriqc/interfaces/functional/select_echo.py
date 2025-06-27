import attrs
from fileformats.generic import File
import logging
import numpy as np
from pydra.compose import python
from pydra.utils.typing import MultiInputObj


logger = logging.getLogger(__name__)


@python.define
class SelectEcho(python.Task["SelectEcho.Outputs"]):
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.functional.select_echo import SelectEcho
    >>> from pydra.utils.typing import MultiInputObj

    """

    in_files: List
    metadata: MultiInputObj
    te_reference: float = 0.03

    class Outputs(python.Outputs):
        out_file: File
        echo_index: int
        is_multiecho: bool

    @staticmethod
    def function(
        in_files: List, metadata: MultiInputObj, te_reference: float
    ) -> tuples[File, int, bool]:
        out_file = attrs.NOTHING
        echo_index = attrs.NOTHING
        is_multiecho = attrs.NOTHING
        (
            out_file,
            echo_index,
        ) = select_echo(
            in_files,
            te_echos=(
                _get_echotime(metadata) if (metadata is not attrs.NOTHING) else None
            ),
            te_reference=te_reference,
        )
        is_multiecho = echo_index != -1

        return out_file, echo_index, is_multiecho


def _get_echotime(inlist):

    if isinstance(inlist, list):
        retval = [_get_echotime(el) for el in inlist]
        return retval[0] if len(retval) == 1 else retval
    echo_time = inlist.get("EchoTime", None) if inlist else None
    if echo_time:
        return float(echo_time)


def select_echo(
    in_files: str | list[str],
    te_echos: list[float | type(attrs.NOTHING) | None] | None = None,
    te_reference: float = 0.030,
) -> str:
    """
    Select the echo file with the closest echo time to the reference echo time.

    Used to grab the echo file when processing multi-echo data through workflows
    that only accept a single file.

    Parameters
    ----------
    in_files : :obj:`str` or :obj:`list`
        A single filename or a list of filenames.
    te_echos : :obj:`list` of :obj:`float`
        List of echo times corresponding to each file.
        If not a number (typically, a :obj:`~nipype.interfaces.base.type(attrs.NOTHING)`),
        the function selects the second echo.
    te_reference : float, optional
        Reference echo time used to find the closest echo time.

    Returns
    -------
    str
        The selected echo file.

    Examples
    --------
    >>> select_echo("single-echo.nii.gz")
    ('single-echo.nii.gz', -1)

    >>> select_echo(["single-echo.nii.gz"])
    ('single-echo.nii.gz', -1)

    >>> select_echo(
    ...     [f"echo{n}.nii.gz" for n in range(1,7)],
    ... )
    ('echo2.nii.gz', 1)

    >>> select_echo(
    ...     [f"echo{n}.nii.gz" for n in range(1,7)],
    ...     te_echos=[12.5, 28.5, 34.2, 45.0, 56.1, 68.4],
    ...     te_reference=33.1,
    ... )
    ('echo3.nii.gz', 2)

    >>> select_echo(
    ...     [f"echo{n}.nii.gz" for n in range(1,7)],
    ...     te_echos=[12.5, 28.5, 34.2, 45.0, 56.1],
    ...     te_reference=33.1,
    ... )
    ('echo2.nii.gz', 1)

    >>> select_echo(
    ...     [f"echo{n}.nii.gz" for n in range(1,7)],
    ...     te_echos=[12.5, 28.5, 34.2, 45.0, 56.1, None],
    ...     te_reference=33.1,
    ... )
    ('echo2.nii.gz', 1)

    """
    if not isinstance(in_files, (list, tuple)):
        return in_files, -1
    if len(in_files) == 1:
        return in_files[0], -1
    import numpy as np

    n_echos = len(in_files)
    if te_echos is not None and len(te_echos) == n_echos:
        try:
            index = np.argmin(np.abs(np.array(te_echos) - te_reference))
            return in_files[index], index
        except TypeError:
            pass
    return in_files[1], 1
