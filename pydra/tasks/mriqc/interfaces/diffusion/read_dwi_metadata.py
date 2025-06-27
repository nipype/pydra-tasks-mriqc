import attrs
from dipy.core.gradients import gradient_table
from dipy.stats.qc import find_qspace_neighbors
from fileformats.generic import Directory, File
import logging
from pydra.tasks.niworkflows.utils.bids import _init_layout
import numpy as np
from pydra.compose import python
import typing as ty


logger = logging.getLogger(__name__)


@python.define
class ReadDWIMetadata(python.Task["ReadDWIMetadata.Outputs"]):
    """
    Examples
    -------

    >>> from fileformats.generic import Directory, File
    >>> from pydra.tasks.mriqc.interfaces.diffusion.read_dwi_metadata import ReadDWIMetadata

    """

    in_file: File
    bids_dir: ty.Any
    bids_validate: bool = True
    index_db: Directory

    class Outputs(python.Outputs):
        out_bvec_file: File
        out_bval_file: File
        out_bmatrix: list
        qspace_neighbors: list
        out_dict: dict
        subject: str
        session: str
        task: str
        acquisition: str
        reconstruction: str
        run: int
        suffix: str

    @staticmethod
    def function(
        in_file: File, bids_dir: ty.Any, bids_validate: bool, index_db: Directory
    ) -> tuples[File, File, list, list, dict, str, str, str, str, str, int, str]:
        out_bvec_file = attrs.NOTHING
        out_bval_file = attrs.NOTHING
        out_bmatrix = attrs.NOTHING
        qspace_neighbors = attrs.NOTHING
        out_dict = attrs.NOTHING
        subject = attrs.NOTHING
        session = attrs.NOTHING
        task = attrs.NOTHING
        acquisition = attrs.NOTHING
        reconstruction = attrs.NOTHING
        run = attrs.NOTHING
        suffix = attrs.NOTHING
        self_dict = {}
        from bids.utils import listify

        self_dict["_fields"] = listify(fields or [])
        self_dict["_undef_fields"] = undef_fields
        self_dict = {}
        runtime = niworkflows_interfaces_bids__ReadSidecarJSON___run_interface(runtime)

        out_bvec_file = str(self_dict["layout"].get_bvec(in_file))
        out_bval_file = str(self_dict["layout"].get_bval(in_file))

        bvecs = np.loadtxt(out_bvec_file).T
        bvals = np.loadtxt(out_bval_file)

        gtab = gradient_table(bvals, bvecs=bvecs)

        qspace_neighbors = find_qspace_neighbors(gtab)
        out_bmatrix = np.hstack((bvecs, bvals[:, np.newaxis])).tolist()

        return (
            out_bvec_file,
            out_bval_file,
            out_bmatrix,
            qspace_neighbors,
            out_dict,
            subject,
            session,
            task,
            acquisition,
            reconstruction,
            run,
            suffix,
        )


def niworkflows_interfaces_bids__ReadSidecarJSON___run_interface():
    self_dict = {}
    self_dict["layout"] = bids_dir or self_dict["layout"]
    self_dict["layout"] = _init_layout(
        in_file,
        self_dict["layout"],
        bids_validate,
        database_path=(index_db if (index_db is not attrs.NOTHING) else None),
    )

    output_keys = list(_BIDSInfoOutputSpec().get().keys())
    params = self_dict["layout"].parse_file_entities(in_file)
    self_dict["_results"] = {
        key: params.get(key.split("_")[0], type(attrs.NOTHING)) for key in output_keys
    }

    metadata = self_dict["layout"].get_metadata(in_file)
    out_dict = metadata

    for fname in self_dict["_fields"]:
        if not self_dict["_undef_fields"] and fname not in metadata:
            raise KeyError(
                'Metadata field "%s" not found for file %s' % (fname, in_file)
            )
        self_dict["_results"][fname] = metadata.get(fname, type(attrs.NOTHING))
