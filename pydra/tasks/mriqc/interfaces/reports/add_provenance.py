import attrs
from fileformats.generic import File
import logging
import nibabel as nb
import numpy as np
from pydra.compose import python


logger = logging.getLogger(__name__)


@python.define
class AddProvenance(python.Task["AddProvenance.Outputs"]):
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.reports.add_provenance import AddProvenance

    """

    in_file: File
    air_msk: File
    rot_msk: File
    modality: str

    class Outputs(python.Outputs):
        out_prov: dict

    @staticmethod
    def function(in_file: File, air_msk: File, rot_msk: File, modality: str) -> dict:
        out_prov = attrs.NOTHING
        from nipype.utils.filemanip import hash_infile

        out_prov = {
            "md5sum": hash_infile(in_file),
            "version": '<version-not-captured>',
            "software": "mriqc",
            "settings": {
                "testing": False,
            },
        }

        if modality in ("T1w", "T2w"):
            air_msk_size = np.asanyarray(nb.load(air_msk).dataobj).astype(bool).sum()
            rot_msk_size = np.asanyarray(nb.load(rot_msk).dataobj).astype(bool).sum()
            out_prov["warnings"] = {
                "small_air_mask": bool(air_msk_size < 5e5),
                "large_rot_frame": bool(rot_msk_size > 500),
            }

        if modality == "bold":
            out_prov["settings"].update(
                {
                    "fd_thres": 0.2,  # <configuration>.fd_thres
                }
            )

        return out_prov
