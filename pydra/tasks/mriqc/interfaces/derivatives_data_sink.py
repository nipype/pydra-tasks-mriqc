import attrs
from contextlib import suppress
from fileformats.generic import Directory
from json import dumps
import logging
import nibabel as nb
from pydra.tasks.niworkflows import data
from pydra.tasks.niworkflows.utils.bids import relative_to_root
from pydra.tasks.niworkflows.utils.images import (
    set_consumables,
    unsafe_write_nifti_header_and_data,
)
from pydra.tasks.niworkflows.utils.misc import _copy_any, unlink
import numpy as np
import os
from pathlib import Path
from pydra.compose import python
from pydra.utils.typing import MultiInputObj
import re


logger = logging.getLogger(__name__)


@python.define
class DerivativesDataSink(python.Task["DerivativesDataSink.Outputs"]):
    """
    Examples
    -------

    >>> from fileformats.generic import Directory, File
    >>> from pydra.tasks.mriqc.interfaces.derivatives_data_sink import DerivativesDataSink
    >>> from pydra.utils.typing import MultiInputObj, MultiOutputType

    """

    base_directory: Directory
    check_hdr: bool = True
    compress: MultiInputObj = []
    data_dtype: str
    dismiss_entities: MultiInputObj = []
    in_file: List
    meta_dict: dict
    source_file: List

    class Outputs(python.Outputs):
        out_file: List
        out_meta: List
        compression: Union
        fixed_hdr: list

    @staticmethod
    def function(
        base_directory: Directory,
        check_hdr: bool,
        compress: MultiInputObj,
        data_dtype: str,
        dismiss_entities: MultiInputObj,
        in_file: List,
        meta_dict: dict,
        source_file: List,
    ) -> tuples[List, List, Union, list]:
        out_file = attrs.NOTHING
        out_meta = attrs.NOTHING
        compression = attrs.NOTHING
        fixed_hdr = attrs.NOTHING
        self_dict = {}
        """Initialize the SimpleInterface and extend inputs with custom entities."""
        self_dict["_allowed_entities"] = set(allowed_entities or []).union(
            set(self_dict["_config_entities"])
        )
        if out_path_base:
            self_dict["out_path_base"] = out_path_base

        self_dict["_metadata"] = {}
        self_dict["_static_traits"] = self_dict[
            "input_spec"
        ].class_editable_traits() + sorted(self_dict["_allowed_entities"])
        for dynamic_input in set(inputs) - set(self_dict["_static_traits"]):
            self_dict["_metadata"][dynamic_input] = inputs.pop(dynamic_input)

        add_traits(self_dict["inputs"], self_dict["_allowed_entities"])
        for k in self_dict["_allowed_entities"].intersection(list(inputs.keys())):

            setattr(self_dict["inputs"], k, inputs[k])
        self_dict = {}
        from bids.layout import parse_file_entities, Config
        from bids.layout.writing import build_path
        from bids.utils import listify

        base_directory = os.getcwd()
        if base_directory is not attrs.NOTHING:
            base_directory = base_directory
        base_directory = Path(base_directory).absolute()
        out_path = base_directory / self_dict["out_path_base"]
        out_path.mkdir(exist_ok=True, parents=True)

        in_file = listify(in_file)

        if meta_dict is not attrs.NOTHING:
            meta = meta_dict

            meta.update(self_dict["_metadata"])
            self_dict["_metadata"] = meta

        custom_config = Config(
            name="custom",
            entities=self_dict["_config_entities_dict"],
            default_path_patterns=self_dict["_file_patterns"],
        )
        in_entities = [
            parse_file_entities(
                str(relative_to_root(source_file)),
                config=["bids", "derivatives", custom_config],
            )
            for source_file in source_file
        ]
        out_entities = {
            k: v
            for k, v in in_entities[0].items()
            if all(ent.get(k) == v for ent in in_entities[1:])
        }
        for drop_entity in listify(dismiss_entities or []):
            out_entities.pop(drop_entity, None)

        out_entities["extension"] = [
            "".join(Path(orig_file).suffixes).lstrip(".") for orig_file in in_file
        ]

        compress = listify(compress) or [None]
        if len(compress) == 1:
            compress = compress * len(in_file)
        for i, ext in enumerate(out_entities["extension"]):
            if compress[i] is not None:
                ext = regz.sub("", ext)
                out_entities["extension"][i] = f"{ext}.gz" if compress[i] else ext

        for key in self_dict["_allowed_entities"]:
            value = getattr(self_dict["inputs"], key)
            if value is not None and (value is not attrs.NOTHING):
                out_entities[key] = value

        if out_entities.get("resolution") == "native" and out_entities.get("space"):
            out_entities.pop("resolution", None)

        resolution = out_entities.get("resolution")
        space = out_entities.get("space")
        if resolution:

            if space in self_dict["_standard_spaces"]:
                res = _get_tf_resolution(space, resolution)
            else:  # TODO: Nonstandard?
                res = "Unknown"
            self_dict["_metadata"]["Resolution"] = res

        if len(set(out_entities["extension"])) == 1:
            out_entities["extension"] = out_entities["extension"][0]

        custom_entities = set(out_entities) - set(self_dict["_config_entities"])
        patterns = self_dict["_file_patterns"]
        if custom_entities:

            custom_pat = "_".join(f"{key}-{{{key}}}" for key in sorted(custom_entities))
            patterns = [
                pat.replace("_{suffix", "_".join(("", custom_pat, "{suffix")))
                for pat in patterns
            ]

        out_file = []
        compression = []
        fixed_hdr = [False] * len(in_file)

        dest_files = build_path(out_entities, path_patterns=patterns)
        if not dest_files:
            raise ValueError(f"Could not build path with entities {out_entities}.")

        dest_files = listify(dest_files)
        if len(in_file) != len(dest_files):
            raise ValueError(
                f"Input files ({len(in_file)}) not matched "
                f"by interpolated patterns ({len(dest_files)})."
            )

        for i, (orig_file, dest_file) in enumerate(zip(in_file, dest_files)):
            out_file = out_path / dest_file
            out_file.parent.mkdir(exist_ok=True, parents=True)
            out_file.append(str(out_file))
            compression.append(str(dest_file).endswith(".gz"))

            try:
                if os.path.samefile(orig_file, out_file):
                    continue
            except FileNotFoundError:
                pass

            new_data, new_header = None, None

            is_nifti = False
            with suppress(nb.filebasedimages.ImageFileError):
                is_nifti = isinstance(nb.load(orig_file), nb.Nifti1Image)

            data_dtype = data_dtype or self_dict["_default_dtypes"][suffix]
            if is_nifti and any((check_hdr, data_dtype)):
                nii = nb.load(orig_file)

                if check_hdr:
                    hdr = nii.header
                    curr_units = tuple(
                        [None if u == "unknown" else u for u in hdr.get_xyzt_units()]
                    )
                    curr_codes = (int(hdr["qform_code"]), int(hdr["sform_code"]))

                    units = (
                        curr_units[0] or "mm",
                        "sec" if out_entities["suffix"] == "bold" else None,
                    )
                    xcodes = (1, 1)  # Derivative in its original scanner space
                    if space:
                        xcodes = (
                            (4, 4) if space in self_dict["_standard_spaces"] else (2, 2)
                        )

                    curr_zooms = zooms = hdr.get_zooms()
                    if "RepetitionTime" in self_dict["inputs"].get():
                        zooms = curr_zooms[:3] + (RepetitionTime,)

                    if (curr_codes, curr_units, curr_zooms) != (xcodes, units, zooms):
                        fixed_hdr[i] = True
                        new_header = hdr.copy()
                        new_header.set_qform(nii.affine, xcodes[0])
                        new_header.set_sform(nii.affine, xcodes[1])
                        new_header.set_xyzt_units(*units)
                        new_header.set_zooms(zooms)

                if data_dtype == "source":  # match source dtype
                    try:
                        data_dtype = nb.load(source_file[0]).get_data_dtype()
                    except Exception:
                        LOGGER.warning(
                            f"Could not get data type of file {source_file[0]}"
                        )
                        data_dtype = None

                if data_dtype:
                    data_dtype = np.dtype(data_dtype)
                    orig_dtype = nii.get_data_dtype()
                    if orig_dtype != data_dtype:
                        LOGGER.warning(
                            f"Changing {out_file} dtype from {orig_dtype} to {data_dtype}"
                        )

                        if np.issubdtype(data_dtype, np.integer):
                            new_data = np.rint(nii.dataobj).astype(data_dtype)
                        else:
                            new_data = np.asanyarray(nii.dataobj, dtype=data_dtype)

                        if new_header is None:
                            new_header = nii.header.copy()
                        new_header.set_data_dtype(data_dtype)
                del nii

            unlink(out_file, missing_ok=True)
            if new_data is new_header is None:
                _copy_any(orig_file, str(out_file))
            else:
                orig_img = nb.load(orig_file)
                if new_data is None:
                    set_consumables(new_header, orig_img.dataobj)
                    new_data = orig_img.dataobj.get_unscaled()
                else:

                    new_header.set_slope_inter(slope=1.0, inter=0.0)
                unsafe_write_nifti_header_and_data(
                    fname=out_file, header=new_header, data=new_data
                )
                del orig_img

        if len(out_file) == 1:
            meta_fields = self_dict["inputs"].copyable_trait_names()
            self_dict["_metadata"].update(
                {
                    k: getattr(self_dict["inputs"], k)
                    for k in meta_fields
                    if k not in self_dict["_static_traits"]
                }
            )
            if self_dict["_metadata"]:
                sidecar = out_file.parent / f"{out_file.name.split('.', 1)[0]}.json"
                unlink(sidecar, missing_ok=True)
                sidecar.write_text(
                    dumps(self_dict["_metadata"], sort_keys=True, indent=2)
                )
                out_meta = str(sidecar)

        return out_file, out_meta, compression, fixed_hdr


LOGGER = logging.getLogger("nipype.interface")

regz = re.compile(r"\.gz$")
