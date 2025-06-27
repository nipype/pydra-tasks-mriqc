import attrs
from fileformats.generic import File
import json
import logging
from pydra.tasks.mriqc.utils.misc import BIDS_COMP
import orjson as json
from pathlib import Path
from pydra.compose import python
import typing as ty


logger = logging.getLogger(__name__)


@python.define
class IQMFileSink(python.Task["IQMFileSink.Outputs"]):
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.bids.iqm_file_sink import IQMFileSink

    """

    in_file: str
    modality: str
    entities: dict
    subject_id: str
    session_id: ty.Any
    task_id: ty.Any
    acq_id: ty.Any
    rec_id: ty.Any
    run_id: ty.Any
    dataset: str
    dismiss_entities: list = ["datatype", "part", "echo", "extension", "suffix"]
    metadata: dict
    provenance: dict
    root: dict
    out_dir: Path
    _outputs: dict = {}

    class Outputs(python.Outputs):
        out_file: File

    @staticmethod
    def function(
        in_file: str,
        modality: str,
        entities: dict,
        subject_id: str,
        session_id: ty.Any,
        task_id: ty.Any,
        acq_id: ty.Any,
        rec_id: ty.Any,
        run_id: ty.Any,
        dataset: str,
        dismiss_entities: list,
        metadata: dict,
        provenance: dict,
        root: dict,
        out_dir: Path,
        _outputs: dict,
    ) -> File:
        out_file = attrs.NOTHING
        self_dict = {}

        if fields is None:
            fields = []

        self_dict["_out_dict"] = {}

        fields = list(set(fields) - set(self_dict["inputs"].copyable_trait_names()))
        self_dict["_input_names"] = fields
        undefined_traits = {
            key: _add_field(key, _outputs=_outputs, add_trait=add_trait)
            for key in fields
        }
        self_dict["inputs"].trait_set(trait_change_notify=False, **undefined_traits)

        if force_run:
            self_dict["_always_run"] = True
        self_dict = {}
        out_file = _gen_outfile(
            dismiss_entities=dismiss_entities, out_dir=out_dir, in_file=in_file
        )

        if root is not attrs.NOTHING:
            self_dict["_out_dict"] = root

        root_adds = []
        for key, val in list(_outputs.items()):
            if (val is attrs.NOTHING) or key == "trait_added":
                continue

            if self_dict["expr"].match(key) is not None:
                root_adds.append(key)
                continue

            key, val = _process_name(key, val)
            self_dict["_out_dict"][key] = val

        for root_key in root_adds:
            val = _outputs.get(root_key, None)
            if isinstance(val, dict):
                self_dict["_out_dict"].update(val)
            else:
                logger.warning(
                    'Output "%s" is not a dictionary (value="%s"), discarding output.',
                    root_key,
                    str(val),
                )

        id_dict = entities if (entities is not attrs.NOTHING) else {}
        for comp in BIDS_COMP:
            comp_val = getattr(self_dict["inputs"], comp, None)
            if (comp_val is not attrs.NOTHING) and comp_val is not None:
                id_dict[comp] = comp_val
        id_dict["modality"] = modality

        if (metadata is not attrs.NOTHING) and metadata:
            id_dict.update(metadata)

        if self_dict["_out_dict"].get("bids_meta") is None:
            self_dict["_out_dict"]["bids_meta"] = {}
        self_dict["_out_dict"]["bids_meta"].update(id_dict)

        if dataset is not attrs.NOTHING:
            self_dict["_out_dict"]["bids_meta"]["dataset"] = dataset

        prov_dict = {}
        if (provenance is not attrs.NOTHING) and provenance:
            prov_dict.update(provenance)

        if self_dict["_out_dict"].get("provenance") is None:
            self_dict["_out_dict"]["provenance"] = {}
        self_dict["_out_dict"]["provenance"].update(prov_dict)

        Path(out_file).write_bytes(
            json.dumps(
                self_dict["_out_dict"],
                option=(
                    json.OPT_SORT_KEYS
                    | json.OPT_INDENT_2
                    | json.OPT_APPEND_NEWLINE
                    | json.OPT_SERIALIZE_NUMPY
                ),
            )
        )

        return out_file


def _add_field(name, value=attrs.NOTHING, _outputs=None, add_trait=None):
    self_dict = {}
    self_dict["inputs"].add_trait(name, traits.Any)
    _outputs[name] = value
    return value


def _gen_outfile(dismiss_entities=None, out_dir=None, in_file=None):
    out_dir = Path()
    if out_dir is not attrs.NOTHING:
        out_dir = Path(out_dir)

    path = Path(in_file)
    for i in range(1, 4):
        if str(path.parents[i].name).startswith("sub-"):
            bids_root = path.parents[i + 1]
            break
    in_file = str(path.relative_to(bids_root))

    if (dismiss_entities is not attrs.NOTHING) and (dismiss := dismiss_entities):
        for entity in dismiss:
            bids_chunks = [
                chunk
                for chunk in path.name.split("_")
                if not chunk.startswith(f"{entity}-")
            ]
            path = path.parent / "_".join(bids_chunks)

    bids_path = out_dir / in_file.replace("".join(Path(in_file).suffixes), ".json")
    bids_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = str(bids_path)
    return out_file


def _process_name(name, val):

    if "." in name:
        newkeys = name.split(".")
        name = newkeys.pop(0)
        nested_dict = {newkeys.pop(): val}
        for nk in reversed(newkeys):
            nested_dict = {nk: nested_dict}
        val = nested_dict
    return name, val
