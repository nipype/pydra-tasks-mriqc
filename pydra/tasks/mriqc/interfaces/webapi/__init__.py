import attrs
from fileformats.generic import Directory, File
import json
import logging
from pydra.tasks.mriqc import messages
import orjson
from pathlib import Path
from pydra.compose import python, shell, workflow
from .upload_iq_ms import UploadIQMs
from pydra.utils.typing import MultiInputObj
import typing as ty
import yaml


logger = logging.getLogger(__name__)


def _hashfields(data):

    from hashlib import sha256

    for name in HASH_BIDS:
        if name in data:
            data[name] = sha256(data[name].encode()).hexdigest()
    return data


def upload_qc_metrics(
    in_iqms, endpoint=None, email=None, auth_token=None, modality=None
):
    """
    Upload qc metrics to remote repository.

    :param str in_iqms: Path to the qc metric json file as a string
    :param str webapi_url: the protocol (either http or https)
    :param str email: email address to be included with the metric submission
    :param str auth_token: authentication token

    :return: either the response object if a response was successfully sent
             or it returns the string "No Response"
    :rtype: object


    """
    from copy import deepcopy
    import requests

    if not endpoint or not auth_token:
        # If endpoint unknown, do not even report what happens to the token.
        errmsg = "Unknown API endpoint" if not endpoint else "Authentication failed."
        return Bunch(status_code=1, text=errmsg)
    in_data = orjson.loads(Path(in_iqms).read_bytes())
    # Extract metadata and provenance
    meta = in_data.pop("bids_meta")
    prov = in_data.pop("provenance")
    # At this point, data should contain only IQMs
    data = deepcopy(in_data)
    # Check modality
    modality = meta.get("modality", None) or meta.get("suffix", None) or modality
    if modality not in ("T1w", "bold", "T2w"):
        errmsg = (
            'Submitting to MRIQCWebAPI: image modality should be "bold", "T1w", or "T2w", '
            f'(found "{modality}")'
        )
        return Bunch(status_code=1, text=errmsg)
    # Filter metadata values that aren't in whitelist
    data["bids_meta"] = {k: meta[k] for k in META_WHITELIST if k in meta}
    # Check for fields with appended _id
    bids_meta_names = {
        k: k.replace("_id", "") for k in META_WHITELIST if k.endswith("_id")
    }
    data["bids_meta"].update(
        {k: meta[v] for k, v in bids_meta_names.items() if v in meta}
    )
    # For compatibility with WebAPI. Should be rolled back to int
    if (run_id := data["bids_meta"].get("run_id", None)) is not None:
        data["bids_meta"]["run_id"] = f"{run_id}"
    # One more chance for spelled-out BIDS entity acquisition
    if (acq_id := meta.get("acquisition", None)) is not None:
        data["bids_meta"]["acq_id"] = acq_id
    # Filter provenance values that aren't in whitelist
    data["provenance"] = {k: prov[k] for k in PROV_WHITELIST if k in prov}
    # Hash fields that may contain personal information
    data["bids_meta"] = _hashfields(data["bids_meta"])
    data["bids_meta"]["modality"] = modality
    if email:
        data["provenance"]["email"] = email
    headers = {"Authorization": auth_token, "Content-Type": "application/json"}
    start_message = messages.QC_UPLOAD_START.format(url=endpoint)
    logger.info(start_message)
    errmsg = None
    try:
        # if the modality is bold, call "bold" endpoint
        response = requests.post(
            f"{endpoint}/{modality}",
            headers=headers,
            data=orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY),
            timeout=15,
        )
    except requests.ConnectionError as err:
        errmsg = ("Error uploading IQMs: Connection error:", f"{err}")
    except requests.exceptions.ReadTimeout as err:
        errmsg = (f"Error uploading IQMs: Server {endpoint} is down.", f"{err}")
    if errmsg is not None:
        response = Bunch(status_code=1, text="\n".join(errmsg))
    return response, data


HASH_BIDS = ["subject_id", "session_id"]

META_WHITELIST = [
    "AccelNumReferenceLines",
    "AccelerationFactorPE",
    "AcquisitionMatrix",
    "CogAtlasID",
    "CogPOID",
    "CoilCombinationMethod",
    "ContrastBolusIngredient",
    "ConversionSoftware",
    "ConversionSoftwareVersion",
    "DelayTime",
    "DeviceSerialNumber",
    "EchoTime",
    "EchoTrainLength",
    "EffectiveEchoSpacing",
    "FlipAngle",
    "GradientSetType",
    "HardcopyDeviceSoftwareVersion",
    "ImageType",
    "ImagingFrequency",
    "InPlanePhaseEncodingDirection",
    "InstitutionAddress",
    "InstitutionName",
    "Instructions",
    "InversionTime",
    "MRAcquisitionType",
    "MRTransmitCoilSequence",
    "MagneticFieldStrength",
    "Manufacturer",
    "ManufacturersModelName",
    "MatrixCoilMode",
    "MultibandAccelerationFactor",
    "NumberOfAverages",
    "NumberOfPhaseEncodingSteps",
    "NumberOfVolumesDiscardedByScanner",
    "NumberOfVolumesDiscardedByUser",
    "NumberShots",
    "ParallelAcquisitionTechnique",
    "ParallelReductionFactorInPlane",
    "PartialFourier",
    "PartialFourierDirection",
    "PatientPosition",
    "PercentPhaseFieldOfView",
    "PercentSampling",
    "PhaseEncodingDirection",
    "PixelBandwidth",
    "ProtocolName",
    "PulseSequenceDetails",
    "PulseSequenceType",
    "ReceiveCoilName",
    "RepetitionTime",
    "ScanOptions",
    "ScanningSequence",
    "SequenceName",
    "SequenceVariant",
    "SliceEncodingDirection",
    "SoftwareVersions",
    "TaskDescription",
    "TaskName",
    "TotalReadoutTime",
    "TotalScanTimeSec",
    "TransmitCoilName",
    "VariableFlipAngleFlag",
    "acq_id",
    "modality",
    "run_id",
    "subject_id",
    "task_id",
    "session_id",
]

PROV_WHITELIST = ["version", "md5sum", "software", "settings"]
